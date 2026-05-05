# OmniVoice Perf — A100

NVIDIA A100 80GB PCIe (sm80, driver 535.183.06), torch 2.10.0+cu128, vllm 0.19.0,
vllm-omni @ `0.1.dev1457+g46eaa290f`. Branch `a100-benchmarking-and-optimizations`.
NUM_STEP=32 throughout. Single sweep per config (no median-of-3); run-to-run
variance is non-trivial at high concurrency, treat ±5% as noise.

## Recommended config

```bash
VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket
VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=256
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=30          # was 0; this is the unlock
VLLM_OMNI_DIFFUSION_BATCH_SIZE=12             # was 32; cap to avoid runaway batches
VLLM_OMNI_DIFFUSION_CONCURRENT=1
VLLM_OMNI_OMNIVOICE_OPT=1
VLLM_OMNI_OMNIVOICE_COMPILE_MODE=default
VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bfloat16
VLLM_OMNI_OMNIVOICE_NUM_STEP=32
# fa3-fwd skipped in requirements/cuda.txt: Hopper-only kernel
```

## Headline numbers

```
recommended config (bucket=256, wait=30ms, cap=12)         vs prior baseline
 c    req/s   wall_avg   wall_p95                            req/s    delta
 1    2.72    0.339 s    0.349 s                             2.77      -2%   (noise)
 2    4.55    0.377 s    0.443 s                             3.06     +49%   ✓
 4    5.97    0.514 s    0.764 s                             4.95     +21%   ✓
 8    6.63    0.872 s    1.228 s                             6.07     +9%    ✓
 16   5.94    1.655 s    2.754 s                             5.79     +3%    wall ↓17%
 32   5.24    3.015 s    4.854 s                             5.54     -5%    (noise)
```

Wins large at c=2..8 (the typical production range), within noise at c=1 and c=32.

## Why these settings — investigation

### The problem

The "morning baseline" config (`bucket=128, wait=0, batch_size=32`) plateaus at
~6 req/s from c=8 upward. Higher concurrency just queues — doesn't go faster.
Adding a single-line log of `B = len(target_lens)` in
`omnivoice_generator.py::forward` exposed the cause:

```
                       forward calls   B histogram
c=1,4,8,16 mixed sweep        73       66×B=1, 4×B=5, 2×B=2, 1×B=11
c=8 only                      16        9×B=1, 3×B=2, 2×B=6, 1×B=7, 1×B=4
```

90% of forward calls run **solo**. The duration_bucket scheduler anchors on
the first waiting request, drains only same-bucket peers (within 128 tokens
of estimated audio length), and dispatches. With BATCH_WAIT_MS=0, the wait
queue is sampled exactly once per dispatch, so peers that arrive 1-2 ms
later are never merged.

### The strategy sweep at c=8

```
config              strategy         bucket  wait  conc  cap   req/s  wall_avg  p95     dominant Bs
A baseline          duration_bucket   128     0    1     32    5.87   1.02 s   1.38    1, 6, 7
B bucket=256        duration_bucket   256     0    1     32    5.82   1.03 s   1.35    1, 7
C bucket=512        duration_bucket   512     0    1     32    5.55   1.16 s   1.62    1, 7
D fifo              fifo               -      0    1     32    2.70   2.84 s   8.95    1, 7  (tail blowup)
E wait30 + b128     duration_bucket   128    30    1     32    6.62   0.90 s   1.44    1, 6, 7, 8
F wait30 + b256     duration_bucket   256    30    1     32    6.60   0.87 s   1.22    1, 5, 8
G concurrent=2      duration_bucket   128     0    2     32    2.97   2.43 s   2.49    100% B=1 (broken)
F12 winner          duration_bucket   256    30    1     12    6.63   0.87 s   1.23    1, 4, 8, 12
```

Key observations:
- **WAIT=30 ms is the unlock.** E and F both gain ~13% over baseline.
  Without wait, by the time the scheduler peeks the queue most peers
  haven't arrived yet — even at c=8 where the asyncio client launches
  8 requests in quick succession.
- **Bucket width is secondary.** 128 → 256 ties on throughput; 256 has
  marginally better p95 because slightly more peers fit. 512 starts to
  lose (longer prompts mixed in lift batch wall).
- **FIFO without bucketing is a tail-latency disaster.** Same merge
  pattern as bucketed configs but p95 = 8.9 s because the longest prompt
  in a B=7 batch dictates wall for all 7.
- **CONCURRENT=2 is broken**, not just slower. Every forward was B=1.
  `base_scheduler.py:134` gates batch formation on `not self._running`,
  so with 2 in-flight slots a new merged batch never forms — the
  scheduler degenerates to dispatching solo requests as they arrive.
- **BATCH_SIZE=32 lets the cap go too high.** F's first c=32 validation
  produced a B=29 batch, which OOMs A100 compute (wall_avg 6.5 s, +30%
  regression vs baseline at c=32). Capping to B=12 keeps batches in the
  scaling regime A100 handles well.

### B-histogram across full c=1..32 sweep with the recommended config

```
   50  B=1
   19  B=2
    7  B=4
    3  B=8
    3  B=12
    2  B=5
    1  B=10
    1  B=7
```

35% of calls still run solo (mostly c=1 traffic and odd-bucket prompts at the
end of waves), but the rest land in B=2..12 range — exactly the regime A100
runs efficiently.

## What didn't pay off

### Cudagraphs (`COMPILE_MODE=reduce-overhead`)

Hypothesis: A100's higher kernel-launch overhead per call (vs H100) should
make cudagraphs a bigger win. Test result: **regressed at every concurrency**.

```
 c    F12     F12+cudagraphs    delta
 1    2.72    2.32              -15%
 2    4.55    3.52              -23%
 4    5.97    4.46              -25%
 8    6.63    5.82              -12%
 16   5.94    5.26              -11%
 32   5.24    5.01               -4%
```

Two compounding losses:

1. **Graph partitioning** — server log: `cudagraph partition due to non gpu ops`.
   torch.compile broke the graph into 2 partitions because of CPU sync points
   (`.item()` calls or device-copies). Half the kernel-launch savings lost.
2. **Dynamic shape thrash** — torch warned "observed 9 distinct sizes" during
   the sweep. Each new (B, S) triggers recapture. With duration_bucket producing
   variable B and S, capture cost dominates.

For cudagraphs to pay off, we'd need either:
- Padding inputs to a small fixed set of (B, S) shapes
- Pre-warming common shapes at server startup (`PREWARM_BUCKETS`, currently
  crashes per H100 doc)

Skipped in favor of cheaper wins.

### Padding-tolerance fold-in (`VLLM_OMNI_DIFFUSION_PAD_TOLERANCE`)

Hypothesis: relaxing the strict same-bucket constraint to a max/min-token
ratio bound would capture singleton-bucket prompts that currently run solo.

The knob is wired (in `base_scheduler.py`, default 1.0 = strict bucket
equality, opt-in via `VLLM_OMNI_DIFFUSION_PAD_TOLERANCE > 1.0`). But it
didn't pay off at our test set's prompt distribution:

```
                tolerance=1.0   tol=1.2   tol=1.5
 c=4   req/s    5.97            5.09      5.89
 c=8   req/s    6.63            5.37      5.70
 c=16  req/s    5.94            5.24      5.34
```

Why:
- bucket=256 already has an *implicit* ratio tolerance ~1.4 inside a single
  bucket (e.g. bucket-0 holds T ∈ [50, 255] → ratio up to 5×, but in practice
  prompts cluster naturally around 1.4×). Tolerance=1.2 is *tighter* than the
  implicit constraint → fewer merges, smaller batches.
- Tolerance=1.5 does fold in cross-bucket peers (B=9 batches were observed
  for the first time), but FMHA cost ~ S²: padding from S=120 to S=180 (1.5×)
  costs +125% compute per slot. The ~2 ms of launch overhead saved per merged
  call is dwarfed by the padding tax.

The headroom from "fixing solo calls" is also smaller than the histogram
suggested. Of 8 B=1 calls per c=8 wave, ~half are intrinsic — the 1374-token
outlier in the test set ALWAYS solos, plus a few prompts at hard bucket
edges. Realistic ceiling from fold-in is ~30%, not 100%, and the padding
tax eats most of it.

The knob stays in code (default off) for future use if a different prompt
distribution makes the math flip.

## Things still on the table

1. **`NUM_STEP=32 → 16`** — biggest untested lever, ~2× wall reduction
   mechanically. Doc claims it works numerically but quality has never been
   A/B'd. Single biggest potential win on this branch.
2. **flash-attn 2.x for sm80** — `fa3-fwd` doesn't build on A100, but
   `flash-attn` 2.x does. The H100 verdict that varlen packing exceeds
   padding savings was integration-driven (graph breaks, per-call
   pack/unpack); a clean integration could be a real win on A100 since
   the implicit padding tax there is larger.
3. **`scheduler._running` gate fix** — `base_scheduler.py:134` blocks
   batch formation while one is in flight, so CONCURRENT≥2 silently
   regresses to 100% B=1. Worth investigating whether the gate is
   fundamentally needed or a vestige; removing it could enable real CPU/GPU
   overlap.
4. **DAC decoder bf16** — currently fp32, ~9% of c=1 wall. Halving = ~4-5%.
   Free if quality holds.
5. **Cudagraphs with shape-anchored padding** — pad (B, S) to a fixed set
   of {(1, 256), (4, 256), (8, 256), (1, 512), ...}. Captures the kernel-launch
   savings without dynamic-shape recapture penalty. Bigger lift but is the
   only path to making cudagraphs work alongside dynamic batching.

## Reproduction

```bash
# venv setup (Python 3.12 required for StrEnum)
python3.12 -m venv .venv
.venv/bin/pip install vllm==0.19.0
# requirements/cuda.txt has fa3-fwd commented out for A100; uncomment for H100
.venv/bin/pip install -e .
.venv/bin/pip install pandas datasets    # bench client deps not in cuda.txt

# Server (recommended config)
HF_TOKEN=... \
VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=256 \
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=30 \
VLLM_OMNI_DIFFUSION_BATCH_SIZE=12 \
VLLM_OMNI_OMNIVOICE_NUM_STEP=32 \
bash examples/online_serving/omnivoice/run_server_optimized.sh

# Bench
HF_TOKEN=... .venv/bin/python examples/online_serving/omnivoice/benchmark_concurrent.py \
  --concurrencies 1,2,4,8,16,32 \
  --candidate_count 32 \
  --warmup_iters 10 \
  --jitter_ms_min 0 --jitter_ms_max 0

# Strategy sweep at c=8 (writes per-config logs to /tmp/sweep/)
bash benchmarks/tts/scripts/a100_batching_sweep.sh

# To capture per-call B sizes, set on the server:
#   VLLM_OMNI_OMNIVOICE_LOG_BATCH=1
```
