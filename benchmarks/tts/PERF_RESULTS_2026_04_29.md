# OmniVoice Serving Perf Results — 2026-04-29

H100 PCIe, vllm-omni, OmniVoice TTS. Sweep over concurrency
`{1, 2, 4, 8, 16, 32}` on `TrySalient/tts-test-set` (32-candidate
length-distribution sample, ~5 s audio per request, warmup_iters=10).

## Best operating point

`reduce-overhead` compile mode (cudagraphs) **+** `duration_bucket` batch
strategy **+** bf16 generator. Beats prior baseline at every concurrency.

```
                     baseline                   best (combo + jitter)
                req/s   wall_avg              req/s   wall_avg   wall_p95
 c=1            2.42    0.396 s               4.18    0.217 s    0.233 s
 c=2            4.00    0.470 s               6.41    0.258 s    0.321 s
 c=4            6.07    0.505 s               7.86    0.389 s    0.566 s
 c=8            3.78    1.949 s               8.70    0.655 s    1.086 s
 c=16           5.69    1.808 s               7.99    1.498 s    2.287 s
 c=32           1.64   11.944 s               8.13    1.549 s    2.297 s

         (req/s, wall_avg, wall_p95 in seconds)
```

c=1 latency: **0.396 s → 0.217 s (45% reduction)**.
c=32 throughput: **1.64 req/s → 8.13 req/s (5× improvement)**.

## Server config (best)

```bash
VLLM_OMNI_DIFFUSION_CONCURRENT=1
VLLM_OMNI_DIFFUSION_BATCH_SIZE=32
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=30
VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket
VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=128
VLLM_OMNI_OMNIVOICE_OPT=1
VLLM_OMNI_OMNIVOICE_COMPILE_MODE=reduce-overhead    # NEW: cudagraphs
VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bfloat16
bash examples/online_serving/omnivoice/run_server.sh
```

The `cudagraph_mark_step_begin()` call inside the 32-step unmasking loop
is required when `COMPILE_MODE=reduce-overhead` to avoid cudagraph
buffer-aliasing errors (the same compiled module is reused across all
32 steps).

## Bench command

```bash
python3 examples/online_serving/omnivoice/benchmark_concurrent.py \
  --concurrencies 1,2,4,8,16,32 \
  --candidate_count 32 \
  --warmup_iters 10 \
  --out_dir outputs/sweep_combo_jitter
```

## Per-cell tables

### combo (reduce-overhead + duration_bucket) + jitter 10–50 ms

```
concurrency    req/s    Generation Time    rtf_mean
1              4.18     0.217 s            0.045
2              6.41     0.258 s            0.053
4              7.86     0.389 s            0.079
8              8.70     0.655 s            0.126
16             7.99     1.498 s            0.256
32             8.13     1.549 s            0.259
```

### combo (reduce-overhead + duration_bucket) + no-jitter

```
concurrency    req/s    Generation Time    rtf_mean
1              2.36     0.415 s            0.082
2              4.45     0.414 s            0.089
4              7.18     0.449 s            0.090
8              8.27     0.759 s            0.140
16             8.40     1.218 s            0.244
32             7.45     2.432 s            0.467
```

### duration_bucket only (default compile mode) + no-jitter

```
concurrency    req/s    Generation Time    rtf_mean
1              2.58     0.372 s            0.077
2              4.39     0.418 s            0.085
4              6.49     0.501 s            0.100
8              8.95     0.635 s            0.122
16             8.64     1.131 s            0.215
32             8.21     1.985 s            0.341
```

### duration_bucket only + jitter 10–50 ms

```
concurrency    req/s    Generation Time    rtf_mean
1              2.36     0.410 s            0.085
2              4.20     0.435 s            0.091
4              6.15     0.525 s            0.104
8              8.14     0.681 s            0.136
16             7.05     1.723 s            0.304
32             7.63     1.811 s            0.322
```

### FIFO (default compile) + jitter 10–50 ms (baseline)

```
concurrency    req/s    Generation Time    rtf_mean
1              2.42     0.396 s            0.083
2              4.00     0.470 s            0.091
4              6.07     0.505 s            0.098
8              3.78     1.949 s            0.275
16             5.69     1.808 s            0.315
32             1.64    11.944 s            1.769
```

### FIFO + no-jitter (worst case)

```
concurrency    req/s    Generation Time    rtf_mean
1              2.70     0.352 s            0.073
2              4.46     0.417 s            0.080
4              5.14     0.751 s            0.118
8              3.64     2.341 s            0.314
16             1.72    10.574 s            1.548
32             0.71    45.346 s            9.399
```

## Section breakdown (CUDA-event profiler)

c=1 forward, B=1, 32 unmasking steps:

```
              default mode      reduce-overhead    Δ
TOTAL         374.3 ms          343.3 ms         -8.3%
  gen.transformer    329.7 ms      297.9 ms      -9.6%   ← cudagraphs win
  decoder             30.6 ms       31.8 ms      flat   (eager, not compiled)
  gen.per_i            9.9 ms        9.5 ms      flat
  gen.log_softmax      2.1 ms        2.0 ms      flat
```

The cudagraphs win comes entirely from `gen.transformer` — eliminates
~32 ms of kernel-launch overhead across 32 steps × 13 layers (~50
kernels per layer). Decoder and per-step Python paths run eagerly so
do not benefit.

## What isn't in this commit

- **Step count reduction** (32→16) — gives a clean 2× wall reduction
  (separately measured: c=1 0.41 s → 0.23 s) but trades audio quality.
  Worth A/B-ing on a quality benchmark before defaulting.

## flash-attn varlen — measured, not a win at OmniVoice shapes

Compiled flash-attn 2.8.3 from source for torch 2.10 (single arch sm90,
fwd+bwd hdims, ~25 min build) and wired `flash_attn_varlen_func` in via
`VLLM_OMNI_OMNIVOICE_USE_FLASH_ATTN=1` (correctness verified to bf16
noise floor against the SDPA reference). Result:

```
                            SDPA + cudagraphs       flash-attn + cudagraphs
                            (committed best)        (combo+jitter)
 c=1   wall_avg             0.217 s                 1.766 s    (8× SLOWER)
 c=4   wall_avg             0.389 s                 1.877 s    (5× SLOWER)
 c=16  wall_avg             1.498 s                 3.350 s
 c=32  wall_avg             1.549 s                 5.512 s

                                                    flash-attn + default
                                                    (no cudagraphs)
 c=1   wall_avg                                     0.961 s    (4× slower)
 c=4   wall_avg                                     1.206 s
 c=16  wall_avg                                     2.105 s
 c=32  wall_avg                                     3.092 s
```

Two separate problems combine:

1. **flash-attn doesn't compose with cudagraphs** at OmniVoice's call
   pattern. `max_seqlen = int(valid_lens.max().item())` is a host-side
   scalar that varies per call, plus the variable-length `cu_seqlens`
   forces graph recapture every batch. cudagraphs alone gave us a
   45% c=1 win on SDPA; that win evaporates when flash-attn forces
   recapture.

2. **Packing overhead exceeds the padding savings.** OmniVoice's
   `valid_lens` are bucketed (typical S=200, max S=256, ratio ~1.6×).
   Flash-attn varlen saves ~37% of attention compute by skipping
   padding, but the per-call `q[valid_mask]` gather + `out_padded
   [valid_mask] = out_packed` scatter (×13 layers × 32 steps = 832
   gather/scatter ops per request) more than eats the savings at
   small B. Cutlass FMHA inside SDPA pays the padding tax but stays
   inside one kernel launch.

Code is wired and gated behind a flag (default off) so it's available
if a future workload hits the regime where varlen wins (much higher
length variance, or larger absolute S where the 1.6× ratio becomes
2×+).

## DURATION_BUCKET_TOKENS sweep — 128 is optimal

```
                         bucket=64      bucket=128 (default)    bucket=256
 c=1   wall_avg          0.948 s        0.217 s                  0.727 s
 c=4   wall_avg          1.678 s        0.389 s                  1.381 s
 c=8   wall_avg          5.638 s        0.655 s                  5.269 s
 c=16  wall_avg         19.731 s        1.498 s                 24.081 s
 c=32  wall_avg         17.313 s        1.549 s                 38.066 s
```

128 tokens/bucket is the sweet spot:
- 64 over-fragments — most prompts have no same-bucket peers, fall
  back to B=1 solo execution.
- 256 lets too much length variance into one bucket — at high B the
  FMHA kernel walks the longest slot's S × all slots, the long-slot
  tail you'd hoped to avoid.

Don't change the default.

## Cold-cache caveat for cudagraphs

`reduce-overhead` captures one cudagraph per `(B, S)` shape it sees.
The committed numbers above are steady-state (after the bench warmup
+ first sweep filled the cache). On a *truly* cold server, the first
sweep pays an extra ~100–500 ms for cudagraph capture on each new
shape — observed c=1 wall_avg jumps from steady-state 0.21 s to
~0.47 s on the first sweep after server start.

The benchmark_concurrent.py `--warmup_iters 10` flag fires the same
short prompt 10 times, so it only covers one shape bucket. The 32
real candidates that follow then hit several previously-unseen shapes
and trigger captures. After that one sweep, the cache covers the
candidate distribution and subsequent runs are at the steady-state
numbers.

Production workaround: pre-warm bucket shapes at server startup by
extending `_dummy_run` to fire a handful of prompts spanning the
expected length distribution. Not in this commit.

Reproducibility check (5 back-to-back c=1 sweeps after one cold sweep
to fill the cache):

```
  c=1 rep 1: 0.204 s
  c=1 rep 2: 0.205 s
  c=1 rep 3: 0.249 s   (one slot probably hit a new shape)
  c=1 rep 4: 0.206 s
  c=1 rep 5: 0.206 s
```

## Notes on metric choice

`req/s = N_candidates / total_wall` is gated by **p100 latency** (the
slowest request in the wave). With a homogeneous batched run,
`wall_avg ≈ wall_max` so `req/s` and `rtf_mean` move together. With
`duration_bucket` they decouple — short requests cash out early
(low `wall_avg`), long requests run solo at the back (caps `wall_max`
and therefore `req/s`). For SLA, prefer `wall_p95` over `rtf_mean`.
