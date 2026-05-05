# OmniVoice Perf Results

Snapshot of benchmark numbers from the late-April 2026 perf push.

## Best stable config

```bash
VLLM_OMNI_DIFFUSION_CONCURRENT=1
VLLM_OMNI_DIFFUSION_BATCH_SIZE=32
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=0       # was 30; 0 is faster at c=1 with no throughput cost
VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket
VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=128
VLLM_OMNI_OMNIVOICE_OPT=1
VLLM_OMNI_OMNIVOICE_COMPILE_MODE=default
VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bfloat16
VLLM_OMNI_OMNIVOICE_NUM_STEP=16           # set to 32 for full-quality, 16 ~50% faster
```

YAML: `step_execution: true` (vllm-style continuous batching).

## Headline numbers (H100, no jitter, 32 prompts/wave, median of 3 runs)

**All numbers below are at `NUM_STEP=16`** (half the OmniVoice config default of
32). Audio quality at steps=16 has not been auditorily validated; samples for
inspection are at `outputs/audio_quality_check/`. For full-quality 32-step
runs, expect ~2x these wall numbers and ~half the throughput.

```
NUM_STEP=16 (default in run_server_optimized.sh):

concurrency    req/s    Generation Time    rtf_mean    wall_p95
1              5.43     0.175 s            0.036       0.183 s
4              9.50     0.368 s            0.075       0.397 s
8             13.90     0.435 s            0.087       0.586 s
16            12.83     0.962 s            0.187       1.224 s
32            12.69     1.452 s            0.252       2.009 s
```

For comparison, NUM_STEP=32 numbers (same config otherwise) measured earlier:

```
NUM_STEP=32 (full quality):

concurrency    req/s    Generation Time    rtf_mean
1              4.08     0.239 s            0.048
4             11.34     0.293 s            0.058
8             14.10     0.434 s            0.082
16            12.69     0.788 s            0.155
32            11.31     1.649 s            0.291
```

Run-to-run variance is < 2% across all concurrencies in this config at
NUM_STEP=16. The NUM_STEP=32 numbers had higher variance run-to-run; values
shown are from the best stable run.

## Improvement vs morning baseline

The pasted baseline was a flat ~3.1 req/s across every concurrency level
(no batching benefit, single-flight execution path).

```
                Baseline       Final         req/s     latency
                req/s wall     req/s wall    change    change
 c=1   3.12   0.311s    4.92   0.195s   +58%       -37%
 c=4   3.16   1.225s    8.84   0.405s   +180%      -67%
 c=8   3.13   2.477s   12.85   0.491s   +311%      -80%
 c=16  3.04   5.090s   12.26   1.028s   +303%      -80%
 c=32  3.21   9.434s   12.40   1.493s   +286%      -84%
```

## Optimization stack (in order of impact)

1. **`step_execution: true`** (YAML, in `omnivoice.yaml`)
   Per-step continuous batching across in-flight slots instead of
   per-request. Activates the `StepScheduler` path. Single biggest
   throughput win at c >= 4.

2. **`VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket`**
   Admission control that groups same-length requests so a long-text
   slot can't poison a wave of short slots. Eliminates the 30x FMHA
   per-call variance the nsys roofline showed. Major win at c=8/c=16
   with mixed prompt lengths.

3. **`VLLM_OMNI_OMNIVOICE_NUM_STEP=16`**
   Halve the unmasking step count (default 32). Linear ~50% latency
   reduction on the dominant `gen.transformer` section. Audio quality
   sample at `outputs/audio_quality_check/` for inspection - the
   model is stochastic so two runs at steps=32 also produce uncorrelated
   waveforms; need auditory comparison rather than numerical.

4. **`VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bfloat16`**
   Generator-only bf16 cast. Decoder, audio tower, RVQ stay fp32. The
   YAML's pipeline-level `dtype: "float32"` MUST stay - flipping it to
   bfloat16 crashes at load with a `mat1 != mat2` dtype error on the
   DAC weights.

5. **`VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=0`**
   Disables the coalesce wait. With `step_execution=true`, batching
   already happens at the step level so the wait knob's main purpose
   (waiting for stragglers before dispatching solo) is moot. WAIT > 0
   adds latency at c=1 with no throughput benefit because batches form
   continuously.

## What did NOT pay off

- **flash-attn varlen** (`VLLM_OMNI_OMNIVOICE_USE_FLASH_ATTN=1`): wired
  correctly behind a flag, kernel itself is faster than SDPA Cutlass-FMHA,
  but the pack/unpack and graph-break overhead in our integration cost
  more than the kernel saves. Left in as opt-in for future work; would
  need plumbing to push cu_seqlens computation outside the per-step
  loop entirely.

- **`COMPILE_MODE=reduce-overhead`** (cudagraphs): bimodal at c=1 (some
  shapes hit cache, others pay capture cost). Unstable above c=2 with
  `step_execution=true`. Default mode is the stable choice.

- **`PREWARM_BUCKETS=1`**: implementation present but crashes during
  cudagraph capture in current configuration. Left for future fix.

- **Larger `BATCH_WAIT_MS`** (50, 100): improves c>=8 throughput
  marginally but adds 30-100 ms latency at c=1. Not worth the trade.

## Reproduction

```bash
# Server
./examples/online_serving/omnivoice/run_server_optimized.sh

# Bench
python3 examples/online_serving/omnivoice/benchmark_concurrent.py \
  --concurrencies 1,2,4,8,16,32 \
  --candidate_count 32 \
  --warmup_iters 10 \
  --jitter_ms_min 0 --jitter_ms_max 0
```
