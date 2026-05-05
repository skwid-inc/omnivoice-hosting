# TTS Universal Benchmark

A model-agnostic serving benchmark for TTS models in vllm-omni. One CLI
(`bench_tts.py`) + one YAML registry (`model_configs.yaml`) drive perf and
quality runs for every registered checkpoint: **Qwen3-TTS** (Base / CustomVoice)
and **VoxCPM2** today, more to come.

The same three task types — `voice_clone`, `default_voice`, `voice_design` —
are wired into both the manual CLI and the DFX nightly CI matrix
(`tests/dfx/perf/tests/test_tts.json`).

## OmniVoice A100 F12 stable baseline

Locked-in reference numbers used to validate every subsequent A100
optimization. **Don't change the config** — anything new should land as a
separate run and be diffed against this table.

- **Hardware**: NVIDIA A100 80GB PCIe (sm80, driver 535.183.06)
- **Stack**: torch 2.10.0+cu128, vllm 0.19.0, vllm-omni @ `46eaa290f`
- **Commit / tag**: `25d855ee` / `a100-baseline-stable`
- **Branch**: `a100-fastdllm-implementation` (started here, no commits ahead)
- **NUM_STEP**: 32 (no quality trade-off vs reference inference)
- **Last verified**: 2026-04-30 (`outputs/baseline_rerun_25d855ee_20260430T182748Z/`)

### Numbers (verified 2026-04-30)

```
 c     req/s   wall_avg   wall_p50   wall_p95   wall_max   rtf_mean
 1     2.65    0.350 s    0.354 s    0.359 s    0.360 s    0.072
 2     4.47    0.384 s    0.365 s    0.448 s    0.805 s    0.077
 4     5.90    0.521 s    0.474 s    0.774 s    0.803 s    0.098
 8     6.61    0.875 s    0.862 s    1.237 s    1.318 s    0.166
 16    5.55    1.851 s    1.972 s    3.109 s    3.227 s    0.375
 32    5.16    3.148 s    3.240 s    4.956 s    4.958 s    0.633
```

Single-sweep noise band is ±5%. c=1..8 reproduce within ±3% of the
original sweep; c=16 is the noisiest cell (one outlier moves the mean
~10%) — take a median of 3 if you need a tight number there.

### Reproduce exactly

The reproduction is a **two-terminal** flow: the server runs in one terminal,
the bench client in the other. The two need to be matched on commit (`25d855ee`)
and the env-var bundle below.

#### 1. Check out the pinned baseline

```bash
git fetch --tags
git checkout a100-baseline-stable     # detached HEAD at 25d855ee
# or to start an experiment branch:
git checkout -b my-experiment a100-baseline-stable
```

Verify hardware, driver, vllm version match the header above. Anything
different → don't trust the deltas, re-baseline.

#### 2. Start the server (terminal A)

Make sure `HF_TOKEN` is exported (or `set -a; source .env.local; set +a`
to load it from a project env file). The model is `k2-fsa/OmniVoice` —
public, but the HF cache lookup still needs a token in some environments.

The launcher's defaults are NOT the F12 config. Override every env var
explicitly so a stale shell doesn't poison the run:

```bash
VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket \
VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=256 \
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=30 \
VLLM_OMNI_DIFFUSION_BATCH_SIZE=12 \
VLLM_OMNI_DIFFUSION_CONCURRENT=1 \
VLLM_OMNI_OMNIVOICE_OPT=1 \
VLLM_OMNI_OMNIVOICE_COMPILE_MODE=default \
VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=bfloat16 \
VLLM_OMNI_OMNIVOICE_NUM_STEP=32 \
VLLM_BIN=.venv/bin/vllm \
bash examples/online_serving/omnivoice/run_server_optimized.sh
```

Wait for `Application startup complete` in the log. The server listens
on `:8091`. Cold cudagraph capture is **off** in this config
(`COMPILE_MODE=default`), so first-request latency is the same as
steady-state.

Why each knob matters (changing any of these moves the numbers):

| Var | F12 value | What it controls |
|---|---|---|
| `BATCH_STRATEGY` | `duration_bucket` | Group same-length prompts into one B>1 forward |
| `DURATION_BUCKET_TOKENS` | `256` | Bucket width. 128 fragments; 512 lifts long-prompt tail |
| `BATCH_WAIT_MS` | `30` | Hold the queue open 30 ms before dispatch — the unlock that lets peers merge |
| `BATCH_SIZE` | `12` | Cap merged batch size. 32 OOM's the FMHA at high c; 12 stays in scaling regime |
| `CONCURRENT` | `1` | =2 silently regresses to 100% B=1 (scheduler `_running` gate, see PERF_A100.md) |
| `COMPILE_MODE` | `default` | `reduce-overhead` regressed -15..-25% on A100 (graph partition + dyn-shape thrash) |
| `GENERATOR_DTYPE` | `bfloat16` | fp16/fp32 not measured |
| `NUM_STEP` | `32` | Steps below 32 are quality drops; record separately |
| `OMNIVOICE_GUMBEL_SEED` | (unset) | Optional. Set to any int to lock the auto-voice for A/B quality tests; default off so the perf baseline isn't affected by a forced RNG path |

#### 3. Run the benchmark (terminal B)

```bash
.venv/bin/python examples/online_serving/omnivoice/benchmark_concurrent.py \
  --api-base http://127.0.0.1:8091 \
  --concurrencies 1,2,4,8,16,32 \
  --candidate_count 32 \
  --warmup_iters 10 \
  --jitter_ms_min 0 --jitter_ms_max 0 \
  --out_dir outputs/baseline_$(date -u +%Y%m%dT%H%M%SZ)
```

Important: `--api-base` is the **bare host** (`http://127.0.0.1:8091`),
NOT `http://127.0.0.1:8091/v1`. The bench script appends `/v1/audio/speech`
itself; passing the trailing `/v1` produces `404`.

The bench prints a markdown-style `CONCURRENCY_SWEEP` table on stdout
and writes `concurrency_sweep.csv` + `summary.json` to `--out_dir`.
Compare those CSVs against the F12 numbers above (or against
`benchmarks/tts/PERF_A100.md` for the original sweep).

#### 4. Confirm you reproduced

A clean reproduction looks like:

- c=1..8 all within ±3% of the F12 row
- c=16 within ±10% (single-run noise; median-of-3 if needed)
- `errors=0` for every concurrency
- `n_in_range=26` (test set drops the 6 too-short / too-long prompts)

If c=8 wall_avg drifts more than ±5% from `0.875 s`, something changed
(driver, kernel, env). Don't run experiments against a drifted baseline —
re-pin and update the numbers above.

### Audio-quality A/B (every config change)

Throughput numbers don't tell you whether the optimization broke audio.
Use `ab_quality.py` after every config change before claiming a win.

**Voice locking.** OmniVoice's `voice="default"` is auto-voice mode —
the model picks a different latent voice on every call, which makes any
A/B impossible. To get reproducible audio, the server must be started
with `VLLM_OMNI_OMNIVOICE_GUMBEL_SEED=<int>` (any int; pick once and
keep it). That env var is the only thing that locks the voice — if it
isn't set, every request gets a different speaker even for the same
text. (Voice cloning via `ref_audio` would also work, but
`HiggsAudioV2TokenizerModel` requires `transformers>=5.3`, which
conflicts with the `vllm 0.19` pin of `transformers<5`.)

The script holds **5 fixed prompts** constant across runs, so the only
variable in the comparison is the server config. Prompts cover the
audio-quality failure modes that matter:

1. `01_greeting` — name pronunciation, conversational prosody
2. `02_numbers` — digit-by-digit articulation
3. `03_question` — rising intonation, prosody
4. `04_narrative` — typical content
5. `05_complex` — long sentence with subordinate clauses (where step
   skipping / KV-cache drift breaks first)

Workflow:

```bash
# 1. Server up with F12 baseline config + VLLM_OMNI_OMNIVOICE_GUMBEL_SEED=42
#    (use the same seed for both A and B). Add it to the env var bundle
#    in step 2 of the F12 reproduction:
#      VLLM_OMNI_OMNIVOICE_GUMBEL_SEED=42 \
#      ...other F12 env vars...
#      bash examples/online_serving/omnivoice/run_server_optimized.sh
#
#    Then:
.venv/bin/python benchmarks/tts/ab_quality.py generate \
  --api-base http://127.0.0.1:8091 \
  --label baseline \
  --out_dir outputs/ab/baseline

# 2. Stop the server. Restart with KEEP=42 GUMBEL_SEED + the experiment
#    env vars (e.g. add VLLM_OMNI_OMNIVOICE_CONFIDENCE_THRESHOLD=0.9 and
#    VLLM_OMNI_OMNIVOICE_CONFIDENCE_SMALL_BLOCK_SIZE=2 for a Fast-dLLM-
#    style confidence-decoding test). Then:
.venv/bin/python benchmarks/tts/ab_quality.py generate \
  --api-base http://127.0.0.1:8091 \
  --label fastdllm_thr09_sb2 \
  --out_dir outputs/ab/fastdllm_thr09_sb2

# 3. Generate a side-by-side listening page + automated delta metrics:
.venv/bin/python benchmarks/tts/ab_quality.py compare \
  outputs/ab/baseline \
  outputs/ab/fastdllm_thr09_sb2 \
  --out_dir outputs/ab/cmp_baseline_vs_fastdllm
```

To verify the seed lock is working, generate the same `--label` twice
in a row against the seeded server and `md5sum` the WAVs — they should
be byte-identical. If they aren't, `VLLM_OMNI_OMNIVOICE_GUMBEL_SEED`
isn't set on the server.

Open `outputs/ab/cmp_baseline_vs_fastdllm/index.html` in a browser. The
five prompts each render two players labeled `X` and `Y` — randomized
per prompt so you can't tell which is the experimental run by position.
Listen, decide, then click "reveal mapping" per prompt.

The HTML also shows automated delta metrics — these are sanity checks,
not a quality verdict:

| Metric | Acceptable | Warning |
|---|---|---|
| `wall_speedup_b_over_a` | report it; > 1 means B is faster | n/a |
| `len_ratio_b_over_a` | 0.95-1.05 | < 0.9 or > 1.1 = different content generated |
| `rms_ratio_b_over_a` | 0.85-1.15 | large drop = clipping or partial silence |
| `centroid_delta_hz` | < 200 Hz typical | > 500 Hz = audible timbre shift |

**Listening overrides metrics.** If the metrics all look fine but the
audio sounds robotic / stuttery / wrong-prosody, the experiment loses.

### Stable baseline A/B reference

Once `ab_quality.py generate --label baseline` has been run against the
F12 baseline server, that output dir is the canonical reference for
every future A/B. Don't regenerate it unless the baseline numbers above
have drifted out of band (in which case both this section and that A/B
reference need refreshing).

## Quick start

### 1. Start the server

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000
```

The server auto-loads its Deploy YAML from `vllm_omni/deploy/qwen3_tts.yaml`
(Pipeline + Deploy schema introduced in #2383). No `--stage-configs-path` or
`--deploy-config` flag is needed for any registered model.

For OmniVoice H100 scheduling tests, start the optimized OmniVoice launcher:

```bash
cd examples/online_serving/omnivoice
VLLM_OMNI_DIFFUSION_BATCH_SIZE=16 \
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=10 \
VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=fifo \
./run_server_optimized.sh
```

Then run the closed-loop voice-agent benchmark from the repo root. Each worker
acts like one live voice agent: send TTS, wait for the response, spend 1-19s
"playing" the audio, then send the next 2-20s utterance.

```bash
python3 benchmarks/tts/voice_agent_latency.py \
    --workers 16,32,64,80,120 \
    --duration-s 600 \
    --warmup-s 60 \
    --target-audio-min-s 2 \
    --target-audio-max-s 20 \
    --playback-min-s 1 \
    --playback-max-s 19 \
    --label bs16_wait10_bf16
```

To compare scheduling strategies, restart the server with
`VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket` and sweep
`VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS` values such as `64`, `96`, and
`128`.

### 2. Run the benchmark (`vllm bench serve --omni`)

The primary, directly-controllable path. Copy-paste one of these and tweak
any bench flag (sampling params, endpoint, extra body, warmups, etc.):

#### voice_clone (Qwen3-TTS-Base, seed-tts dataset)

```bash
vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts \
    --dataset-path /path/to/seed-tts-eval \
    --seed-tts-locale en \
    --num-prompts 20 --num-warmups 2 \
    --extra-body '{"task_type":"Base"}' \
    --max-concurrency 1 --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration \
    --save-result --result-dir ./results
```

#### default_voice (Qwen3-TTS-CustomVoice, bundled seed_tts_smoke)

```bash
vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts-text \
    --dataset-path benchmarks/build_dataset/seed_tts_smoke \
    --seed-tts-locale en \
    --num-prompts 20 --num-warmups 2 \
    --extra-body '{"voice":"Vivian","language":"English","task_type":"CustomVoice"}' \
    --max-concurrency 1 --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration \
    --save-result --result-dir ./results
```

#### voice_design (Qwen3-TTS-CustomVoice, bundled seed_tts_design)

```bash
vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts-design \
    --dataset-path benchmarks/build_dataset/seed_tts_design \
    --seed-tts-locale en \
    --num-prompts 20 --num-warmups 2 \
    --extra-body '{"task_type":"VoiceDesign","language":"English"}' \
    --max-concurrency 1 --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration \
    --save-result --result-dir ./results
```

#### Add WER / SIM / UTMOS to any of the above

Append `--seed-tts-wer-eval` (and optionally `SEED_TTS_EVAL_DEVICE=cuda:0`
in the env, per PR #2558). This triggers the seed-tts-eval protocol:
Whisper-large-v3 ASR → WER, WavLM embeddings → SIM, balacoon/utmos → UTMOS.

### 3. Convenience wrapper (`bench_tts.py`)

If you're running the **canonical** configuration for a registered model,
`bench_tts.py` loads the right defaults from `model_configs.yaml` and
emits the exact `vllm bench serve --omni` command above — useful for
concurrency sweeps and multi-task runs:

```bash
# Smallest smoke — 5 prompts, concurrency=1
python benchmarks/tts/bench_tts.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task voice_clone \
    --dataset-path /path/to/seed-tts-eval \
    --concurrency 1 --num-prompts 5 \
    --output-dir ./results

# Full concurrency sweep
python benchmarks/tts/bench_tts.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task voice_clone \
    --dataset-path /path/to/seed-tts-eval \
    --concurrency 1 2 4 8 16 32 \
    --num-prompts 20 \
    --output-dir ./results

# With WER / SIM / UTMOS quality eval (adds ASR + embedding compute)
python benchmarks/tts/bench_tts.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task voice_clone \
    --dataset-path /path/to/seed-tts-eval \
    --wer-eval \
    --concurrency 4 --num-prompts 200 \
    --output-dir ./results
```

### 4. Plot a sweep

```bash
python benchmarks/tts/plot_results.py \
    --results ./results/*.json \
    --output ./results/curve.png
```

Outputs TTFP / RTF / throughput curves (and a markdown table) for every
`(task, concurrency)` combination in the result set.

## Task types

| Task            | Dataset           | Request body                                        | Checkpoints that support it              |
|-----------------|-------------------|-----------------------------------------------------|------------------------------------------|
| `voice_clone`   | `seed-tts`        | `ref_audio` + `ref_text` + `task_type=Base`         | `Qwen3-TTS-*-Base`, `VoxCPM2`            |
| `default_voice` | `seed-tts-text`   | `voice=Vivian` + `task_type=CustomVoice`            | `Qwen3-TTS-*-CustomVoice`                |
| `voice_design`  | `seed-tts-design` | `instructions=<natural-language description>` + `task_type=VoiceDesign` | `Qwen3-TTS-*-CustomVoice` |

**`-CustomVoice` checkpoints do NOT ship `speaker_encoder` weights**, so
voice_clone requests raise `ValueError` at model runtime. Use `-Base` for
voice_clone.

## Adding a new TTS model

Drop an entry into `model_configs.yaml` — no Python changes required:

```yaml
models:
  <org>/<model-id>:
    supported_tasks: [voice_clone]          # or default_voice / voice_design
    backend: openai-audio-speech            # vllm bench serve backend
    endpoint: /v1/audio/speech              # OpenAI-compatible endpoint
    task_extra_body:                        # merged into every request's body
      voice_clone:
        task_type: Base
```

Then add the model's Deploy YAML under `vllm_omni/deploy/<model>.yaml`
(Pipeline + Deploy schema) and it's immediately benchable.

## Datasets

| Dataset            | Bundled? | Format            | Source                                                         |
|--------------------|----------|-------------------|----------------------------------------------------------------|
| `seed-tts-design`  | ✅       | 5-field meta.lst  | `benchmarks/build_dataset/seed_tts_design/en/meta.lst` (20 prompts) |
| `seed_tts_smoke`   | ✅       | 4-field meta.lst  | `benchmarks/build_dataset/seed_tts_smoke/en/meta.lst` (20 text-only) |
| `seed-tts`         | ❌       | 4-field meta.lst + WAVs | Google-Drive: [BytedanceSpeech/seed-tts-eval][seedtts] (~1.2 GB) |
| `seed-tts-text`    | ❌       | 4-field meta.lst  | Same archive as `seed-tts` (wav column unused)                 |

[seedtts]: https://github.com/BytedanceSpeech/seed-tts-eval

For manual voice_clone / default_voice runs against the full corpus, follow
`benchmarks/build_dataset/download_process_data_seedtts.md` and point
`--dataset-path` at the extracted `seedtts_testset` directory.

## DFX nightly CI

`tests/dfx/perf/tests/test_tts.json` wires three perf regimes plus quality:

| eval_phase    | concurrency | purpose                                                 | Baseline metrics                        |
|---------------|-------------|---------------------------------------------------------|-----------------------------------------|
| `latency`     | 1           | Single-request TTFP / RTF SLO                           | `median_audio_ttfp_ms`, `median_audio_rtf` |
| `throughput`  | 8           | Codec-batching cliff sentinel (PDF #272 concurrency≥8)  | `median_audio_ttfp_ms`, `median_audio_rtf` |
| `quality`     | 4           | WER / SIM / UTMOS regression (disabled in CI by default)| `mean_audio_rtf`                        |

Why `median_*` for latency/throughput and `mean_*` for quality: latency
distributions have cold-start tails that drag the mean; quality aggregates
over 200 prompts so single-request outliers don't matter.

Quality entries are `enabled: false` in CI because seed-tts-eval is not
staged in the Buildkite container (matches the precedent in
PR #2558 — quality runs are manual / release-validation, not nightly).

## Concurrency cliff regression sentinel

Observed on H20-3e, Qwen3-TTS-1.7B (measured pre-merge on this branch):

| Task          | Model         | c=1    | c=4    | **c=8**    | c=16   | c=32   |
|---------------|---------------|--------|--------|------------|--------|--------|
| voice_clone   | 1.7B-Base     | RTF 0.15 / TTFP 165ms | 0.28 / 412ms | **0.49 / 1701ms** | 0.72 / 3355ms | 0.77 / 3772ms |
| voice_design  | 1.7B-CustomVoice | RTF 0.08 / TTFP 53ms  | 0.11 / 154ms | **0.21 / 872ms**  | 0.33 / 1801ms | 0.38 / 1989ms |

Both models show a **4–6× TTFP jump from c=4 to c=8** while audio throughput
saturates around c=4–8 — the codec-bs=1 bottleneck documented in
vllm-project/vllm-omni#272. The `throughput` CI regime at c=8 is the
sentinel for regressions in this area.

## File layout

```
benchmarks/tts/
├── README.md                  (this file)
├── bench_tts.py               CLI — serve-mode benchmark driver
├── bench_voxcpm_offline.py    CLI — offline VoxCPM benchmark (sync + streaming)
├── plot_results.py            Generate per-task / per-concurrency curves
└── model_configs.yaml         Model registry (supported tasks + extra body)
```

## Related

- Upstream seed-tts-eval integration: vllm-project/vllm-omni#2558
- Pipeline + Deploy schema: vllm-project/vllm-omni#2383
- Concurrency cliff RFC: vllm-project/vllm-omni#272
