# OmniVoice — Streaming TTS hosting guide

Quick reference for deploying this fork of `vllm-omni` as a production OmniVoice
TTS service (block-wise streaming, voice cloning, OpenAI-compatible API).

Model: [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) — zero-shot
multilingual diffusion TTS (Qwen3-0.6B backbone, HiggsAudio V2 tokenizer,
DAC acoustic decoder, 24 kHz output).

## What this branch provides

- **`start_omnivoice_server.sh`** — one-command launcher for a single-GPU server
  with the tuned defaults (block-wise streaming, BF16 generator, compiled path,
  duration-bucketed scheduler).
- **`examples/online_serving/omnivoice/modal_app.py`** — turnkey Modal
  deployment (autoscaling, persistent HF cache volume, length-spread warmup).
- **`examples/online_serving/omnivoice/run_server_block_streaming.sh`** — the
  underlying launcher script used by `start_omnivoice_server.sh`.
- **`examples/online_serving/omnivoice/README.md`** — full client/API docs.

## Requirements

- GPU: **H100 80 GB** or **A100 80 GB** (uses ~5 GB VRAM, so memory is not
  the bottleneck; either works).
- Python 3.12 with PyTorch + CUDA.
- `transformers >= 5.3.0` (for the HiggsAudio V2 audio tokenizer used in
  voice cloning).
- `--trust-remote-code` to load the custom model modules.

## Quick start (single machine)

```bash
git clone https://github.com/skwid-inc/omnivoice-hosting.git
cd omnivoice-hosting

# install into an existing venv where vllm-omni can be built
uv pip install -e . 'transformers>=5.3.0'

./start_omnivoice_server.sh
```

The server listens on `http://0.0.0.0:8091/v1/audio/speech`.

## Deploy on Modal

```bash
pip install modal
modal token new
modal deploy examples/online_serving/omnivoice/modal_app.py
```

Overrides (all optional):

```bash
MODAL_GPU=H100                     # or A100
MODAL_MAX_CONCURRENT_INPUTS=16     # match VLLM_OMNI_DIFFUSION_BATCH_SIZE
MODAL_MAX_CONTAINERS=4             # autoscaling ceiling
modal deploy examples/online_serving/omnivoice/modal_app.py
```

See `examples/online_serving/omnivoice/README.md#deploy-on-modal`.

## Deploy on Baseten / other platforms

The same pattern applies: build a container from this branch + the
`transformers>=5.3.0` pin, run `start_omnivoice_server.sh` as the entrypoint,
expose port 8091, use the built-in `/health` endpoint for readiness probes.

## Streaming request (OpenAI-compatible + voice cloning extensions)

```bash
curl -N -X POST https://<endpoint>/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "k2-fsa/OmniVoice",
    "input": "Hello, how are you?",
    "voice": "default",
    "response_format": "wav",
    "stream": true
  }' --output out.wav
```

For voice cloning, add `ref_audio` (base64 WAV or URL) and `ref_text`
(transcript). See `examples/online_serving/omnivoice/README.md` for full API.

## Tuned defaults

Set by `start_omnivoice_server.sh` and `modal_app.py`. Override via env vars if
needed.

| Knob | Default | Purpose |
|---|---|---|
| `VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP` | `8` | Denoising steps on first block — drives TTFA. |
| `VLLM_OMNI_OMNIVOICE_NUM_STEP` | `32` | Steps on subsequent blocks. |
| `VLLM_OMNI_OMNIVOICE_BLOCK_SIZE` | `32` | Frames per temporal block. |
| `VLLM_OMNI_DIFFUSION_BATCH_SIZE` | `16` | Per-GPU concurrent request ceiling. |
| `VLLM_OMNI_DIFFUSION_BATCH_STRATEGY` | `duration_bucket` | Groups requests by predicted audio length. |
| `VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE` | `bf16` | Generator dtype (DAC decoder stays fp32). |
| `VLLM_OMNI_OMNIVOICE_OPT` | `1` | Compiled generator path. |
| `VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES` | `2` | Trailing frames held per block for smoother boundaries. |

## Measured performance (single H100 PCIe)

Length-spread sample of 100 prompts from a real TTS workload, streaming mode:

| concurrency | TTFA p50 | TTFA p90 | RTF mean | req/s | audio/wall |
|--:|---:|---:|---:|---:|---:|
| 1 | 101 ms | 110 ms | 0.24 | 0.61 | 3.1× |
| 4 | 116 ms | 136 ms | 0.35 | 1.21 | 6.2× |
| 8 | 151 ms | 1.29 s | 0.41 | 1.45 | 7.4× |
| 16 | 299 ms | 1.84 s | 0.70 | 1.48 | 7.5× (saturated) |

A100-80GB is ~10–15% slower per GPU, everything else identical.

**Recommended operating point**: container concurrency = `16`,
autoscale replicas based on queue depth.

## Healthcheck

```
GET /health   -> 200 OK
```

Server cold-boot is ~30 s (vllm init + stage load + warmup). Use the warmup
loop in `modal_app.py` as a reference for readiness gating.

## Contact

File an issue in this repo: https://github.com/skwid-inc/omnivoice-hosting/issues
