# OmniVoice — Streaming TTS hosting guide

Hosting-ready fork of `vllm-omni` that serves
[`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) — zero-shot
multilingual diffusion TTS (Qwen3-0.6B backbone, HiggsAudio V2 tokenizer,
DAC acoustic decoder, 24 kHz output) — as a block-wise streaming service
behind an OpenAI-compatible `/v1/audio/speech` endpoint.

## What this repo provides

- **`start_omnivoice_server.sh`** — one-command launcher with the tuned
  defaults (block-wise streaming, BF16 generator, compiled path, duration-
  bucketed scheduler).
- **Core server code** under `vllm_omni/` — the model runner, diffusion
  engine, and OpenAI-compatible HTTP entrypoints.
- **`examples/online_serving/omnivoice/`** — client examples (curl, Python,
  OpenAI SDK) and API docs.

> Hosting/autoscaling glue (Modal, Baseten, Kubernetes, etc.) is intentionally
> not included — the server is a plain HTTP service on port 8091, and each
> platform has its own preferred way to wire it up. See "Productionizing"
> below for the concrete contract.

## Requirements

- GPU: **H100 80 GB** or **A100 80 GB** (uses ~5 GB VRAM; memory is not
  the bottleneck).
- Python 3.12 with PyTorch + CUDA.
- `transformers >= 5.3.0` (for the HiggsAudio V2 audio tokenizer used in
  voice cloning).
- `--trust-remote-code` is passed automatically to load the custom model
  modules.

## Quick start (bare metal)

```bash
git clone https://github.com/skwid-inc/omnivoice-hosting.git
cd omnivoice-hosting

uv pip install -e . 'transformers>=5.3.0'

./start_omnivoice_server.sh
```

Server listens on `http://0.0.0.0:8091/v1/audio/speech`. Health check:
`GET /health` → `200 OK`.

## Quick start (Docker-style)

```
# Entrypoint: bash ./start_omnivoice_server.sh
# Port:       8091/tcp
# Volumes:    /root/.cache/huggingface   (persist HF downloads across restarts)
# Env:        HF_TOKEN=...               (only if a gated model is used)
```

The launcher honors common overrides — `PORT`, `HOST`, `MODEL`, `VENV_PATH`,
`VLLM_BIN`, and all `VLLM_OMNI_OMNIVOICE_*` / `VLLM_OMNI_DIFFUSION_*` knobs
below.

## Tuned defaults

Baked into `start_omnivoice_server.sh`. All override-able via env.

| Knob | Default | Purpose |
|---|---|---|
| `VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP` | `8` | Denoising steps on first block — drives TTFA. |
| `VLLM_OMNI_OMNIVOICE_NUM_STEP` | `32` | Denoising steps on subsequent blocks. |
| `VLLM_OMNI_OMNIVOICE_BLOCK_SIZE` | `32` | Frames per temporal block. |
| `VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES` | `2` | Trailing frames held per block for smoother boundaries. |
| `VLLM_OMNI_DIFFUSION_BATCH_SIZE` | `16` | Per-GPU concurrent request ceiling. |
| `VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS` | `10` | Coalescing window for the batch scheduler. |
| `VLLM_OMNI_DIFFUSION_BATCH_STRATEGY` | `duration_bucket` | Groups requests by predicted audio length. |
| `VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS` | `128` | Bucket width. |
| `VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE` | `bf16` | Generator dtype (DAC decoder stays fp32). |
| `VLLM_OMNI_OMNIVOICE_OPT` | `1` | Compiled generator path. |
| `VLLM_OMNI_OMNIVOICE_COMPILE_MODE` | `default` | torch.compile mode. |

## Measured performance (single H100 PCIe)

Length-spread sample of 100 prompts from a real TTS workload, streaming mode:

| concurrency | TTFA p50 | TTFA p90 | RTF mean | req/s | audio/wall |
|--:|---:|---:|---:|---:|---:|
| 1 | 101 ms | 110 ms | 0.24 | 0.61 | 3.1× |
| 4 | 116 ms | 136 ms | 0.35 | 1.21 | 6.2× |
| 8 | 151 ms | 1.29 s | 0.41 | 1.45 | 7.4× |
| 16 | 299 ms | 1.84 s | 0.70 | 1.48 | 7.5× (saturated) |

A100-80GB is ~10–15% slower per GPU; everything else identical.

**Recommended operating point**: set container concurrency = `16` to match
`VLLM_OMNI_DIFFUSION_BATCH_SIZE`. Autoscale replicas on queue depth.

## Productionizing — the contract

If you are wrapping this in Modal, Baseten, Kubernetes, or anywhere else:

1. **Image**: install this repo + `transformers>=5.3.0` into a Python 3.12
   env with CUDA. Use `uv pip install -e . 'transformers>=5.3.0'` or the
   equivalent `pip` commands.
2. **Entrypoint**: `./start_omnivoice_server.sh`. The script activates a venv
   at `$VENV_PATH` if present, or uses whatever `vllm` is on `PATH`.
3. **Port**: `8091` (overridable via `PORT`).
4. **Readiness probe**: `GET /health` → `200`. Expect ~30 s cold boot.
5. **Warmup**: before marking ready, fire 3 length-spread requests to
   populate graph captures. Example:

   ```python
   import httpx, time
   base = "http://127.0.0.1:8091"
   for text in ["Short warmup.",
                "Medium length warmup sentence for the TTS service.",
                "Long warmup sample. " * 40]:
       r = httpx.post(f"{base}/v1/audio/speech",
                      json={"model": "k2-fsa/OmniVoice", "input": text,
                            "voice": "default", "response_format": "wav"},
                      timeout=300)
       r.raise_for_status()
   ```

6. **Per-container concurrency**: 16. Anything higher per container hurts
   TTFA without adding throughput.
7. **Scaling**: horizontal. Each replica is independent.
8. **HF cache volume**: mount `/root/.cache/huggingface` to persist model
   weights across container restarts (~600 MB).
9. **HF token**: only required if you swap in a gated model. `k2-fsa/OmniVoice`
   itself is public.

## API surface

Standard OpenAI `/v1/audio/speech` plus two extra fields for voice cloning:

- `ref_audio`: URL (`http://`, `https://`, `file://`) or `data:audio/wav;base64,...`.
- `ref_text`: transcript of the reference audio (needed for best fidelity).
- `stream: true`: enables block-wise WAV/PCM streaming.

See `examples/online_serving/omnivoice/README.md` for full client examples.

## Contact

File an issue in this repo: https://github.com/skwid-inc/omnivoice-hosting/issues
