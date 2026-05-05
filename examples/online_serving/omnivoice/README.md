# OmniVoice

## Model Overview

| Model | Description |
|-------|-------------|
| `k2-fsa/OmniVoice` | Zero-shot multilingual TTS (600+ languages) with diffusion language model (Qwen3-0.6B backbone) |

> **Note:** Requires `transformers>=5.3.0` for voice cloning (HiggsAudioV2 tokenizer). Auto voice and voice design work with `transformers>=4.57.0`.

## Launch the Server

```bash
vllm serve k2-fsa/OmniVoice \
    --omni \
    --port 8091 \
    --trust-remote-code
```

Or use the convenience script:

```bash
./run_server.sh
```

For H100 throughput/latency sweeps, use the optimized launcher. It enables
concurrent diffusion batching, generator-only BF16, and the compiled OmniVoice
generator path while leaving the DAC decoder in FP32:

```bash
./run_server_optimized.sh
```

The main knobs are env-overridable:

```bash
VLLM_OMNI_DIFFUSION_BATCH_SIZE=16 \
VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=10 \
VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=fifo \
VLLM_OMNI_OMNIVOICE_COMPILE_MODE=max-autotune-no-cudagraphs \
./run_server_optimized.sh
```

Set `VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=duration_bucket` to test batching
utterances with similar estimated output lengths together. Use
`VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS` to sweep bucket width.

For block-wise speech streaming, use the step-execution launcher:

```bash
./run_server_block_streaming.sh
```

This enables `/v1/audio/speech` with `stream=true` and emits PCM/WAV audio as
each temporal block completes. Active streams stay on the diffusion scheduler,
so concurrent requests can still batch generator forward passes. The main
streaming knobs are `VLLM_OMNI_OMNIVOICE_BLOCK_SIZE`,
`VLLM_OMNI_DIFFUSION_BATCH_SIZE`, and
`VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS`.

For server-side voice cloning, set `VLLM_OMNI_OMNIVOICE_VOICE_MAP` to a JSON
file that maps client `voice` names to local reference recordings. The
OmniVoice worker loads and encodes these clips at startup, so clients only send
the `voice` string:

```json
{
  "default": {
    "audio": "/srv/voices/default.wav",
    "ref_text": "Transcript of the 5 to 6 second reference recording.",
    "language": "English"
  },
  "amy": {
    "audio": "/srv/voices/amy.wav",
    "ref_text": "Transcript of Amy's reference recording."
  }
}
```

Then launch with:

```bash
VLLM_OMNI_OMNIVOICE_VOICE_MAP=/srv/voices/omnivoice_voices.json \
./run_server_block_streaming.sh
```

If clients omit `voice`, set `VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE=amy` or include
a `default` entry in the voice map.

## Send TTS Request

### Using curl

```bash
# Basic TTS (auto voice)
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav"
    }' --output output.wav
```

### Using Python

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.audio.speech.create(
    model="k2-fsa/OmniVoice",
    voice="default",
    input="Hello, how are you?",
)

response.stream_to_file("output.wav")
```

### Using the CLI Client

```bash
cd examples/online_serving/omnivoice

# Basic TTS
python speech_client.py --text "Hello, how are you?"

# Specify language for improved quality
python speech_client.py --text "Bonjour, comment allez-vous?" --language French

# Stream block-wise audio to a WAV file
python speech_client.py --stream --text "Hello, how are you?" --response-format wav
```

The CLI client supports:

- `--api-base`: API base URL (default: `http://localhost:8091`)
- `--model` (or `-m`): Model name (default: `k2-fsa/OmniVoice`)
- `--text`: Text to synthesize (required)
- `--response-format`: Audio format: wav, mp3, flac, pcm, aac, opus (default: wav)
- `--stream`: Request streaming audio chunks; use wav or pcm response format
- `--language`: Language hint (default: Auto)
- `--output` (or `-o`): Output file path (default: `omnivoice_output.wav`)

## Inference Modes

OmniVoice supports three inference modes. The online Speech API supports auto
voice and server-side voice-clone presets through `voice` plus
`VLLM_OMNI_OMNIVOICE_VOICE_MAP`. Ad hoc client-supplied cloning remains
available through `ref_audio` / `ref_text`.

| Mode | Description | Online API | Offline |
|------|-------------|:----------:|:-------:|
| Auto Voice | Generate speech without reference | Yes | Yes |
| Voice Clone | Clone from reference audio + transcript | Yes, via server-side presets or `ref_audio` / `ref_text` | Yes |
| Voice Design | Control style via natural language instruction | - | Yes |

## Architecture

OmniVoice uses a single-stage diffusion pipeline:

- **Stage 0 (Generator)**: Qwen3-0.6B transformer with 32-step iterative masked unmasking and classifier-free guidance. Generates 8-codebook audio tokens from text, then decodes to 24kHz waveform via HiggsAudioV2 RVQ quantizer + DAC acoustic decoder.

## API Parameters

OmniVoice uses the standard `/v1/audio/speech` endpoint. See the [Speech API reference](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/) for full documentation.

Key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize |
| `voice` | string | "default" | Voice name |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |

## Troubleshooting

1. **TTS model did not produce audio output**: Ensure the model is fully downloaded (`huggingface-cli download k2-fsa/OmniVoice`)
2. **Connection refused**: Make sure the server is running on the correct port
3. **Out of memory**: Reduce `--gpu-memory-utilization` (default stage config uses 0.5)
4. **Slow first request**: The model performs warmup on first inference; subsequent requests are faster
