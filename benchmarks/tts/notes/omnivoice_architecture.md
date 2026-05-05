# OmniVoice Architecture — Implementation-Grade Map for Fast-dLLM Port

This document is a code-archaeology pass through the vllm-omni OmniVoice
implementation, focused on the data and control flow a Fast-dLLM-style
confidence-based parallel-decoding port has to interact with. Everything
below is grounded in code at the line numbers shown; if you change any file
that's quoted here, fix the line numbers too.

Repo head: `a100-fastdllm-implementation` branch, last verified 2026-04-30.
File paths are absolute under `/home/ubuntu/vllm-omni/`.

---

## 1. Top-level architecture

OmniVoice is a two-stage TTS model (`k2-fsa/OmniVoice` HF checkpoint):

1. **Stage 0 — Generator** (`omnivoice_generator.py`)
   Takes a text prompt (and optional reference audio for voice cloning),
   produces 8 parallel "codebooks" of audio tokens (vocab 1025 each, where
   1024 is the `[MASK]` token). The generator is **non-autoregressive**: it
   starts with a fully-masked target region and runs **32 iterative
   unmasking steps**, each of which is a full bidirectional Qwen3-0.6B
   transformer pass over the entire `(B, S)` sequence. Each step
   "unmasks" `k_step ≈ total_mask / 32` tokens chosen by a top-k over a
   confidence-plus-Gumbel-noise score. Output: `[B, 8, target_len]` tokens.

2. **Stage 1 — Decoder / Vocoder** (`omnivoice_decoder.py`)
   HiggsAudioV2 RVQ codebook lookup → `Linear(1024 → 256)` → DAC
   acoustic decoder (transposed-conv stack) → 24 kHz mono waveform.
   Frame rate is 25 Hz, so each token row → 960 samples. Output:
   `[B, 1, target_len * 960]` float32 PCM.

The `OmniVoicePipeline` (`vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py`)
wraps both stages plus tokenization into one `forward(req)` call. It is
the request-mode entry point — no explicit step-by-step scheduling at this
level; the diffusion engine fires the whole pipeline per-request and the
32-step loop lives entirely inside the generator.

### Stage diagram

```
                                                        VLLM_OMNI_OMNIVOICE_NUM_STEP
                                                        (default 32)
                                                              │
text  ─►  HFTokenizer  ─►  text_ids [N_text]                  │
                              │                               │
ref_audio ─► HiggsAudioV2     │                               ▼
              tokenizer  ─►  ref_ids [8, T_ref]   ┌──────────────────────────┐
                              │                  │  OmniVoiceGenerator      │
                              ▼                  │                          │
                       cond_ids [8, S_cond]      │  init: tokens = MASK     │
                       uncond_ids [8, T_target]  │                          │
                              │                  │  for step in 0..31:      │
                              ▼                  │    embed cond+uncond     │
                       batch [2*B, 8, S]   ──►   │    transformer_forward   │
                       audio_mask [2*B, S]       │    log_softmax + CFG     │
                       attention_mask            │    score = max(log_p)    │
                                                 │           + gumbel_noise │
                                                 │           - layer_pen.   │
                                                 │    topk(score, k_step)   │
                                                 │    write tokens at topk  │
                                                 │  end loop                │
                                                 └──────────────────────────┘
                                                              │
                                              tokens [B, 8, target_len]
                                                              │
                                                              ▼
                                                ┌──────────────────────────┐
                                                │ OmniVoiceDecoder         │
                                                │  RVQ lookup + sum        │
                                                │  Linear(1024→256)        │
                                                │  DAC acoustic decoder    │
                                                │   (ConvTranspose1d stack)│
                                                └──────────────────────────┘
                                                              │
                                              waveform [B, 1, T*960] @ 24 kHz
```

### Shapes flowing through

| Stage                               | Tensor                                                                   | dtype           |
|------------------------------------|--------------------------------------------------------------------------|------------------|
| text tokens                         | `text_ids: [N_text]`                                                     | int64            |
| replicated text per codebook        | `text_ids: [8, N_text]`                                                  | int64            |
| reference audio (optional)          | `ref_ids: [8, T_ref]`                                                    | int64            |
| target region (initially all MASK)  | `target_ids: [8, T_target]` with value `mask_id=1024`                    | int64            |
| conditional row                     | `cond_ids: [8, S_cond]` (text + ref + target, padded to bucket)          | int64            |
| unconditional row                   | `uncond_ids: [8, S_uncond]` (target only, padded to bucket)              | int64            |
| batched generator input             | `input_ids: [2*B, 8, S]` (cond rows 0..B-1, uncond rows B..2B-1)         | int64            |
| audio_mask                          | `[2*B, S]` bool — True at audio positions                                | bool             |
| attention_mask (full bidirectional) | `[2*B, 1, S, S]` bool                                                    | bool             |
| transformer hidden states           | `[2*B, S, 1024]` (`llm_hidden_size`)                                     | bf16 (default)   |
| logits                              | `[2*B, 8, S, 1025]`                                                      | float32          |
| log-probs after CFG                 | `[1, 8, T_target, 1025]` per request                                     | float32          |
| selected tokens (per step)          | added to `tokens: [B, 8, max_target_len]` at top-k positions             | int64            |
| decoder output waveform             | `[B, 1, T_target * 960]`                                                 | float32          |

`B` is the batch size in a single generator forward (= number of requests
merged into one diffusion-step). `2*B` because every request is replicated
into a conditional and an unconditional row for classifier-free guidance
(CFG).

The "fact" that the unconditional sequence has only the target region (no
text) is what makes CFG meaningful here: the unconditional log-probs
represent "what the model would emit with no text guidance," and the CFG
formula amplifies the cond-vs-uncond delta.

---

## 2. The generator forward loop

File: `vllm_omni/model_executor/models/omnivoice/omnivoice_generator.py`
(781 lines total).

The 32-step loop is in `OmniVoiceGenerator.forward()` at
**`omnivoice_generator.py:504-700`**. This is the function Fast-dLLM has to
wrap or rewrite.

### 2.1 Inputs to `forward()`

Signature at `omnivoice_generator.py:504-517`:

```python
@torch.inference_mode()
def forward(
    self,
    input_ids: torch.Tensor,         # [2*B, 8, S]  cond rows then uncond rows
    audio_mask: torch.Tensor,        # [2*B, S] bool, True at audio positions
    attention_mask: torch.Tensor,    # [2*B, 1, S, S] bool, full bidirectional
    target_lens: list[int],          # length B, per-request target audio length
    num_step: int = 32,
    guidance_scale: float = 2.0,
    t_shift: float = 0.1,
    layer_penalty_factor: float = 5.0,
    position_temperature: float = 5.0,
    class_temperature: float = 0.0,
) -> torch.Tensor:
```

Returns `tokens: [B, 8, max_target_len]` int64. Already-unmasked positions
have the predicted codebook IDs; positions beyond the per-request
`target_lens[i]` are still `mask_id=1024` (the caller is expected to crop
each row to its target length before decoding, see
`pipeline_omnivoice.py:323-336`).

The B requests are *not* independent: they share one `[2*B, 8, S]`
forward, padded to a common `S` chosen by the bucket schedule
(`_FULL_SEQUENCE_BUCKETS = (640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664)`,
`omnivoice.py:48-50`). Per-request masks decouple them inside attention
(see "Risks" below).

### 2.2 Step-0 input: the masked target region

At `omnivoice_generator.py:564-569`:

```python
tokens = torch.full(
    (B, num_codebooks, max_target_len),
    mask_id,                         # = config.audio_mask_id = 1024
    dtype=torch.long,
    device=device,
)
```

This is a separate working tensor that tracks which positions have been
unmasked in the canonical "B-side" space (one row per request). The
`input_ids` tensor passed in already has `[mask_id]` filled into the
target region of both the cond and uncond rows by the caller
(`omnivoice.py:401` and `pipeline_omnivoice.py:213-214`).

`input_ids` is cloned once before the loop so it can be mutated in place
each step (`omnivoice_generator.py:604`):

```python
input_ids = input_ids.clone()
```

This was a 2026-04-29 perf fix — cloning per `(step, i)` was allocating
`B * num_step` tensors per request.

### 2.3 Final output at step 32

After the loop ends (line 700), `tokens` has every position in
`[0, target_lens[i])` filled with predicted IDs from `[0, 1023]` (the
mask token is excluded by `log_probs[..., mask_id] = -float("inf")` at
`omnivoice_generator.py:660`). Trailing positions
`[target_lens[i], max_target_len)` may still be `mask_id`; the pipeline
post-fills those with `tokens[..., target_lens[i]-1]` before decoding
(`pipeline_omnivoice.py:325-329`) so the DAC convolution receptive field
sees a clean continuation rather than OOV mask IDs.

### 2.4 Per-step structure

The full step body (one of 32) is `omnivoice_generator.py:612-693`. Key
phases:

#### a) Cudagraph step boundary marker (line 617-618)

```python
if cudagraphs_enabled:
    torch.compiler.cudagraph_mark_step_begin()
```

Only fires when `VLLM_OMNI_OMNIVOICE_COMPILE_MODE` is `reduce-overhead`
or `max-autotune`. Without this, torch flags tensor-aliasing errors
because subsequent steps reuse the same compiled-module output buffer.

#### b) Embedding + transformer + projection (line 619-624)

```python
with prof.section("gen.transformer"):
    inputs_embeds = self._prepare_embeddings(input_ids, audio_mask)
    hidden_states = self._transformer_forward(inputs_embeds, attention_mask)
    batch_logits = self._get_logits(hidden_states).to(torch.float32)
    # batch_logits: [2*B, 8, S, 1025]
```

This is the dominant cost. ~100% of the wall time at high B is here.
`_prepare_embeddings` (`omnivoice_generator.py:395-418`) does:

- text path: `text_embedding(input_ids[:, 0, :])` — one row of the 8
  codebooks suffices because text tokens are replicated identically
  across all 8 (`omnivoice.py:398`, `pipeline_omnivoice.py:212`)
- audio path: `audio_embeddings(input_ids + codebook_layer_offsets)`
  summed across the codebook dimension; offsets shift each codebook's
  IDs into its own slice of the `8*1025=8200`-entry shared embedding
- merge by `torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)`

`_transformer_forward` (`omnivoice_generator.py:420-484`) precomputes
RoPE and (optionally) flash-attn varlen unpacking metadata once per
step, then runs 28 `OmniVoiceTransformerBlock` layers in sequence. Each
layer has Qwen3-style GQA self-attention (16 q heads, 8 kv heads,
head_dim=64) → SwiGLU MLP. RMSNorm pre-attention and pre-MLP.

`_get_logits` (`omnivoice_generator.py:486-502`):

```python
logits_flat = self.audio_heads(hidden_states)  # [B, S, 8*1025]
return logits_flat.view(B, S, 8, 1025).permute(0, 2, 1, 3)  # [B, 8, S, 1025]
```

Single fused `Linear(1024, 8200)` then reshape — there is no per-codebook
head, the 8 codebooks share the projection.

#### c) Log-softmax for cond and uncond (line 630-635)

```python
with prof.section("gen.log_softmax"):
    cond_log_probs = F.log_softmax(batch_logits[:B], dim=-1)
    uncond_log_probs = (
        F.log_softmax(batch_logits[B:], dim=-1)
        if guidance_scale != 0 else None
    )
```

Batched once per step over the full `[B, 8, S, 1025]` and `[B, 8, S, 1025]`
slabs, even though only target positions matter — the CFG combination is
done per-request below.

#### d) Per-request unmask logic (line 637-693)

This is the **innermost loop** Fast-dLLM has to modify. Quoted in full:

```python
with prof.section("gen.per_i"):
    for i in range(B):
        k = schedules[i][step]
        if k <= 0:
            continue

        c_len = c_lens[i]
        t_len = target_lens[i]

        # View into the per-step log-probs (no compute, no copy).
        c_log_probs = cond_log_probs[i : i + 1, :, c_len - t_len : c_len, :]

        # Classifier-free guidance
        if guidance_scale != 0:
            u_log_probs = uncond_log_probs[i : i + 1, :, :t_len, :]
            log_probs = torch.log_softmax(
                c_log_probs + guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = c_log_probs

        # Prevent predicting [MASK]
        log_probs[..., mask_id] = -float("inf")

        # Token prediction
        if class_temperature > 0.0:
            pred_tokens = _gumbel_sample(log_probs, class_temperature).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)  # [1, 8, T]

        # Confidence scores
        scores = log_probs.max(dim=-1)[0]  # [1, 8, T]

        # Layer penalty (earlier codebooks get higher priority)
        scores = scores - (layer_ids * layer_penalty_factor)

        # Gumbel noise for position selection
        if position_temperature > 0.0:
            scores = _gumbel_sample(scores, position_temperature)

        # Mask out already unmasked positions
        sample_tokens = tokens[i : i + 1, :, :t_len]
        scores.masked_fill_(sample_tokens != mask_id, -float("inf"))

        # Select top-k positions to unmask
        _, topk_idx = torch.topk(scores.flatten(), k)
        flat_tokens = sample_tokens.flatten().clone()
        flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
        sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

        # Update tokens and batch inputs for next iteration. We
        # already cloned input_ids once at the top of forward(),
        # so in-place index-assignment here is safe.
        tokens[i : i + 1, :, :t_len] = sample_tokens
        input_ids[i, :, c_len - t_len : c_len] = sample_tokens.squeeze(0)
        input_ids[B + i, :, :t_len] = sample_tokens.squeeze(0)
```

This is the piece Fast-dLLM most directly replaces. It currently:

- chooses `k = schedules[i][step]` deterministic top-k positions
- the score it sorts on is `max log-prob over vocab − layer_penalty −
  Gumbel(temperature=position_temperature)`
- never reverts decisions: once a position is unmasked it gets
  `-inf` score for all future steps (line 680)
- the `pred_tokens` for the unmasked positions is `argmax` over vocab
  (with optional class-temperature Gumbel noise at line 664)

#### e) Schedule construction (line 571-589)

`k = schedules[i][step]` comes from a fixed time-shifted schedule
computed before the loop:

```python
timesteps = _get_time_steps(0.0, 1.0, num_step + 1, t_shift).tolist()
schedules = []
for t_len in target_lens:
    total_mask = t_len * num_codebooks
    rem = total_mask
    sched = []
    for step in range(num_step):
        num = (
            rem
            if step == num_step - 1
            else min(
                math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                rem,
            )
        )
        sched.append(int(num))
        rem -= int(num)
    schedules.append(sched)
```

`_get_time_steps` (`omnivoice_generator.py:55-68`) implements the time-
shift formula `r_n = t_shift * (n/N) / (1 + (t_shift - 1) * (n/N))` —
with `t_shift=0.1` (default) it's biased to unmask **more** tokens
early. The schedule `sum(sched) == total_mask = t_len * 8`, so by step
32 every position in every codebook has been unmasked exactly once.

This is the schedule Fast-dLLM replaces with "unmask everything whose
confidence > threshold this step, but at least 1 per block, at most
some `block_size` total."

### 2.5 The `gumbel_seed` env hook (just landed)

`omnivoice_generator.py:518-533`, executed at the very top of `forward()`:

```python
# Optional reproducible-RNG hook for A/B quality testing. When the
# env var is set, every forward() resets torch's RNG so the gumbel
# sampling produces the same voice across calls. Default off; F12
# baseline behavior is unchanged unless the operator opts in.
_seed_env = os.environ.get("VLLM_OMNI_OMNIVOICE_GUMBEL_SEED")
if _seed_env:
    try:
        _seed = int(_seed_env)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
    except ValueError:
        logger.warning(
            "VLLM_OMNI_OMNIVOICE_GUMBEL_SEED=%r is not an int; ignoring",
            _seed_env,
        )
```

This confirms the only RNG entry points in the loop are:
- `_gumbel_sample(log_probs, class_temperature)` at line 664
  (only when `class_temperature > 0`; default 0 → never fires)
- `_gumbel_sample(scores, position_temperature)` at line 676
  (always fires; default `position_temperature=5.0`)

`_gumbel_sample` (`omnivoice_generator.py:71-74`) is:

```python
def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-8)))
    return logits / max(temperature, 1e-8) + noise
```

For Fast-dLLM, the position-selection Gumbel at line 676 is the noise
that has to be removed (or measured before adding) when computing the
"true" softmax confidence used for thresholding. With the default
`position_temperature=5.0`, the Gumbel noise dwarfs the layer-penalty
delta and partially the log-prob magnitude — that's the whole point of
the noise, it's a stochastic position selector, not a confidence sort.
Confidence-based decoding has to either:

1. measure confidence on `log_probs.max(-1)` *before* adding Gumbel,
   then use the threshold as the gating function (preferred),
2. or set `position_temperature=0` for confidence runs (loses the
   position-stochasticity that produces voice variability — auto-voice
   would degenerate).

### 2.6 Where each parameter ends up

| Parameter                | Default | First use                        | Effect                                                                 |
|--------------------------|---------|----------------------------------|------------------------------------------------------------------------|
| `num_step`               | 32      | line 612 `for step in range(...)` | Loop iteration count = total transformer forwards per request          |
| `guidance_scale`         | 2.0     | line 650 (CFG mix)               | Strength of CFG; 0 → use cond log-probs directly                       |
| `t_shift`                | 0.1     | line 572 schedule construction   | Front-loads unmasking schedule (more unmasks in early steps)           |
| `layer_penalty_factor`   | 5.0     | line 672                         | Subtracted as `layer_id * factor` — biases score so codebook 0 is unmasked first, codebook 7 last |
| `position_temperature`   | 5.0     | line 676 (Gumbel)                | Stochastic position selection; 0 disables noise                        |
| `class_temperature`      | 0.0     | line 663                         | Token-prediction Gumbel; 0 disables noise (greedy argmax)              |
| `audio_mask_id` (=1024)  | (config) | line 660                         | Forbidden output token (always `-inf` log-prob)                        |
| `audio_vocab_size` (=1025)| (config) | logits shape                     | Per-codebook vocab size including the mask token                       |
| `num_audio_codebook` (=8) | (config) | layer_ids shape                  | Number of parallel RVQ streams                                         |

`layer_ids` at line 591 is `[1, 8, 1]` — broadcast against the score
tensor `[1, 8, T]`. So earlier codebooks get a less-negative penalty
and are preferred for early unmasking. This is one of the things
Fast-dLLM has to be aware of — confidence thresholding has to either
preserve this prior or break it.

### 2.7 Logits and masked-token tensor shapes per step

| Tensor                                                     | Shape                  |
|------------------------------------------------------------|------------------------|
| `inputs_embeds`                                            | `[2*B, S, 1024]`       |
| `hidden_states` (per-layer, after final norm)              | `[2*B, S, 1024]`       |
| `batch_logits = self._get_logits(hidden_states).fp32`      | `[2*B, 8, S, 1025]`    |
| `cond_log_probs`                                           | `[B, 8, S, 1025]`      |
| `uncond_log_probs`                                         | `[B, 8, S, 1025]` or None |
| Per-i `c_log_probs` (cond, target slice)                   | `[1, 8, t_len, 1025]`  |
| Per-i `u_log_probs` (uncond, target slice)                 | `[1, 8, t_len, 1025]`  |
| Per-i `log_probs` (after CFG, after mask-id mask)          | `[1, 8, t_len, 1025]`  |
| `pred_tokens = log_probs.argmax(-1)`                       | `[1, 8, t_len]`        |
| `scores = log_probs.max(-1).values`                        | `[1, 8, t_len]`        |
| `tokens` (working state, all B requests)                   | `[B, 8, max_target_len]` |

`S` is the bucket-rounded total sequence length (one of `(640, 768, 896,
1024, 1152, 1280, 1408, 1536, 1664)`); `t_len` is the per-request target
length (frames @ 25 Hz). For one request, target audio of 1 s = 25
target tokens, 4 s = 100 target tokens, etc.

---

## 3. Sampling / token selection — the Fast-dLLM insertion point

### 3.1 Current selection — exactly two argmaxes per step per request

For each `(i, step)`:

1. Token prediction (what value to write at chosen positions):
   - `pred_tokens = log_probs.argmax(dim=-1)` (line 666)
   - or `_gumbel_sample(log_probs, class_temperature).argmax(-1)` if
     `class_temperature > 0` (line 664) — never enabled in default
     OmniVoice config (`class_temperature=0.0`).

2. Position selection (which positions to unmask this step):
   - confidence baseline: `scores = log_probs.max(-1).values` (line 669)
   - layer prior: `scores -= layer_ids * layer_penalty_factor` (line 672)
   - position-stochasticity: `scores = _gumbel_sample(scores,
     position_temperature)` (line 676) — adds Gumbel(0,1) noise scaled
     by `1/position_temperature`
   - already-unmasked exclusion: `scores.masked_fill_(sample_tokens
     != mask_id, -inf)` (line 680)
   - top-k pick: `_, topk_idx = torch.topk(scores.flatten(), k)` (line 683)

`k` is from the deterministic schedule (line 639). The number of
unmasks per step is fixed; only which positions get those unmasks is
data-dependent.

### 3.2 What Fast-dLLM changes

Fast-dLLM's confidence-based parallel decoding swaps the schedule-driven
top-k with a threshold-driven gate: "unmask all positions where
`max softmax_prob > threshold`, capped at some `small_block_size`
positions per step." The natural transformations:

- compute confidence as `softmax(log_probs).max(-1).values` (or
  equivalently `exp(log_probs).max(-1).values` since `log_probs` is
  already log-softmax) — *before* adding the Gumbel noise at line 676.
- select positions where `confidence > threshold`, with a min/max cap
  controlled by `block_size` (this becomes the new `k`).
- if confidence is below threshold for all unmasked positions, fall
  back to the schedule's `k` (or just keep one minimum forward-progress
  unmask so the loop doesn't deadlock).
- early-exit the loop once all target positions are unmasked (line
  680's `sample_tokens != mask_id` check tells you that).

Insertion site: replace lines **669-685** with confidence-threshold
logic. Keep the layer penalty (line 672) — it's a structural prior, not
a confidence proxy. Keep the `mask != mask_id` filter (line 680) — it's
the "don't re-decide" guard. Replace the fixed-`k` top-k with a
threshold-mask `topk` whose `k` is `max(min_block_size, count_above_thr)`,
clamped to `≤ small_block_size` and `≤ remaining_masks`.

### 3.3 What "ready to unmask" currently means

A position is "ready" when its `score` is in the top-`k` of all still-masked
positions in this request, where the score has the layer-penalty bias and
Gumbel noise applied. There is no explicit confidence threshold today —
the model commits to unmasking exactly `k_step` positions every step,
even if all of them have low confidence.

That's why Fast-dLLM is interesting here: when the confidence is high,
many more than `k` positions are *actually* committable, but the schedule
is leaving them in for next step. Confidence-based unlocks ~`32/k_avg`
fewer total transformer passes, where `k_avg` is the average ratio of
confident-positions-per-step to scheduled-positions-per-step.

---

## 4. KV-cache behavior — there is no KV cache

**Key finding: the OmniVoice generator does NOT maintain a KV cache
across the 32 unmask steps.** Every step is a full bidirectional
transformer pass over the entire `[2*B, 8, S]` sequence.

Evidence:

- `OmniVoiceAttention.forward` (`omnivoice_generator.py:115-201`) takes
  `hidden_states`, recomputes `q = q_proj(...)`, `k = k_proj(...)`,
  `v = v_proj(...)` from scratch, applies QK norm + RoPE, then runs
  `flash_attn_varlen_func` or `F.scaled_dot_product_attention` with the
  full attention mask. No `past_key_values` argument, no cache state, no
  cache-position bookkeeping.
- `OmniVoiceTransformerBlock.forward` (lines 227-255) just chains
  attention → MLP with residuals. No cache passthrough.
- `_transformer_forward` (lines 420-484) iterates layers in a vanilla
  for-loop. No cache context object.
- `grep "kv_cache\|past_key_values\|cache_position"
  vllm_omni/model_executor/models/omnivoice/` returns nothing.

This is consistent with the model architecture: bidirectional attention
over a fixed-length sequence where positions are revealed via masking
rather than appended autoregressively. The K/V at every position depend
on the values at every other position (via the RMSNorm and RoPE-modulated
QK projections of *all* positions), and as positions get unmasked their
embeddings change (audio embedding for newly-decided token vs. mask-token
embedding), so the K/V at unchanged positions can shift step-to-step
because attention is full-bidirectional and the *queries* into them now
come from differently-embedded queries.

### 4.1 Implication for Fast-dLLM's KV-cache approximation

Fast-dLLM's KV-cache reuse trick (cache the K/V of "frozen" positions
between decode steps and only recompute K/V for positions whose
embedding changed this step) is **not a tweak** to existing OmniVoice
machinery — there is no cache to tweak. It's a from-scratch piece of
infrastructure if you want it.

Sketch of what would have to change:
- introduce per-layer K and V tensors of shape `[2*B, S, num_kv_heads,
  head_dim]` stored on the generator object (or a transient context
  passed through the loop) and zero them at step 0.
- in `OmniVoiceAttention.forward`, accept a `kv_cache_mask: [2*B, S]`
  (True at positions whose K/V are stale and need recompute) and a
  `kv_cache_state` ref. For False positions, read K/V from cache; for
  True positions, recompute and write back.
- mark positions stale every step at: (a) all already-stale ones from
  previous step (cheap union), (b) positions newly unmasked this step
  (because their embedding flipped from mask-embed to audio-embed), and
  (c) (more subtle) positions whose attention output was a function of
  newly-unmasked positions. Fast-dLLM's approximation is to skip (c)
  — i.e., reuse the cache despite the formal staleness — and accept
  that as the quality-vs-speed knob.
- handle cudagraph compatibility (the generator's compiled layers
  capture K/V projection inputs; threading a cache through breaks the
  graph signature unless the cache tensor is a captured buffer).

Significant work; see Section 8 for the line-by-line insertion spots.
The `OPT=1` path uses `torch.compile(layer)` per layer (line 372),
which captures the layer signature — adding cache args means you have
to either bypass `_enable_compile_optimizations` or recompile against
the new signature.

### 4.2 Bonus: there's no KV cache in the diffusion-engine attention layer either

`vllm_omni/diffusion/attention/layer.py` is the generic vllm-omni
diffusion attention wrapper, but OmniVoice doesn't use it — the
generator implements its own attention using `flash_attn_varlen_func`
or `F.scaled_dot_product_attention` directly. The flashinfer
`single_prefill_with_kv_cache` paths in
`diffusion/attention/backends/ring/` are for ring-parallel diffusion
models (image/video DiTs), not OmniVoice. So there's no shared cache
infrastructure to plug into.

---

## 5. Batching layer (`vllm_omni/diffusion/sched/base_scheduler.py`)

This file is the brain of how requests turn into B>1 generator forwards.
401 lines, all worth reading.

### 5.1 Three interacting variables

- `self.max_num_running_reqs` — caps how many WAITING → RUNNING in one
  `schedule()` call. Set by `VLLM_OMNI_DIFFUSION_BATCH_SIZE` (default 1).
- `self._batch_strategy` — `fifo` (default) or `duration_bucket`.
- `self._duration_bucket_tokens` — bucket width in target audio frames
  (default 128).
- `self._pad_tolerance` — alternative to fixed-bucket (default 1.0 = strict
  bucket equality, opt-in via `VLLM_OMNI_DIFFUSION_PAD_TOLERANCE > 1.0`).

### 5.2 The duration-bucket admission flow

`_select_duration_bucket_waiting` at `base_scheduler.py:308-358`:

1. Anchor on the first WAITING request (FIFO order) — it sets the bucket
   for this dispatch.
2. Walk the WAITING deque in order; pull each peer that lands in the
   same bucket (or within `pad_tolerance` of the anchor's target_tokens
   estimate, if tolerance > 1.0); skip and *defer* peers in other
   buckets.
3. Stop when capacity (`max_num_running_reqs`) reached.
4. Push deferred peers back to the front of the deque so they're
   considered next dispatch.

`_duration_bucket()` (line 360-362):
```python
def _duration_bucket(self, state) -> int:
    target_tokens = self._estimate_target_tokens(state)
    return max(0, target_tokens // self._duration_bucket_tokens)
```

`_estimate_target_tokens` (line 364-378) calls
`RuleDurationEstimator.estimate_duration(text, "Nice to meet you.", 25)`
to get the target audio length in 25-Hz frames; falls back to character
length if the duration estimator failed to load.

### 5.3 The `_running` gate

`base_scheduler.py:149`:
```python
if self._batch_strategy == "duration_bucket" and not self._running:
    waiting_to_schedule = self._select_duration_bucket_waiting(...)
else:
    waiting_to_schedule = self._select_fifo_waiting(...)
```

This is the gate the PERF_A100.md doc flags: with `CONCURRENT=2`, once
one merged batch is in-flight, the next `schedule()` call sees
`self._running = [...]` and falls back to FIFO single-request path. So
`CONCURRENT≥2` silently regresses to `B=1` per forward.

### 5.4 The BATCH_WAIT_MS path

This is in the engine, not the scheduler:
`vllm_omni/diffusion/diffusion_engine.py:445-478`. The engine driver
loop, before calling `scheduler.schedule()`, sleeps up to
`VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS` to let stragglers arrive. Polls
every `batch_wait_ms / 4000` seconds and exits early if the waiting
queue has hit `max_num_running_reqs` or stops growing.

The default is 0 (no wait); the F12 A100 config uses 30 ms. Without
this knob, c=8 gets ~90% solo (B=1) forwards because scheduling
happens before all 8 HTTP coroutines have called `add_request`.

### 5.5 Invariants the generator relies on

1. **Uniform shape per batch.** The generator's `forward()` takes
   `input_ids: [2*B, 8, S]` with one shared `S`. The pipeline pads each
   request's cond/uncond to `max(cond_len, uncond_len)` and then to a
   bucket size (`omnivoice.py:422` calls `_full_sequence_bucket(...)`,
   which rounds up to one of `(640, 768, 896, ..., 1664)`).
2. **Shared `num_step`.** All B requests in one forward run the *same*
   number of unmask steps. Currently 32 by default; settable via
   `VLLM_OMNI_OMNIVOICE_NUM_STEP`. There's no per-request step count.
3. **Shared CFG.** All B requests use the same `guidance_scale`. There
   is no per-request CFG today.
4. **Per-request `target_lens`.** The one thing that *can* differ per
   request inside the batch is the target audio length. The schedule is
   computed per-request (`omnivoice_generator.py:573-589`), and the
   per-i unmask body already crops to `t_len` for log-prob extraction
   and topk.
5. **Shared `S` is wasteful for short members.** The transformer cost
   is `O(B * S^2)` so a B=8 batch where one member needs S=1664 forces
   the other 7 to also pay S=1664. Bucketing exists exactly to keep
   this manageable.

### 5.6 What confidence-decoding breaks

If Fast-dLLM does **per-sample early-exit** (request `i` finishes after
step 12 because all of its positions cleared the threshold, while other
requests need 32), the current generator forward has no way to express
that — every step's transformer pass is monolithic over all `2*B` rows.
You either:

- early-exit the **batch**: stop the loop the step where *every*
  request hits zero remaining masks. This is a strictly-monotone
  improvement (you only stop when every request agrees), but the
  speedup is gated by the slowest request in the batch.
- "freeze" finished requests: keep them in the batch with their tokens
  filled in (so attention still includes them), but stop computing
  their per-i unmask logic (the per-i loop body already has `if k <= 0:
  continue` — confidence decoding can replicate that). This costs the
  full transformer per step but skips the per-i softmax/topk, which is
  not the dominant cost (the per-i section is `gen.per_i` in profiler;
  gen.transformer dominates).
- truly per-request step counts: needs a rewrite that splits the
  forward by per-request done-ness or makes the scheduler aware of
  remaining steps and re-batches each step (harder than it sounds —
  the input_ids tensor shape and CFG-pairing assumes a stable B).

The duration-bucket scheduler tries to put requests with similar
target_lens together precisely so the schedule lengths are similar. So
in practice the "stop when all done" approach is fine: members of a
bucket-aligned batch will mostly agree on when they're done, and the
slow-tail amortizes across the bucket.

---

## 6. Env-var control surface

Every `VLLM_OMNI_*` env var the model + scheduler check, what each
does, defaults, and the line where they're read.

### Generator (`vllm_omni/model_executor/models/omnivoice/`)

| Var                                       | Default      | File:Line                        | Effect                                                                                            |
|-------------------------------------------|--------------|----------------------------------|---------------------------------------------------------------------------------------------------|
| `VLLM_OMNI_OMNIVOICE_OPT`                 | unset (off)  | `omnivoice_generator.py:30`      | When `=1`, applies the bf16/tf32/SDPA backend toggles + per-layer + audio_heads `torch.compile` (`omnivoice_generator.py:348-382`). Also triggers full-sequence bucket rounding in `omnivoice.py:53-65`. |
| `VLLM_OMNI_OMNIVOICE_USE_FLASH_ATTN`      | unset (off)  | `omnivoice_generator.py:34`      | When `=1`, attention uses flash-attn varlen path (precomputed indices) instead of SDPA fallback.   |
| `VLLM_OMNI_OMNIVOICE_COMPILE_MODE`        | `default`    | `omnivoice_generator.py:369, 610`| `default` | `reduce-overhead` | `max-autotune`. The latter two enable cudagraphs; `reduce-overhead` requires `cudagraph_mark_step_begin()` per step (line 618). |
| `VLLM_OMNI_OMNIVOICE_GUMBEL_SEED`         | unset        | `omnivoice_generator.py:522`     | Integer seed; if set, every `forward()` calls `torch.manual_seed(seed)` to lock the auto-voice for A/B testing. |
| `VLLM_OMNI_OMNIVOICE_LOG_BATCH`           | unset (off)  | `omnivoice_generator.py:552`     | When `=1`, logs the `B` and wall ms of every forward — the diagnostic that surfaced the c=8 batching problem. |
| `VLLM_OMNI_OMNIVOICE_PROFILE`             | unset (off)  | `profiling.py:28`                | Enables the per-section CUDA-event profiler used by `prof.section(...)` calls in the generator + pipeline. |
| `VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE`     | unset        | `pipeline_omnivoice.py:46`       | `bf16`/`bfloat16`/`fp16`/`float16`/`fp32`. Cast applied to the generator only (decoder stays fp32). |
| `VLLM_OMNI_OMNIVOICE_NUM_STEP`            | 32 (config)  | `pipeline_omnivoice.py:134`      | Override `config.num_step`. Setting to 16 ~doubles throughput but quality A/B is open.            |
| `VLLM_OMNI_OMNIVOICE_PREWARM_BUCKETS`     | unset (off)  | `pipeline_omnivoice.py:365`      | When `=1`, runs synthetic forwards at every bucket size at startup to fill the cudagraph cache.   |

### Scheduler / engine (`vllm_omni/diffusion/`)

| Var                                              | Default       | File:Line                                                | Effect                                                                                            |
|--------------------------------------------------|---------------|----------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `VLLM_OMNI_DIFFUSION_BATCH_SIZE`                 | 1             | `base_scheduler.py:55`                                   | `max_num_running_reqs` cap. F12 uses 12.                                                          |
| `VLLM_OMNI_DIFFUSION_BATCH_STRATEGY`             | `fifo`        | `base_scheduler.py:67`                                   | `fifo` or `duration_bucket`.                                                                      |
| `VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS`     | 128           | `base_scheduler.py:77`                                   | Bucket width in target frames (25 Hz). F12 uses 256.                                              |
| `VLLM_OMNI_DIFFUSION_PAD_TOLERANCE`              | 1.0           | `base_scheduler.py:90`                                   | If > 1.0, batches by max/min target ratio rather than fixed buckets. Default 1.0 = strict bucket. |
| `VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS`              | 0             | `diffusion_engine.py:124`                                | Coalesce-wait window before dispatching first batch. F12 uses 30.                                  |
| `VLLM_OMNI_DIFFUSION_DRIVER_IDLE_SLEEP`          | 0.0005        | `diffusion_engine.py:117`                                | Driver-loop idle sleep when no work.                                                              |
| `VLLM_OMNI_DIFFUSION_CONCURRENT`                 | unset (off)   | `diffusion_engine.py:107`                                | When `=1`, runs the engine in concurrent driver mode. Note: combined with `BATCH_SIZE=2` regresses to B=1 due to `_running` gate (PERF_A100.md, Section 5.3). |

### Confidence-decoding env vars planned but **NOT yet implemented**

The README at `benchmarks/tts/README.md:178-179` references:

- `VLLM_OMNI_OMNIVOICE_CONFIDENCE_THRESHOLD` — does not exist in code.
  `grep -rn "CONFIDENCE_THRESHOLD" vllm_omni/` returns no hits in
  source code.
- `VLLM_OMNI_OMNIVOICE_CONFIDENCE_SMALL_BLOCK_SIZE` — does not exist in
  code either.

These are the knobs the colleague will be **adding**. Suggested
implementation:

- read both in `OmniVoiceGenerator.forward()` next to the existing
  `_seed_env` block (around `omnivoice_generator.py:522-533`), parse
  to float / int, validate ranges (`0 < thr <= 1.0`, `block_size >= 1`).
- thread them through the per-i loop body (lines 637-693) where the
  current schedule-driven `k = schedules[i][step]` lives.
- with both unset, behavior must be identical to today (perf baseline
  contract).

---

## 7. Performance properties

### 7.1 Current operating point (A100 80GB, F12 baseline)

From `benchmarks/tts/PERF_A100.md` (NUM_STEP=32, full quality):

| c    | req/s | wall_avg | wall_p95 |
|------|-------|----------|----------|
| 1    | 2.65  | 0.350 s  | 0.359 s  |
| 2    | 4.47  | 0.384 s  | 0.448 s  |
| 4    | 5.90  | 0.521 s  | 0.774 s  |
| 8    | 6.61  | 0.875 s  | 1.237 s  |
| 16   | 5.55  | 1.851 s  | 3.109 s  |
| 32   | 5.16  | 3.148 s  | 4.956 s  |

At `NUM_STEP=16` (current production default in
`run_server_optimized.sh`), wall time roughly halves (PERF_RESULTS.md);
the model still produces audio but quality has not been formally A/B'd.

### 7.2 Where the wall time goes

From the inline `prof.section(...)` instrumentation:

- `gen.transformer` — full transformer + logits projection. Dominant.
  ~80-95% of generator wall at every B.
- `gen.log_softmax` — batched log-softmax over `[2*B, 8, S, 1025]`.
  Small (~1-2 ms).
- `gen.per_i` — the inner per-request loop (CFG mix + topk +
  in-place writeback). Small at small B (~1 ms × B); grows with B.
- `decoder` — RVQ + DAC. ~9% of c=1 wall (PERF_A100.md), ~50 ms × B
  at high B before the 2026-04-29 batched-decode fix (now single call).

So the 32-step loop's transformer time accounts for essentially the
whole generator latency. Throughput improvements by reducing step count
are linear: 32 → 16 ≈ 2× faster generator; 32 → 8 ≈ 4× faster.

### 7.3 Amdahl ceiling for confidence-decoding

Generator is ~85-95% of end-to-end wall (decoder + tokenization +
overhead is the rest). If confidence-decoding lets you drop from 32
steps to ~8 average steps (Fast-dLLM paper claim), and the per-step
cost stays the same (no KV-cache reuse), the speedup is:

- if generator was 90% of wall: 1 / (0.10 + 0.90 * 8/32) = 1 / 0.325 ≈ **3.1×**.
- if generator was 95% of wall: 1 / (0.05 + 0.95 * 8/32) = 1 / 0.2875 ≈ **3.5×**.
- with the per-step KV-cache reuse approximation, each step itself gets
  ~2× faster on the unchanged-positions path: another ~30-50% on top
  of the step-count reduction, so ~4-5× ceiling.

The dominant lever is the step-count reduction, not the KV-cache
trick. Get the threshold-based step elision working first, profile,
then decide if the cache-approximation is worth the implementation
cost.

### 7.4 Why the perf baseline is fragile

- Bucket=256 was chosen empirically; bucket=128 fragments more,
  bucket=512 lifts long-prompt tails. Confidence-decoding may need
  re-tuning if step counts now vary per-request.
- `BATCH_WAIT_MS=30` is the unlock; without it ~90% of c=8 forwards
  are B=1 solo. If confidence-decoding shortens individual sequence
  wall time, the wait window may need to drop too (less benefit from
  waiting if forwards are faster).
- `BATCH_SIZE=12` is a manual cap to avoid OOM/compute saturation on
  A100 with FMHA. If per-request work shrinks via confidence-decoding,
  the cap could probably go higher.

---

## 8. Where to insert Fast-dLLM — concrete pointers

All paths are absolute under `/home/ubuntu/vllm-omni/`.

### 8.1 Confidence threshold check (the primary insertion)

File: `vllm_omni/model_executor/models/omnivoice/omnivoice_generator.py`

Lines **669-685** (the score → topk → write-back block) is what the
threshold replaces. Sketch:

```python
# At line 669 — replace from here through line 685 with:
log_probs_for_conf = log_probs                  # [1, 8, t_len, 1025]
# Mask the mask-id channel before computing confidence
# (already done at line 660 — log_probs[..., mask_id] = -inf).
prob_max, _ = log_probs_for_conf.max(dim=-1)    # [1, 8, t_len] in log-space

# Layer prior (keep this — it's not a confidence proxy)
prob_max = prob_max - (layer_ids * layer_penalty_factor)

# already-unmasked positions get -inf so they're not re-decided
sample_tokens = tokens[i : i + 1, :, :t_len]
prob_max.masked_fill_(sample_tokens != mask_id, -float("inf"))

# Confidence in real-prob space (recall log_probs is log_softmax)
conf = prob_max.exp()                           # in [0, 1] on still-masked rows

# Fast-dLLM thresholding
above = conf >= confidence_threshold
n_above = int(above.sum().item())

if n_above >= small_block_size:
    # Cap unmask count this step
    flat_conf = conf.flatten()
    _, top_idx = torch.topk(flat_conf, small_block_size)
    chosen = top_idx
elif n_above > 0:
    chosen = torch.nonzero(above.flatten(), as_tuple=False).flatten()
else:
    # Fallback: keep schedule's k so the loop makes progress
    # (otherwise loop deadlocks — every step picks zero positions)
    scores = prob_max + ...gumbel...            # current behavior
    _, chosen = torch.topk(scores.flatten(), schedules[i][step])

flat_tokens = sample_tokens.flatten().clone()
flat_tokens[chosen] = pred_tokens.flatten()[chosen]
sample_tokens.copy_(flat_tokens.view_as(sample_tokens))
tokens[i : i + 1, :, :t_len] = sample_tokens
input_ids[i, :, c_len - t_len : c_len] = sample_tokens.squeeze(0)
input_ids[B + i, :, :t_len] = sample_tokens.squeeze(0)
```

Knob locations:
- `confidence_threshold` — read in `forward()` at line 522 area (next
  to `_seed_env`); float in (0, 1].
- `small_block_size` — read alongside; int >= 1. Acts as a per-step cap.

### 8.2 Block-size logic wrapping the step loop

File: `omnivoice_generator.py`, line **612**.

The `for step in range(num_step):` loop. To support the "stop when all
done" early exit, add at the bottom of the per-step body (after line
693):

```python
# Early-exit: if every request has zero remaining masks, stop.
remaining_masks = sum(
    int((tokens[i : i + 1, :, :target_lens[i]] == mask_id).sum().item())
    for i in range(B)
)
if remaining_masks == 0:
    break
```

This costs one device→host sync per step. To avoid that:

```python
remaining = (tokens[..., :max_target_len] == mask_id).any()
if not bool(remaining):  # one CUDA→host sync but only at end of each step
    break
```

For B=1 single-request mode this is cheap. For B>1, all members must
be done before the loop exits; the slowest member dominates. That's
acceptable given duration-bucketing already aligns sequence lengths.

The Fast-dLLM paper's max-block / small-block split corresponds to:
- small_block_size = `block_size` per step cap (above)
- max_block / "completion check" = the early-exit test (this snippet)

### 8.3 KV-cache reuse — new state in attention.py-equivalent

There is no `attention.py` for OmniVoice — the attention is inlined in
`omnivoice_generator.py`. The hooks would go in:

- `OmniVoiceAttention.__init__` (line 97-113) — register K/V cache
  buffers per layer. Probably easier as an external dict keyed by layer
  index, owned by the generator.
- `OmniVoiceAttention.forward` (line 115-201) — branch on a new
  `kv_cache_mask` argument: for cached positions, slice K/V from the
  cache; for uncached positions, recompute via `k_proj/v_proj`.
- `_transformer_forward` (line 420-484) — compute the per-step
  `kv_cache_mask` ([2*B, S] bool, True = recompute) once before the
  layer loop. Recompute is needed at: every position whose embedding
  changed this step (audio_mask True AND tokens just got written this
  step).
- `OmniVoiceGenerator.forward` (line 504-700) — between `for step in
  range(num_step):` (line 612) and the transformer call (line 622),
  compute `kv_cache_mask` from the diff `input_ids_prev_step` vs
  `input_ids` and pass it through.

Caveats:
- `_enable_compile_optimizations` (line 348-382) wraps each layer in
  `torch.compile`. If you change the layer signature, recompilation
  fires; either bypass `OPT=1` for confidence runs, or recompile.
- The QK norm (`q_norm`, `k_norm` in `OmniVoiceAttention.__init__`) is
  per-head — cached values must include the post-norm K, not the
  pre-norm K, to be reusable.
- RoPE (`_apply_rotary_pos_emb`, line 278-286) is position-dependent —
  cached K already has RoPE applied at its position, so cache hits are
  fine if positions don't move (they don't — `S` is fixed within a
  forward).

### 8.4 Don't forget the env-var read

File: `omnivoice_generator.py:518-533` is the natural spot. Add right
below the `_seed_env` block:

```python
_thr_env = os.environ.get("VLLM_OMNI_OMNIVOICE_CONFIDENCE_THRESHOLD")
_block_env = os.environ.get("VLLM_OMNI_OMNIVOICE_CONFIDENCE_SMALL_BLOCK_SIZE")
if _thr_env is not None:
    try:
        confidence_threshold = float(_thr_env)
    except ValueError:
        logger.warning("Invalid VLLM_OMNI_OMNIVOICE_CONFIDENCE_THRESHOLD=%r; ignoring", _thr_env)
        confidence_threshold = None
else:
    confidence_threshold = None

if _block_env is not None:
    try:
        small_block_size = max(1, int(_block_env))
    except ValueError:
        logger.warning("Invalid VLLM_OMNI_OMNIVOICE_CONFIDENCE_SMALL_BLOCK_SIZE=%r; ignoring", _block_env)
        small_block_size = None
else:
    small_block_size = None
```

When either is None, fall through to the existing schedule-driven
unmask. When both are set, use the threshold path.

This preserves the F12 baseline contract: F12 doesn't set these env
vars, so its behavior must be unchanged.

### 8.5 Logging hook for measuring effectiveness

The `VLLM_OMNI_OMNIVOICE_LOG_BATCH=1` path (line 552-557, 695-699)
already emits per-forward `B` and wall_ms. For confidence-decoding
diagnostics, add (gated by the same flag or a new one):

- per-step `n_above` counts (how many positions cleared the threshold)
- per-step `chosen_size` (how many were actually unmasked, capped by
  `small_block_size`)
- final step count vs `num_step` (how often early-exit fired)
- ratio of "fallback" steps (where `n_above == 0`) vs threshold-fired

This is the data needed to tune `confidence_threshold` and
`small_block_size`. Without it, A/B is opaque.

---

## 9. Risks specific to the OmniVoice port — gotchas

### 9.1 Iterative-refinement semantics: positions are not re-decided once committed

Line 680 of `omnivoice_generator.py` filters out already-unmasked
positions in score computation:

```python
scores.masked_fill_(sample_tokens != mask_id, -float("inf"))
```

A naive Fast-dLLM port that skips this step will re-decide previously
committed positions, which can cascade into instability. Preserve this
filter.

This *also* means if you're tempted to add "reverse-mask" rejection
(re-masking a previously-decided position whose confidence dropped due
to neighboring unmasks), you have to flip the semantics — but that's a
larger architectural change than Fast-dLLM proposes.

### 9.2 The Gumbel noise *is* the voice-stochasticity knob

`position_temperature=5.0` (default) means the Gumbel noise standard
deviation is large relative to the layer-penalty delta and confidence
range. Removing it (setting to 0) makes the auto-voice deterministic
given a fixed transformer state. That's fine for confidence-decoding
benchmarks if `VLLM_OMNI_OMNIVOICE_GUMBEL_SEED` is also set, but you
should not measure confidence on the post-Gumbel score —
`scores = log_probs.max(-1)[0]` (line 669) is computed *before* Gumbel
is added at line 676. Confidence thresholding goes on the line-669
value. Gumbel only matters for the position-selection tiebreak when
`n_above >= small_block_size`. Even then, you can either:

- keep Gumbel on the ties (preserves voice variability)
- drop Gumbel for confidence runs (reproducible, but voice may shift)

### 9.3 Layer penalty disrupts the "naive confidence" interpretation

`scores = scores - (layer_ids * layer_penalty_factor)` at line 672 means
codebook 7's score is `5.0 * 7 = 35` log-probs *worse* than codebook 0's
even when actual log-prob is identical. This is a structural prior that
prevents codebook 7 from being committed before codebook 0 — not a
real confidence signal. If you measure confidence as
`prob_max.exp()`, you get the *real* probability without this bias,
which means codebook 7 positions might cross the threshold before
codebook 0 positions do.

Decision: keep the layer prior in the threshold gate. Subtract
`layer_ids * layer_penalty_factor` from the log-prob *before* the
threshold check, so confident-but-late codebooks don't sneak in early.
Or implement per-codebook thresholds. The paper assumes a flat
language-model decode where positions are interchangeable — they
aren't in OmniVoice.

### 9.4 CFG must run before confidence is measured

Line 650 mixes cond and uncond log-probs:

```python
log_probs = torch.log_softmax(
    c_log_probs + guidance_scale * (c_log_probs - u_log_probs), dim=-1,
)
```

The CFG-mixed log-probs are sharper (or flatter) than the bare
conditional log-probs. Measure confidence on the post-CFG `log_probs`
(line 660-669 area), not on `c_log_probs`. Otherwise the threshold
calibration depends on whether CFG is on, which the operator usually
wants invariant.

### 9.5 Per-request `target_lens` differ — schedule construction needs care

Even within a duration-bucket-aligned batch, per-request
`target_lens[i]` can differ by up to `bucket_tokens` (128 by default,
256 in F12). The current schedule (`schedules[i]`) is computed per-i
because of this. Fast-dLLM's "block_size" is naturally per-i too, but
if you implement per-step early-exit you have to track per-i remaining
work, not a global counter.

### 9.6 The CFG split `[2*B, 8, S]` shape is load-bearing

`input_ids[:B]` is conditional (text + ref + target masked), `input_ids[B:]`
is unconditional (target only). They share the same `S` (the
unconditional rows are right-padded). This shape constraint:

- breaks if you try to early-exit per-request and shrink B mid-loop —
  the `B` index used for slicing log-probs (`cond_log_probs[i]` and
  `uncond_log_probs[i]`) is hardcoded to the original batch size.
- breaks if you try to use different sequence lengths per request mid-
  loop — the transformer expects uniform `S`.

Stick with "freeze finished requests in place, don't shrink the batch."

### 9.7 The `self._running` gate makes CONCURRENT≥2 a trap

`base_scheduler.py:149` only fires duration-bucket batching when
`not self._running`. If you set `VLLM_OMNI_DIFFUSION_CONCURRENT=1` and
`VLLM_OMNI_DIFFUSION_BATCH_SIZE >= 2` while also running confidence
decoding, you'll silently get B=1 forwards and your speedup will be
"step-count-only, no batching." See PERF_A100.md Section 5.3.

### 9.8 `mask_id` is the same as `audio_vocab_size - 1`

`config.audio_mask_id = 1024 = audio_vocab_size - 1` (config.py:32-33).
The mask-id zeroing at line 660 (`log_probs[..., mask_id] = -inf`)
assumes this. If you ever flatten over the vocab dimension to get
top-k, the inf must already be in place — Fast-dLLM threshold check
must run *after* this line.

### 9.9 The `audio_mask` argument is not the unmask-mask

Easy to confuse: `audio_mask: [2*B, S]` (line 508) marks which
positions in the sequence are *audio* (vs text). It's static across
the loop. The "which positions still need unmasking" mask is computed
inline via `tokens != mask_id` at line 680. Fast-dLLM logic should
operate on the latter, not the former.

### 9.10 `torch.compile` cache-size is bounded

`omnivoice_generator.py:365`:
```python
torch._dynamo.config.cache_size_limit = 256
```

When `OPT=1`, every distinct `(B, S)` shape gets a compiled cache
entry. If your confidence-decoding port introduces new tensor shapes
(e.g. a `confidence_mask: [B, 8, T]` with variable T), each new T can
recompile. Stick to shapes already in the cache (full `S` masks,
fixed bucket widths) where possible, or keep `OPT=0` during port
development.

### 9.11 Profile sections must stay enclosed for `prof.report()`

`omnivoice_generator.py` uses `prof.section(...)` context managers at
lines 620, 630, 637. If you add new control flow that bypasses these
sections (early-exit before all are hit), `InferenceProfiler.report()`
will still work — sections that were never opened are ignored — but
the `gen.transformer / gen.log_softmax / gen.per_i` ratios will
become harder to interpret because they no longer all run the same
number of times. Consider adding a new section like
`gen.confidence_check` so the profile tells you which sections were
entered.

### 9.12 The `pad_tolerance=1.0` invariant: same-bucket peers share the same `S` bucket up to rounding

Even with strict bucket equality, two members of the same bucket can
have target_lens differing by `bucket_tokens - 1`. If Fast-dLLM relies
on "all batch members have the same target_len for confidence
calibration" it'll need to handle the mismatch.

### 9.13 The cudagraph step boundary

`torch.compiler.cudagraph_mark_step_begin()` at line 618 fires only
when COMPILE_MODE is `reduce-overhead`/`max-autotune`. Confidence-
decoding's variable step count interacts badly with cudagraphs because
the captured graph spans one transformer pass; that's still per-step,
so the marker still works. But the *number* of step invocations
varies per request, which is fine. Cudagraphs were already known
broken on A100 (PERF_A100.md Section "What didn't pay off"); leave
COMPILE_MODE=default for confidence-decoding bring-up.

### 9.14 Quality A/B is mandatory

`benchmarks/tts/ab_quality.py` is the established workflow (README
sections "Audio-quality A/B" and "Stable baseline A/B reference").
Every confidence_threshold and block_size change has to go through
the 5-fixed-prompt A/B with `VLLM_OMNI_OMNIVOICE_GUMBEL_SEED=42`
before it can claim a perf win. The numerical metrics (rms_ratio,
centroid_delta) are sanity checks, not verdicts — a robotic-sounding
output passes the metrics. Listening overrides metrics.

The seed-locking mechanism (line 522-533) makes the A/B reproducible
because Gumbel and class-temperature are the only stochasticity in
the loop. If you add stochasticity (e.g., a probabilistic threshold
fallback), it must be seeded by the same env var or A/B comparisons
break.

### 9.15 No KV cache means "KV-cache approximation" is a from-scratch project

This is the biggest gotcha of the port. The Fast-dLLM paper assumes a
prior KV cache exists (autoregressive decode); the trick is reusing
it across diffusion steps. OmniVoice has no cache. If the colleague
implements only the confidence-threshold step elision (Section 8.1
+ 8.2), they get the step-count savings (~3-4× ceiling). Adding the
cache trick (Section 8.3) is a separate, larger piece of work and
gets another ~30-50% on top.

Document expectations clearly: phase 1 = threshold + early-exit
(self-contained), phase 2 = K/V cache reuse (requires attention
plumbing).

---

## Appendix A — File summary

| File                                                                                  | Role                                                                              | Size      |
|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-----------|
| `vllm_omni/model_executor/models/omnivoice/omnivoice_generator.py`                    | The 32-step generator + Qwen3 transformer (heart of Fast-dLLM port).              | 781 lines |
| `vllm_omni/model_executor/models/omnivoice/omnivoice_decoder.py`                      | RVQ + DAC vocoder (Stage 1).                                                      | 211 lines |
| `vllm_omni/model_executor/models/omnivoice/omnivoice.py`                              | vLLM model adapter (Stage 0/1 routing + bucketing).                               | 532 lines |
| `vllm_omni/model_executor/models/omnivoice/config.py`                                 | `OmniVoiceConfig` (num_step, t_shift, etc.).                                       | 81 lines  |
| `vllm_omni/model_executor/models/omnivoice/duration.py`                               | `RuleDurationEstimator` for target audio length.                                  | 281 lines |
| `vllm_omni/model_executor/models/omnivoice/profiling.py`                              | Per-section CUDA-event profiler.                                                  | 205 lines |
| `vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py`                          | The `OmniVoicePipeline` (request-mode, batched B>1 forward).                      | 442 lines |
| `vllm_omni/diffusion/sched/base_scheduler.py`                                         | The duration-bucket batching brain.                                               | 401 lines |
| `vllm_omni/diffusion/sched/step_scheduler.py`                                         | Per-step continuous batching (alternative to request-mode).                       | 129 lines |
| `vllm_omni/diffusion/sched/request_scheduler.py`                                      | Per-request batching (paired with `OmniVoicePipeline`).                            | 50 lines  |
| `vllm_omni/diffusion/diffusion_engine.py`                                             | Engine driver loop + BATCH_WAIT_MS coalesce.                                      | 600+ lines |
| `examples/online_serving/omnivoice/run_server_optimized.sh`                           | Production launcher with F12 env vars.                                            | 42 lines  |
| `benchmarks/tts/PERF_A100.md`                                                         | A100 perf investigation + recommended config.                                     | 230 lines |
| `benchmarks/tts/PERF_RESULTS.md`                                                      | H100 perf snapshot.                                                               | 135 lines |
| `benchmarks/tts/README.md`                                                            | F12 baseline + A/B workflow + Fast-dLLM env-var hint.                             | 470 lines |

---

## Appendix B — Useful greps

```bash
# Find all VLLM_OMNI env-var reads in the omnivoice path
grep -rn "VLLM_OMNI_" /home/ubuntu/vllm-omni/vllm_omni/model_executor/models/omnivoice/ \
    /home/ubuntu/vllm-omni/vllm_omni/diffusion/

# Confirm there's no KV cache anywhere in OmniVoice
grep -rn "kv_cache\|past_key_values\|cache_position" \
    /home/ubuntu/vllm-omni/vllm_omni/model_executor/models/omnivoice/

# Check whether confidence env vars are wired
grep -rn "CONFIDENCE_THRESHOLD\|CONFIDENCE_SMALL_BLOCK" /home/ubuntu/vllm-omni/

# Find every place num_step is referenced
grep -rn "num_step" /home/ubuntu/vllm-omni/vllm_omni/

# B-histogram diagnostic from the perf doc
VLLM_OMNI_OMNIVOICE_LOG_BATCH=1
# Then grep "OmniVoiceGenerator.forward: B=" in server log
```

---

## Appendix C — Confidence-decoding quick checklist

1. [ ] Read `confidence_threshold` and `small_block_size` env vars in
       `forward()` near line 522.
2. [ ] Replace lines 669-685 with threshold-based selection (Section 8.1).
       Preserve layer penalty, mask-id `-inf`, already-unmasked filter.
3. [ ] Keep schedule-driven `k` as fallback when `n_above == 0`.
4. [ ] Add early-exit at end of step body (Section 8.2).
5. [ ] Verify behavior is identical when both env vars are unset
       (F12 baseline contract).
6. [ ] Run full A/B: `benchmark_concurrent.py` + `ab_quality.py`
       at threshold values 0.6, 0.8, 0.9, 0.95 and block_size 1, 2, 4, 8.
7. [ ] (Optional, larger) Wire KV-cache approximation through
       attention.py-equivalent (Section 8.3).
8. [ ] Update PERF_A100.md and PERF_RESULTS.md with the new numbers.
