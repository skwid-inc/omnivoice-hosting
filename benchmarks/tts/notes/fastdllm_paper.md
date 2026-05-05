# Fast-dLLM: Implementation Notes for OmniVoice Port

> Research notes compiled from the Fast-dLLM project page, arxiv paper (2505.22618), and the
> NVlabs/Fast-dLLM GitHub repo. Audience: an engineer who already knows transformers,
> KV-caching, and Gumbel/argmax sampling and is about to port confidence-based parallel
> decoding to OmniVoice (a 32-step iterative-unmasking TTS model).
>
> Sources:
> - Project page: https://nvlabs.github.io/Fast-dLLM/
> - Arxiv (paper): https://arxiv.org/abs/2505.22618 (HTML: https://arxiv.org/html/2505.22618v1)
> - Repo: https://github.com/NVlabs/Fast-dLLM
> - Canonical reference impl: `v1/llada/generate.py`
> - Title: *Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache
>   and Parallel Decoding*
> - Authors: Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu,
>   Ping Luo, Song Han, Enze Xie (HKU + NVIDIA + MIT)

---

## 1. Problem Fast-dLLM solves

### 1.1 What's slow about diffusion / iterative-unmasking generators

Masked Diffusion Models (MDMs) for text — LLaDA-8B, Dream-7B/Base — are bidirectional
non-autoregressive token-level denoisers. They start with a fully masked answer
`[p_0; [MASK], ..., [MASK]]` and run T denoising steps that progressively replace
`[MASK]` with sampled tokens until the answer region is fully unmasked. In principle
this is parallel; in practice the open-source releases have two compounding
inefficiencies:

1. **No KV cache**. Because attention is bidirectional and every step re-attends over
   every position, a vanilla MDM forward pass is `O(L^2)` per step and cannot reuse
   prior steps' K/V the way a causal LM can. With T denoising steps, this turns into
   a `T x O(L^2)` cost — at gen_length=512 the baseline LLaDA throughput is
   2–8 tok/s on an A100-80GB, vs. autoregressive LLMs which deliver 50–150 tok/s.

2. **Quality collapses if you unmask too many tokens at once.** Naively unmasking K
   tokens per step from the conditional-independence factorisation
   `prod_i p(x_i | x_unmasked)` violates true joint dependencies. Paper's running
   example: prompt "the list of poker hands that consist of two English words are:
   ___ ___". The marginals are bimodal over {high, two, full, straight} for slot 1 and
   {card, pair, house, flush} for slot 2; sampling them independently produces "high
   pair" or "straight house" with high probability — token-pairs that are individually
   plausible but jointly nonsensical.

So the baseline trade-off is brutal: T = gen_length steps gives quality but is
crippling-slow; T << gen_length is fast but garbage.

### 1.2 Baseline they beat

Plain LLaDA-Instruct / Dream-Base inference, single batch, A100-80GB, no inference
framework. Same backbone weights, no fine-tune. Fast-dLLM is **training-free** — it's a
pure decoding-loop and KV-management change.

### 1.3 Headline numbers (LLaDA-Instruct, A100-80GB, batch=1)

| Bench (gen_len) | Baseline acc / tok/s | Fast-dLLM acc / tok/s | Speedup |
|---|---|---|---|
| GSM8K-5shot (256) | 79.3% / 6.7  | 78.5% / 54.4 | 8.1x |
| GSM8K-5shot (512) | 77.5% / 3.2  | 77.2% / 35.3 | 11.0x |
| MATH-4shot (512)  | 37.2% / 8.0  | 36.0% / 47.1 | 5.9x  |
| HumanEval (512)   | 43.9% / 18.4 | 44.5% / 73.7 | 4.0x  |
| MBPP (512)        | 14.8% / 4.3  | 13.8% / 39.5 | 9.2x  |

Peak number: **27.6x** on LLaDA GSM8K-8shot, gen_len=1024, DualCache + parallel
decoding (Table 3, see §5).

---

## 2. Two core ideas

The whole paper is two orthogonal speedups that compose multiplicatively.

### 2.1 Block-wise approximate KV cache

Idea in one sentence: split the answer region into K contiguous blocks of size B; within
a block, freeze the K/V activations of everything outside the block (prefix and suffix)
and only recompute attention for the B current block tokens; refresh the full K/V cache
once per block.

Why this is non-trivial for an MDM: attention is **full / bidirectional**, so a token
in block 2 mathematically attends both backward (block 1, prompt) and forward (block 3,
still all `[MASK]`). Caching the prefix is fine for the prefix's own K/V, but the key
question is whether token i in block 2 sees the same K/V from block 1 across consecutive
denoising steps. The paper's empirical claim (Section 3.2, Figure 3 heatmaps): for
adjacent inference steps within a block, prefix K/V have very high cosine similarity
along the diagonal. So freezing them introduces only an approximation error, no
correctness violation.

Formally there is **no closed-form bound on cache-approximation error**. The paper
relies on the empirical similarity observation rather than a Lipschitz-type bound. The
approximation is then refreshed every block, which keeps cumulative drift small.

**DualCache** is the same idea applied to the *suffix* (the still-fully-masked tokens
after the current block). Since those positions also produce K/V each step, and since
their K/V change slowly across inference steps (they're all `[MASK]` queries against the
same context), they too can be cached and reused within a block. DualCache caches *both*
prefix and suffix; the single-cache variant ("PrefixCache") caches only the prefix.

Standalone speedup from cache alone: **2.0x – 3.6x** depending on prefill length
(longer prefill = bigger win because the prefix-recompute cost is the dominant term).

### 2.2 Confidence-aware parallel decoding

Idea in one sentence: at each step, rather than committing a *fixed* number of tokens
(say `block_size / steps_per_block`), commit *every* masked token whose top-1 softmax
probability exceeds a threshold τ — and always commit at least one (the argmax-confidence
token) to guarantee progress.

The intuition is that the conditional-independence error (the "high pair" problem) is
small *exactly when* each marginal is concentrated on one value. If
`p(x_i = v_i | context) > 1 - ε` for each i, then the product distribution over a tuple
is dominated by the single tuple `(v_1, ..., v_n)` and matches the joint argmax. The
formal version is Theorem 1 (§3.3, see §4 below).

Standalone speedup from parallel decoding alone: **4.0x – 6.0x** on LLaDA, **1.6x – 4.0x**
on Dream.

The two ideas compose: prefix/suffix caching cuts per-step cost; confidence-aware
parallel decoding cuts the *number of steps* needed to fully unmask a block. Combined:
up to 27.6x.

---

## 3. Confidence-aware parallel decoding — full algorithm

### 3.1 Definitions

- `x ∈ {V ∪ [MASK]}^L` — full sequence (prompt + answer region), L = |prompt| + gen_length.
- `p_θ(· | x) ∈ Δ^V` — model's softmax distribution at each position given the current x.
- `block_size B`, `num_blocks K = gen_length / B`, `steps_per_block T`.
- `τ ∈ (0, 1]` — confidence threshold. Default **τ = 0.9**.
- For a masked position i, **confidence is defined as the top-1 softmax probability**:
  `c_i = max_{v ∈ V} p_θ(x_i = v | x)`.
  This is **not** entropy, **not** margin to second-best, **not** logit gap. Just
  `softmax(logits).max(dim=-1)`. (Code: `p = softmax(logits.float64); x0_p = gather(p, x0)`.)

### 3.2 Algorithm 1 (paper, lightly cleaned)

```
Inputs: model p_θ, prompt p_0, gen_length L, num_blocks K, block_size B,
        steps_per_block T, threshold τ, use_DualCache ∈ {True, False}

1.  x ← [p_0; MASK, MASK, ..., MASK]                  # length |p_0| + L
2.  Initialise KV cache for x (jointly with first decode pass)
3.  for k = 1 ... K do
4.     s ← |p_0| + (k-1) * B
5.     e ← |p_0| + k * B
6.     for t = 1 ... T do
7.        Run p_θ on x[s:e]   (with DualCache)
                or  x[s:]     (with PrefixCache only)
                using the cached prefix (and suffix if DualCache)
8.        For every masked i in [s, e):
              x0_i = argmax_v p_θ(x_i = v | x)              # proposal
              c_i  = max_v   p_θ(x_i = v | x)               # confidence
9.        Unmask all i ∈ [s, e) with c_i ≥ τ; ALWAYS unmask argmax_i c_i
10.       if all positions in [s, e) are unmasked: break
11.    end for
12.    Refresh KV cache (prefix and, if DualCache, suffix) using updated x
13. end for
14. return x
```

### 3.3 Step-by-step semantics

- **Line 7 — model call.** With DualCache, the query passed to the model is only the B
  tokens of the current block. With PrefixCache, it's the block + the still-masked
  suffix (because the suffix isn't cached). The cache supplies frozen K/V for the
  *complementary* positions.
- **Line 8 — proposal + confidence.** One forward pass produces logits for every
  position. We compute `x0 = argmax(logits + Gumbel(temperature))`, but **the confidence
  used for the threshold is computed from the noiseless softmax**, not the gumbel-noised
  logits. (See `get_transfer_index`: gumbel goes into x0; softmax for x0_p is on bare
  logits.)
- **Line 9 — threshold rule.** Two clauses combined by OR:
  - `transfer_index = mask_index & (confidence >= τ)` — every masked position whose
    top-1 prob ≥ τ.
  - `force_mask = one-hot(argmax(confidence, dim=1))` — guarantee at least one token
    even if no position exceeds τ.
  - Final: `transfer_index = (transfer_index | force_mask) & mask_index`.
  The `& mask_index` re-anding is a paranoia clause: don't unmask anything already
  decoded. (Important when `force_mask` could in principle land on a non-masked index
  if confidence on a previously-decoded position was somehow encoded — in practice
  confidence at non-masked positions is set to `-inf`, but the safety AND is cheap.)
- **Line 10 — early exit.** If the block is fully unmasked before T steps elapse, break.
  This is what makes the confidence path *fewer steps on average* than the fixed-quota
  path: you don't pay for steps you don't need.
- **Line 12 — cache refresh.** After a block is complete, run a full forward pass once
  to recompute K/V for the whole sequence (now with this block decoded). This is the
  "expensive" pass that amortises across the next block's T steps.

### 3.4 What "confidence" doesn't mean

- **Not entropy.** The paper specifically uses top-1 softmax probability. Entropy is
  cheap and would correlate, but the algorithm is as written.
- **Not normalized by alternatives.** No log-margin to second-best.
- **Computed from the un-noised softmax**, not from the Gumbel-noised proposal scores.
  In code: `p = F.softmax(logits.to(float64), dim=-1); x0_p = gather(p, x0)`.
- **Computed in float64** for numerical stability, especially near 1 - ε regions where
  bf16 rounds to 1.0.

### 3.5 What happens to tokens that don't meet τ

They stay masked. Next iteration they get re-evaluated with a freshly-context (because
some of their neighbours are now decoded), and their confidence typically jumps. This is
the whole reason the technique works: low-confidence positions are *exactly* the ones
where dependencies on undecoded neighbours dominate, and decoding the easy ones first
makes the hard ones easy.

### 3.6 Block size's role in this algorithm

The threshold rule operates **only over masked positions in the current block** (line 9
restricts to `i ∈ [s, e)`). So:

- Small B: each step's threshold operates on at most B positions, you finish the block
  in fewer iterations relative to B (often 1–2 steps for B=4), but you pay the cache
  refresh K = L/B times.
- Large B: more positions to threshold simultaneously, more steps needed per block, but
  fewer cache refreshes.
- Paper sweep (Figure 4): **B = 32 is the sweet spot** for LLaDA at gen_len 256–1024.

### 3.7 Alternative variant: dynamic per-rank thresholds (`get_transfer_index_dynamic`)

The repo also implements a `factor`-based variant (paper hints at this in the ablations
but doesn't make it canonical). For a row with n masked positions, it derives a
per-rank threshold sequence:

```
n_j ∈ {1, 2, ..., n}
ε_j = factor / (n_j + 1)
τ_j = 1 - ε_j
```

Sort the row's confidences descending; find the largest top-k such that the k-th
sorted confidence is ≥ τ_k. This is a direct realisation of the Theorem 1 bound
ε ≤ 1/(n+1) per number-of-tokens-decoded. With `factor=1` it's the bound itself;
with `factor<1` it's stricter (more conservative, fewer parallel decodes); `factor>1`
is more aggressive.

---

## 4. KV-cache approximation, in detail

### 4.1 Why naive KV cache fails for bidirectional MDMs

In a causal LM, position i's K/V never change once computed: K_i and V_i depend only
on positions ≤ i, all of which are frozen once decoded. In an MDM, position i attends
bidirectionally and is itself a function of positions both before and after i. Each
denoising step that decodes a token *somewhere* changes the K/V of *every other*
position, in principle.

Empirically, the paper's Figure 3 shows the cosine similarity matrix between K (or V)
activations at consecutive steps t and t+1 is dominated by the diagonal — i.e. K_i^{(t)}
≈ K_i^{(t+1)} for most i, especially i in the prompt prefix. So freezing prefix K/V
across short windows of steps is a small approximation, not a correctness violation.

### 4.2 PrefixCache (single cache)

Cache discipline:

```
cache_init at start of block k:
    run full forward on entire x with use_cache=True
    past_key_values = output.past_key_values      # K/V for the whole sequence
    keep only positions [0, s)  →  prefix_kv      # truncate suffix
within steps t = 1..T of block k:
    forward(x[s:], past_key_values=prefix_kv, use_cache=True)
    # model recomputes K/V for positions [s, L); attends to cached [0, s) + new
end-of-block:
    drop prefix_kv, repeat cache_init for next block
```

Per step cost: full attention over [s, L), but K/V for [0, s) are looked up from cache.
If gen_length and prompt are both 512, prefix is 512 + (k-1)*B, so by block k=K the
prefix dominates. This is why the paper's prefill-length ablation (Table 3) shows the
biggest cache wins on long prompts (8-shot prompt, ~2x more prefill than 5-shot) —
13.3x for cache-only with 8-shot vs 10.6x with 5-shot.

### 4.3 DualCache

DualCache also caches the suffix (still-masked tokens after the block's end). Per step:

```
cache_init at start of block k:
    run full forward on entire x; keep past_kv for [0, s) ∪ [e, L)
within steps t = 1..T:
    forward(x[s:e], past_key_values=dual_kv, use_cache=True,
            replace_position=mask_with_True_in_[s,e))
    # model only computes K/V for the B positions in the block
```

The `replace_position` boolean mask tells the model where in the cached layout the new
queries go (so the static graph stays static even as the block index s..e advances).
This is the key to making DualCache CUDAGraph-friendly: every block-level forward has
the same (B,) query length and the same KV-cache shape; only `replace_position` shifts.

Per step cost: attention over [s, e), reading cached K/V for everything else. Cost
becomes nearly independent of L for fixed B, which is why DualCache scales much better
to long generations: 27.6x at gen_len=1024 vs 8.1x at gen_len=256.

### 4.4 Approximation error bound

The paper does **not** provide a formal bound on cache-approximation error. It provides:

- Empirical similarity heatmaps (Figure 3).
- End-to-end accuracy parity within 1–2 points of baseline (Tables 1, 2).
- A theorem on parallel-decoding error (Theorem 1) — separate concern, see §4.6.

If you want a worst-case bound for OmniVoice, you'll have to measure cosine-sim of K/V
across denoising steps yourself (heatmap as in Fig 3) and verify it's diagonally
dominant for the audio-token-mask regime.

### 4.5 Refresh frequency

Cache is refreshed **once per block**, never within a block. In LLaDA terms with default
B=32 and gen_length=128, that's K=4 refreshes plus the initial init, vs T*K=128 forward
passes in the no-cache baseline.

### 4.6 Theorem 1 — confidence ⇒ parallel == sequential

Statement (paraphrased from §3.3): suppose at step t, the marginal product distribution
`q(z|E) = prod_i p(z_i|E)` and the true joint distribution `p(z|E)` agree on a single
mode `x*` such that `p_i(z_i = x*_i | E) > 1 - ε` for all n positions being decoded
in parallel. Then if `(n+1) ε ≤ 1`, i.e. **`ε ≤ 1/(n+1)`**:

> argmax_z p(z | E) = argmax_z q(z | E) = x*

Proof sketch: x* is the unique product-of-marginals argmax (each factor is > 1 - ε).
By a union bound, every other tuple has joint mass ≤ n * ε. Since the n+1 sum
inequality forces `1 - ε > n * ε`, x* dominates. The bound is tight (counterexample
constructed for ε > 1/(n+1)). The TV-distance bound between joint and product is
`D_TV(p, q) < (3n - 1) / 2 · ε`.

Practical reading: **threshold τ = 1 - ε**. If you set τ = 0.9 and the algorithm decides
to unmask n positions in one step, the theorem only guarantees correctness when
n ≤ 1/ε - 1 = 9. So τ=0.9 is reasonable for n ≲ 10 simultaneous tokens. The paper's
default block_size 32 with τ=0.9 routinely violates this assumption *strictly* but
performs well empirically because most blocks decode in 4–8 batches of 4–8 tokens, not
one batch of 32.

---

## 5. Reported speedups and quality numbers

Hardware: NVIDIA A100-80GB. Batch size = 1. No inference framework (vanilla HF
`transformers`). Backbones: LLaDA-8B-Instruct and Dream-Base-7B.

### 5.1 LLaDA Table 1 (acc % / tok/s / speedup vs baseline)

Block size 32, threshold 0.9 (Fast-dLLM defaults).

| Bench | Gen | Baseline | +Cache only | +Parallel only | +Both |
|---|---|---|---|---|---|
| GSM8K (5-shot) | 256 | 79.3 / 6.7 (1×) | 79.5 / 21.2 (3.2×) | 79.2 / 16.5 (2.5×) | **78.5 / 54.4 (8.1×)** |
| GSM8K (5-shot) | 512 | 77.5 / 3.2 (1×) | 77.0 / 10.4 (3.3×) | 77.6 / 18.6 (5.8×) | **77.2 / 35.3 (11.0×)** |
| MATH (4-shot)  | 256 | 33.5 / 9.1 (1×) | 33.3 / 23.7 (2.6×) | 33.4 / 24.8 (2.7×) | **33.2 / 51.7 (5.7×)** |
| MATH (4-shot)  | 512 | 37.2 / 8.0 (1×) | 36.2 / 19.7 (2.5×) | 36.8 / 23.8 (3.0×) | **36.0 / 47.1 (5.9×)** |
| HumanEval      | 256 | 41.5 / 30.5 (1×) | 42.7 / 40.7 (1.3×) | 43.9 / 101.5 (3.3×) | **43.3 / 114.1 (3.7×)** |
| HumanEval      | 512 | 43.9 / 18.4 (1×) | 45.7 / 29.3 (1.6×) | 43.3 / 57.1 (3.1×) | **44.5 / 73.7 (4.0×)** |
| MBPP (3-shot)  | 256 | 29.4 / 6.0 (1×) | 29.6 / 17.0 (2.8×) | 28.4 / 24.8 (4.1×) | **28.2 / 44.8 (7.5×)** |
| MBPP (3-shot)  | 512 | 14.8 / 4.3 (1×) | 13.4 / 10.1 (2.3×) | 15.0 / 22.3 (5.1×) | **13.8 / 39.5 (9.2×)** |

### 5.2 Dream Table 2 (same legend)

| Bench | Gen | Baseline | +Cache | +Parallel | +Both |
|---|---|---|---|---|---|
| GSM8K (5) | 256 | 75.0 / 9.1 | 74.3 / 32.5 (3.6×) | 74.2 / 14.2 (1.6×) | **74.8 / 48.2 (5.3×)** |
| GSM8K (5) | 512 | 76.0 / 7.7 | 74.3 / 25.6 (3.3×) | 73.4 / 14.6 (1.9×) | **74.0 / 42.9 (5.6×)** |
| MATH (4)  | 256 | 38.4 / 11.4 | 36.8 / 34.3 (3.0×) | 37.9 / 27.3 (2.4×) | **37.6 / 66.8 (5.9×)** |
| MATH (4)  | 512 | 39.8 / 9.6  | 38.0 / 26.8 (2.8×) | 39.5 / 31.6 (3.2×) | **39.3 / 63.3 (6.5×)** |
| HumanEval | 256 | 49.4 / 23.3 | 53.7 / 35.2 (1.5×) | 49.4 / 45.6 (2.0×) | **54.3 / 62.0 (2.8×)** |
| HumanEval | 512 | 54.3 / 16.3 | 54.9 / 27.8 (1.7×) | 51.8 / 29.8 (1.8×) | **54.3 / 52.8 (3.2×)** |
| MBPP      | 256 | 56.6 / 11.2 | 53.2 / 34.5 (3.1×) | 53.8 / 31.8 (2.8×) | **56.4 / 76.0 (6.8×)** |
| MBPP      | 512 | 55.6 / 9.4  | 53.8 / 26.7 (2.8×) | 55.4 / 37.6 (4.0×) | **55.2 / 73.6 (7.8×)** |

Observation: parallel decoding alone is much weaker on Dream than on LLaDA (1.6x vs
5.8x on GSM8K-512), but cache alone is comparable. Combined speedups are also lower
on Dream. Likely cause: Dream uses a cosine schedule that decodes more tokens per step
already; LLaDA has more "wasted" low-confidence steps to skip.

### 5.3 Table 3 — prefill length effect (LLaDA, gen_len 1024, A100-80GB)

| Setup | Baseline | +PrefixCache | +DualCache | Both + DualCache |
|---|---|---|---|---|
| 5-shot prompts | 77.0 / 1.1 | 77.4 / 11.7 (10.6×) | 75.2 / 14.4 (13.1×) | 74.7 / 21.6 (19.6×) |
| 8-shot prompts | 77.3 / 0.7 | 78.0 / 9.3 (13.3×) | 75.7 / 13.0 (18.6×) | **76.0 / 19.3 (27.6×)** |

The 27.6x headline is here: longest prompt + longest gen + DualCache + parallel.

### 5.4 Table 4 — generation length effect (LLaDA, 8-shot)

| Gen len | Baseline | +PrefixCache | +DualCache | Both + DualCache |
|---|---|---|---|---|
| 256  | 77.6 / 4.9 | 77.9 / 16.4 (3.3×) | 77.3 / 49.2 (10.0×) | 76.9 / 46.3 (9.4×) |
| 512  | 78.9 / 2.3 | 78.9 / 14.0 (6.1×) | 74.8 / 32.0 (13.9×) | 75.4 / 36.4 (15.8×) |
| 1024 | 77.3 / 0.7 | 78.0 / 9.3 (13.3×) | 75.7 / 13.0 (18.6×) | 76.0 / 19.3 (27.6×) |

Longer generations get bigger speedups, because per-block O(B^2) cost is bounded while
no-cache cost grows quadratically.

### 5.5 Block size sweep (Figure 4)

Paper text: "Block size of 32 achieves the best trade-off." Curves (paraphrased):
B=4: maximum speedup but accuracy starts dropping ~3 points.
B=8, 16: middle ground.
B=32: best speedup-vs-accuracy.
B=64: speedup levels off, accuracy fine.
B=128: diminishing returns.

### 5.6 Threshold sweep (Figure 5)

Range explored: τ ∈ [0.5, 1.0]. Paper does not publish per-τ accuracy in tabular form;
text says **τ=0.9 is the chosen default**. Trends from Figure 5:

- τ=0.5: high speedup, accuracy collapse (model commits low-confidence tokens, joint
  errors compound).
- τ=0.7: still some accuracy degradation on harder benches (MATH).
- τ=0.85–0.95: sweet spot — speedup retained, accuracy within 1–2 points of baseline.
- τ=0.99: very conservative; speedup degrades toward parallel-disabled baseline.

### 5.7 Where quality breaks

- Threshold τ < 0.7 — joint-dependency violations dominate.
- Block size > 64 — cache approximation error grows; longer windows of frozen K/V
  drift further from true K/V.
- Long generation with PrefixCache only (no DualCache) — accuracy stays good but
  speedup plateaus because suffix recompute dominates.

---

## 6. Hyperparameters that matter for a port

Listed in order of "if you mistune this you'll feel it most".

| Knob | Default | Range tested | Sensitivity |
|---|---|---|---|
| `threshold τ` | **0.9** | [0.5, 1.0] | **HIGH**. Below 0.7 → quality collapse. Above 0.95 → speedup vanishes. |
| `block_size B` | **32** | {4, 8, 16, 32, 64} | **HIGH**. Both axes affected. 32 is the sweet spot for LLaDA. |
| `steps_per_block T` | **gen_length / num_blocks**, e.g. 32 if steps=128 and num_blocks=4 | follows `steps` | MEDIUM. T = B is the natural choice; algorithm exits early via line 10 if a block decodes faster. |
| `total steps` | gen_length (one step per token, paper default; effectively unused with τ-mode because of early exit) | — | LOW once τ-mode is on; serves as upper-bound budget. |
| `temperature` | 0.0 (greedy) for benchmarks | {0, 0.7+} | MEDIUM. Affects x0 sampling (Gumbel) but not confidence (which uses raw softmax). |
| `remasking` | `'low_confidence'` | {`low_confidence`, `random`} | HIGH. `random` is the ablation baseline; always use `low_confidence` for production. |
| `use_DualCache` | True | bool | MEDIUM. DualCache > PrefixCache on long generations; equal on short. |
| `mask_id` | 126336 (LLaDA's vocab) | — | model-specific, will differ for OmniVoice |
| `factor` (dynamic mode) | None (uses fixed τ); or 1.0 if dynamic | (0, ∞) | MEDIUM. Direct realisation of Theorem 1 bound; rarely used. |
| `gen_length` | benchmark-dependent (256, 512, 1024) | — | system-level |

OmniVoice-specific knobs to add:

- Number of denoising steps: **32** (vs LLaDA's 128).
- Audio codebook size and number of streams (RVQ-style? VQ?). Confidence is per-stream
  if multi-codebook.
- Whether OmniVoice has a "prefill" prompt at all, or is the conditioning handled via
  cross-attention.

---

## 7. Caveats / where Fast-dLLM doesn't work

### 7.1 Model-class assumptions

- **Masked-diffusion specifically.** The technique assumes the model is trained with a
  masked-token reconstruction objective producing a softmax distribution over the vocab
  per position, with [MASK] as a distinct token. Continuous-noise diffusion (e.g., score
  matching on token embeddings, vector-field denoising) does not produce per-token
  confidence in the required form; you'd need to redefine confidence (e.g., norm of
  predicted noise, distance to nearest codebook entry).
- **Bidirectional / full-attention transformer.** Encoder-style. Causal-only models
  don't need this — they have native KV caching.
- **One-step prediction.** Model must output `p(x_i | rest)` for masked positions in one
  forward pass. If the iterative-unmasking model uses a multi-step inner loop per
  denoising step, the cache structure differs.

### 7.2 Failure modes in ablations

- **Threshold too low** (τ ≤ 0.5) — accuracy drops 5–10 points (paper Figure 5 trend).
- **Block size too large** (B ≥ 128) — cache approximation breaks; the further
  prefix K/V are reused, the more they diverge from the true K/V at the current step.
- **Random remasking** instead of `low_confidence` — kills the parallel-decoding gain
  entirely, since random selection no longer correlates with low joint-error positions.
- **Very short prompts + long gen with PrefixCache** — cache savings tiny; you'd want
  DualCache.
- **Codebook collapse / very peaked priors** — if the model trivially commits all
  tokens above τ in one shot, you skip the iterative refinement entirely; not necessarily
  bad, but you should validate the resulting joint sample isn't degenerate (e.g., long
  silences in TTS).

### 7.3 Quality-vs-speed not strictly Pareto

In several rows of Table 1 (HumanEval especially), accuracy is *higher* under
Fast-dLLM than baseline. Don't read too much into this — it's within noise of single-run
greedy decoding. Treat parity within ±1.5 points as the success criterion.

### 7.4 No formal cache-error bound

The paper's only theory is Theorem 1 (parallel-decoding error). The cache approximation
is justified empirically. So if your modality (audio tokens) has very different K/V
similarity behaviour across steps, you'll need to remeasure.

---

## 8. Implementation pointers

### 8.1 Canonical implementation

Repo: https://github.com/NVlabs/Fast-dLLM
Key file: `v1/llada/generate.py`
- `generate(...)` — vanilla, no cache. Reference for correctness.
- `generate_with_prefix_cache(...)` — single (prefix-only) cache, naive.
- `generate_with_dual_cache(...)` — production path. CUDAGraph-friendly: uses
  `replace_position` and `torch.where` instead of dynamic slice-assignment.
- `get_transfer_index(...)` — the threshold-vs-topk transfer logic (the hot path).
- `get_transfer_index_dynamic(...)` — Theorem-1-derived per-rank thresholds (`factor`).
- `get_num_transfer_tokens(...)` — fallback fixed-quota schedule.
- `add_gumbel_noise(...)` — float64 Gumbel-max for stability.

Adjacent files (look at these next when porting):
- `v1/llada/chat.py` — interactive harness, shows recommended CLI args.
- `v1/llada/eval.py` — benchmarking driver (also `accelerate launch` setup).
- `v1/dream/eval.py` — Dream variant; useful to compare against a *different*
  MDM that already had its own KV scaffolding.
- `v1/model/modeling_llada.py` — the modified LLaDA model that supports `use_cache=True`,
  `past_key_values=...`, and `replace_position=...`. **This is the most important
  file for the port** because it shows what changes to a bidirectional model are
  needed for cached forward.
- `v2/` — block-diffusion training-aware variant; uses LMFlow + DeepSpeed.
- `fast_dvlm/` — vision-language extension.

### 8.2 Hot-path snippets (paraphrased from `generate.py`)

The transfer-index logic in PyTorch terms:

```python
@torch.no_grad()
def get_transfer_index(logits, temperature, remasking, mask_index, x,
                       num_transfer_tokens, threshold=None):
    # logits: (B, L, V)   mask_index: (B, L) bool   x: (B, L) long
    logits_with_noise = add_gumbel_noise(logits, temperature)
    x0 = logits_with_noise.argmax(dim=-1)            # (B, L)

    # Confidence = softmax max-prob (NOT entropy, NOT margin), float64.
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)   # (B, L)
    else:  # 'random'
        x0_p = torch.rand_like(x0, dtype=torch.float64)

    # Only valid at masked positions.
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p,
                             torch.tensor(float('-inf'),
                                          device=x.device,
                                          dtype=torch.float64))

    if threshold is not None:
        transfer = mask_index & (confidence >= threshold)
        # Force at least one (highest-confidence) per row.
        argmax_idx = confidence.argmax(dim=1, keepdim=True)
        force = torch.zeros_like(transfer).scatter_(1, argmax_idx, True)
        transfer = (transfer | force) & mask_index
        return x0, transfer

    # Else fixed-quota top-k by confidence per row.
    _, idx = confidence.sort(dim=1, descending=True)
    cols = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    select_sorted = cols < num_transfer_tokens.unsqueeze(1)
    transfer = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    transfer.scatter_(1, idx, select_sorted)
    transfer = transfer & mask_index
    return x0, transfer
```

The `generate_with_dual_cache` outer loop in pseudocode form:

```python
@torch.no_grad()
def generate_with_dual_cache(model, prompt, *, gen_length, block_size,
                             steps, threshold, temperature, mask_id):
    B, Lp = prompt.shape
    num_blocks = gen_length // block_size
    steps_per_block = steps // num_blocks
    L = Lp + gen_length

    x = torch.full((B, L), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :Lp] = prompt

    for nb in range(num_blocks):
        s = Lp + nb * block_size
        e = s + block_size

        # 1. Warm cache on full sequence — this is the only "expensive" pass per block.
        out = model(x, use_cache=True)                       # forward over L
        past_kv = out.past_key_values

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        # Step 0 uses the full-sequence logits we already computed.
        block_mask = (x == mask_id); block_mask[:, e:] = False
        x0, transfer = get_transfer_index(
            out.logits, temperature, 'low_confidence',
            block_mask, x, None, threshold,
        )
        x = torch.where(transfer, x0, x)

        # Steps 1..T: forward only the B-token block, attending via cache.
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e],
                past_key_values=past_kv,
                use_cache=True,
                replace_position=replace_position,
            ).logits                                           # (B, B, V)
            mask_blk = (x[:, s:e] == mask_id)
            x0_blk, transfer_blk = get_transfer_index(
                logits_blk, temperature, 'low_confidence',
                mask_blk, x[:, s:e], None, threshold,
            )
            new_blk = torch.where(transfer_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], new_blk, x[:, e:]], dim=1)
    return x
```

Critical engineering notes for the port:

- **Static-shape forwards.** The block-level forward always has query length B; the
  KV-cache shape is fixed; `replace_position` is the only thing that changes per block.
  This is what makes the inner loop CUDAGraph-capturable. **Do not** use Python-level
  slicing of `past_key_values` inside the inner loop — it will break graph capture.
- **`torch.where` not masked assignment.** `x[mask] = x0[mask]` triggers dynamic shape;
  `torch.where(mask, x0, x)` is shape-stable.
- **Float64 softmax.** Use `logits.to(torch.float64)` before softmax for the confidence
  computation. bf16 saturates to 1.0 above ~0.996, which would make τ=0.9 always trigger
  near the end of decoding.
- **Gumbel goes into x0, not into confidence.** Two different code paths:
  `argmax` of (logits + Gumbel) for the proposal token; `softmax` of bare logits
  for the threshold. Don't conflate.
- **Force-unmask the argmax** even when no token meets τ. Without this, you can
  livelock on degenerate prompts.
- **One full-sequence forward per block** (the "warm" pass). This is unavoidable
  because the block transition needs current K/V everywhere.

### 8.3 Where the model file (`modeling_llada.py`) needs corresponding changes

(Inferred from `generate_with_dual_cache` call signature; verify in the actual repo.)

The transformer needs to support, in its forward:

- `use_cache: bool` — return `past_key_values`.
- `past_key_values: Optional[List[Tuple[K, V]]]` — when given, *don't* recompute K/V for
  positions already in cache; instead concatenate the new K/V to the cached ones for
  the attention computation.
- `replace_position: Optional[BoolTensor]` of shape (B, L_total) — when given, the
  "new" K/V from the small forward go into specific positions in the cached tensor
  (not appended at the end as in causal models). This is the bidirectional twist:
  position writes are *positional*, not append-only.

For the OmniVoice port, these three knobs need to be threaded through whatever
attention layer OmniVoice uses (likely a flash-attention or xformers kernel — verify
the kernel supports a `replace_position`-style scatter into K/V).

### 8.4 Quick-start command-line defaults from the README

```
python llada/chat.py --gen_length 128 --steps 128 --block_size 32 \
                     --threshold 0.9 --remasking low_confidence
```

For Dream, the README cites `max_new_tokens=256, diffusion_steps=8` (note the much
smaller step count — Dream-Base does fewer denoising steps natively).

---

## 9. Open questions for the OmniVoice port

OmniVoice is a 32-step iterative-unmasking TTS model. Mapping Fast-dLLM onto it raises
the following concrete questions / risks:

1. **Token semantics.** OmniVoice operates over audio tokens (presumably an
   RVQ-codec like Encodec/SoundStream/DAC). Are these single-stream or
   multi-stream (e.g., 8 codebooks per timestep)?
   - If multi-stream, "confidence" needs to be defined: top-1 prob *per codebook*
     and only commit a (timestep, codebook) cell when *all* its codebooks pass
     τ? Or pick the lowest of the n top-1 probs?
   - The conditional-independence violation may be more severe across codebooks
     of the same timestep than across timesteps.

2. **What "block" means in time.** For text, a block is a contiguous span of
   token positions. For audio, a block could be:
   - A contiguous span of *audio frames* (most natural — preserves temporal locality).
   - A contiguous span of *codebook layers* across all frames (more like RVQ-stage
     decoding, similar to Vall-E's NAR stages).
   The right choice probably maps frames→positions and treats the multi-codebook
   per-frame factorization separately.

3. **32 denoising steps total — is parallel decoding even helpful?** If the
   baseline model already commits ~all tokens in 32 steps for a long sequence,
   the parallel-decoding speedup ratio (~3–6x in text) might be smaller. The
   cache speedup (2–3.6x) is more straightforward; you may get most of the win
   from cache alone.

4. **Cache validity across denoising steps for audio.** The paper relies on
   empirical K/V similarity (Figure 3). Does this hold for OmniVoice? Easy
   verification experiment: dump K/V tensors from each layer at each step, plot
   per-position cosine similarity step-over-step. If diagonal-dominant, cache
   reuse is safe; if not, block_size needs to shrink.

5. **Threshold τ on audio tokens.** Audio token vocabularies are smaller (often
   1024 per codebook) than text (~150k). Top-1 softmax probabilities are
   intrinsically higher on smaller vocabs (less mass to spread). τ=0.9 on a
   1024-vocab might be far less conservative than τ=0.9 on a 150k-vocab. May
   need to recalibrate — try τ=0.7 / 0.5 first and compare with text intuition.

6. **Does OmniVoice use cross-attention to text?** If yes, the cross-attn K/V
   are computed from the text encoder *once* and never change across denoising
   steps. That's a free permanent cache, and it's likely already implemented.
   Fast-dLLM's contribution then narrows to caching the *self-attention* K/V
   in the audio decoder.

7. **CUDAGraph compatibility with TTS-specific kernels.** The repo uses static
   `replace_position` boolean masks to keep graphs stable. If OmniVoice uses
   custom CUDA kernels (e.g., diff-attn, sliding-window), the new `replace_position`
   path needs to be implemented in those kernels too.

8. **Quality eval.** No GSM8K analogue for TTS. Need objective WER (via Whisper
   transcription), MOS or UTMOS for naturalness, and possibly speaker-similarity
   if voice cloning. Paper's "within 1–2 points" criterion needs translating to
   "within 0.05 UTMOS" or "within 0.5% WER".

9. **Streaming vs offline.** OmniVoice may expose a streaming interface; block-wise
   decoding is naturally compatible (block 1 starts emitting before block K is
   computed). Make sure block boundaries align with whatever frame-rate the
   downstream vocoder/decoder expects.

10. **The "always unmask argmax" guarantee.** Need to make sure that on a
    multi-codebook system, this rule still produces progress (e.g., always unmask
    the highest-confidence (frame, codebook) cell, even if the same frame already
    has some codebooks unmasked).

11. **Schedule alignment.** LLaDA's `get_num_transfer_tokens` distributes masked
    tokens evenly across T steps when not using τ. OmniVoice may have its own
    cosine/linear schedule for how many tokens to commit per step. Decide whether
    Fast-dLLM's confidence rule replaces that schedule entirely, or only kicks in
    when the schedule's per-step quota would have committed something the threshold
    rejects.

12. **Effect on the diffusion noise schedule.** LLaDA uses a linear noise schedule
    (Eq. 8 in the original LLaDA paper, referenced in `get_num_transfer_tokens`'s
    docstring). If OmniVoice's noise schedule is non-linear (cosine, sigmoid),
    the equal-distribution fallback path may need adjusting before τ-mode kicks in.

---

## 10. Quick reference card

Things to remember when reading the code or porting:

- **τ default = 0.9**, applied to top-1 softmax probability.
- **B default = 32**, **gen_length must be divisible by B**.
- `steps` is divided across blocks: `steps_per_block = steps // num_blocks`. With
  τ-mode and early exit, this is mostly an upper bound.
- **Always force-unmask argmax confidence** to guarantee progress.
- Confidence at non-masked positions is set to **-inf** (so they never get re-touched).
- Use `low_confidence` remasking; `random` is the ablation baseline.
- Use **DualCache** for long generations; **PrefixCache** is fine for short.
- **Cache refresh is per-block**, not per-step.
- **Float64 softmax** for confidence; **Gumbel-noised argmax** for proposals.
- The KV-cache *approximation* is empirical, not provably bounded — verify on your
  modality before trusting B=32.
- Theorem 1 says parallel decoding ≡ greedy joint when ε ≤ 1/(n+1) — i.e. τ ≥ 1 - 1/(n+1).
  For τ=0.9 this strictly holds for n ≤ 9; the paper exceeds this and works empirically.
- Ports of `modeling_llada.py` into your model need three new forward args:
  `use_cache`, `past_key_values`, `replace_position`.
