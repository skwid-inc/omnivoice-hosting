"""Microbenchmark that decomposes A100 wall into compute vs. launch overhead.

Validates the t_wall ≈ max(t_compute, t_memory, t_adhoc) model by isolating
each term on the actual hardware with OmniVoice-realistic shapes.

Sections measured:

  1. NOOP_LAUNCH       — bare kernel-launch latency (single-element op)
  2. PER_STEP_LOOP     — gen.per_i (loop) vs gen.per_i (vectorized) at varying B
                          → tells us how much T1.4 is worth
  3. TRANSFORMER_LAYER — single linear at OmniVoice shapes
                          → compute floor vs measured per-call
  4. FULL_FORWARD      — 28-layer stack at varying B
                          → models the full gen.transformer cost
  5. ITERATIVE_LOOP    — 32-step loop of FULL_FORWARD + PER_STEP_LOOP
                          → end-to-end model-prediction wall

Run:
    .venv/bin/python benchmarks/tts/scripts/measure_adhoc_cost.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

DEVICE = "cuda"
DTYPE = torch.bfloat16

# OmniVoice constants (match config.py + speed_of_light.md)
HIDDEN = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
FFN_DIM = 3072
N_LAYERS = 28
NUM_CODEBOOKS = 8
VOCAB = 1025
NUM_STEP = 32

# A100 BF16 peak (sm80, PCIe variant per F12 baseline doc)
A100_BF16_TFLOPS = 312
A100_HBM_GBPS = 1940


def sync() -> None:
    torch.cuda.synchronize()


def timed(fn, n_warmup: int = 5, n_iter: int = 50) -> dict:
    for _ in range(n_warmup):
        fn()
    sync()
    times_ms = []
    for _ in range(n_iter):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "p95_ms": times_ms[int(0.95 * len(times_ms))],
    }


def fmt(x: float) -> str:
    if x < 0.1:
        return f"{x*1000:.1f}us"
    if x < 1.0:
        return f"{x:.3f}ms"
    return f"{x:.2f}ms"


# ------------------------------------------------------------------ NOOP

def measure_noop() -> None:
    """Bare kernel-launch latency. Single-element add. ~no compute."""
    x = torch.zeros(1, device=DEVICE)

    def fn() -> None:
        x.add_(1.0)

    res = timed(fn, n_iter=200)
    print(f"\n[NOOP_LAUNCH]  single-element add (kernel-launch floor)")
    print(f"  median={fmt(res['median_ms'])}  min={fmt(res['min_ms'])}  p95={fmt(res['p95_ms'])}")
    print(f"  → ~{res['median_ms']*1000:.1f} us per kernel launch")


# ------------------------------------------------------------------ PER-STEP

def measure_per_step(B: int, S: int = 200) -> dict:
    """Mimic gen.per_i exactly: CFG add, log_softmax, masked_fill, topk, scatter.

    Compares the legacy `for i in range(B)` loop vs a fully-vectorized version.
    """
    cond = torch.randn(B, NUM_CODEBOOKS, S, VOCAB, device=DEVICE, dtype=DTYPE)
    uncond = torch.randn(B, NUM_CODEBOOKS, S, VOCAB, device=DEVICE, dtype=DTYPE)
    tokens = torch.randint(0, VOCAB, (B, NUM_CODEBOOKS, S), device=DEVICE)
    mask_id = VOCAB - 1
    layer_ids = torch.arange(NUM_CODEBOOKS, device=DEVICE).view(1, -1, 1)
    k = 8  # representative per-step k from the cosine schedule

    def fn_loop() -> None:
        for i in range(B):
            c = cond[i:i+1]
            u = uncond[i:i+1]
            lp = c + 2.0 * (c - u)
            lp = F.log_softmax(lp, dim=-1)
            lp[..., mask_id] = -float("inf")
            pred = lp.argmax(dim=-1)
            scores = lp.max(dim=-1)[0] - layer_ids * 5.0
            sample_tokens = tokens[i:i+1]
            scores.masked_fill_(sample_tokens != mask_id, -float("inf"))
            _, idx = torch.topk(scores.flatten(), k)
            flat = sample_tokens.flatten().clone()
            flat[idx] = pred.flatten()[idx]
            sample_tokens.copy_(flat.view_as(sample_tokens))

    def fn_vec() -> None:
        lp = cond + 2.0 * (cond - uncond)
        lp = F.log_softmax(lp, dim=-1)
        lp[..., mask_id] = -float("inf")
        pred = lp.argmax(dim=-1)                         # [B, 8, S]
        scores = lp.max(dim=-1)[0] - layer_ids * 5.0     # [B, 8, S]
        scores = scores.masked_fill(tokens != mask_id, -float("inf"))
        flat_scores = scores.reshape(B, -1)
        flat_pred = pred.reshape(B, -1)
        _, topk_idx = flat_scores.topk(k, dim=-1)        # [B, k]
        flat_tokens = tokens.reshape(B, -1).clone()
        src = flat_pred.gather(1, topk_idx)
        flat_tokens.scatter_(1, topk_idx, src)

    return {
        "loop": timed(fn_loop, n_iter=30),
        "vec": timed(fn_vec, n_iter=30),
    }


# ------------------------------------------------------------------ ATTENTION + FFN LAYER

class MockTransformerLayer(torch.nn.Module):
    """One Qwen-like attention + FFN layer at OmniVoice dims."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False, dtype=DTYPE, device=DEVICE)
        self.k_proj = torch.nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False, dtype=DTYPE, device=DEVICE)
        self.v_proj = torch.nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False, dtype=DTYPE, device=DEVICE)
        self.o_proj = torch.nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)
        self.gate = torch.nn.Linear(HIDDEN, FFN_DIM, bias=False, dtype=DTYPE, device=DEVICE)
        self.up = torch.nn.Linear(HIDDEN, FFN_DIM, bias=False, dtype=DTYPE, device=DEVICE)
        self.down = torch.nn.Linear(FFN_DIM, HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        v = v.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        x = x + self.o_proj(attn)
        x = x + self.down(F.silu(self.gate(x)) * self.up(x))
        return x


def measure_full_forward(B: int, S: int = 200) -> dict:
    """28-layer stack on OmniVoice shapes. Models gen.transformer's per-step cost."""
    x = torch.randn(B, S, HIDDEN, device=DEVICE, dtype=DTYPE)
    layers = torch.nn.ModuleList(
        [MockTransformerLayer() for _ in range(N_LAYERS)]
    ).to(DEVICE)

    def fn() -> None:
        h = x
        for layer in layers:
            h = layer(h)

    return timed(fn, n_warmup=3, n_iter=20)


# ------------------------------------------------------------------ FLOPS HELPERS

def per_layer_flops(B: int, S: int) -> float:
    """FLOPs per transformer layer. 2 × M × N × K per matmul."""
    qkv = 2 * B * S * HIDDEN * (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
    o = 2 * B * S * (N_HEADS * HEAD_DIM) * HIDDEN
    attn = 4 * B * N_HEADS * S * S * HEAD_DIM     # QK^T + softmax + attn @ V
    ffn = 6 * B * S * HIDDEN * FFN_DIM            # gate + up + down (3 matmuls)
    return qkv + o + attn + ffn


def compute_floor_ms(B: int, S: int) -> float:
    """Compute speed-of-light per full forward (28 layers × CFG=2)."""
    flops = N_LAYERS * 2 * per_layer_flops(B, S)  # ×2 for CFG
    return (flops / (A100_BF16_TFLOPS * 1e12)) * 1000.0


# ------------------------------------------------------------------ MAIN

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--Bs", default="1,2,4,8,16",
                   help="comma-separated batch sizes")
    p.add_argument("--S", type=int, default=200,
                   help="audio frame count (typical OmniVoice S)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this microbench needs an A100/H100")

    print(f"# Microbench on {torch.cuda.get_device_name(0)}, dtype={DTYPE}, S={args.S}")
    print(f"# Theoretical: BF16 peak={A100_BF16_TFLOPS} TFLOPS, HBM={A100_HBM_GBPS} GB/s")

    measure_noop()

    print("\n[PER_STEP_LOOP]  gen.per_i mock — loop vs vectorized")
    print(f"  {'B':>3}  {'loop':>10}  {'vectorized':>12}  {'speedup':>8}")
    for B in [int(b) for b in args.Bs.split(",")]:
        r = measure_per_step(B, args.S)
        loop_ms = r["loop"]["median_ms"]
        vec_ms = r["vec"]["median_ms"]
        print(f"  {B:>3}  {fmt(loop_ms):>10}  {fmt(vec_ms):>12}  {loop_ms/vec_ms:>7.1f}x")

    print("\n[FULL_FORWARD]  28-layer transformer stack — measured vs compute floor")
    print(f"  {'B':>3}  {'measured':>10}  {'compute floor':>14}  {'gap':>6}  {'utilization':>11}")
    for B in [int(b) for b in args.Bs.split(",")]:
        r = measure_full_forward(B, args.S)
        meas_ms = r["median_ms"]
        floor_ms = compute_floor_ms(B, args.S) / N_LAYERS / 2  # one forward, no CFG
        ratio = meas_ms / floor_ms
        util = 100.0 / ratio
        print(f"  {B:>3}  {fmt(meas_ms):>10}  {fmt(floor_ms):>14}  {ratio:>5.1f}x  {util:>10.1f}%")

    print("\n[MODEL_PREDICT]  full request wall = max(compute, t_adhoc) for 32 steps × CFG=2")
    print(f"  {'B':>3}  {'t_compute':>10}  {'t_adhoc(loop)':>14}  {'t_adhoc(vec)':>14}  "
          f"{'predicted_wall_loop':>20}  {'predicted_wall_vec':>20}")
    # Use measured per-step values from the loops above
    fwd_per_step_ms = {
        B: measure_full_forward(B, args.S)["median_ms"]
        for B in [int(b) for b in args.Bs.split(",")]
    }
    per_step_loop_ms = {
        B: measure_per_step(B, args.S)["loop"]["median_ms"]
        for B in [int(b) for b in args.Bs.split(",")]
    }
    per_step_vec_ms = {
        B: measure_per_step(B, args.S)["vec"]["median_ms"]
        for B in [int(b) for b in args.Bs.split(",")]
    }
    for B in [int(b) for b in args.Bs.split(",")]:
        # Total compute is one full forward × 32 steps × 2 (CFG)
        t_compute = fwd_per_step_ms[B] * NUM_STEP * 2
        # Per-i loop runs once per step
        t_adhoc_loop = per_step_loop_ms[B] * NUM_STEP
        t_adhoc_vec = per_step_vec_ms[B] * NUM_STEP
        wall_loop = max(t_compute, t_adhoc_loop)
        wall_vec = max(t_compute, t_adhoc_vec)
        print(
            f"  {B:>3}  {fmt(t_compute):>10}  {fmt(t_adhoc_loop):>14}  {fmt(t_adhoc_vec):>14}  "
            f"{fmt(wall_loop):>20}  {fmt(wall_vec):>20}"
        )

    print("\n# Reference F12 measured wall (PERF_A100.md):")
    print("#   B=1: 339 ms   B=2: 377 ms   B=4: 514 ms   B=8: 872 ms   B=16: 1655 ms")


if __name__ == "__main__":
    main()
