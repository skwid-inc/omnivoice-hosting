# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice Generator (Stage 0) - Iterative unmasking with Qwen3 backbone.

Generates 8-codebook audio tokens from text via 32-step non-autoregressive
iterative masked prediction with classifier-free guidance.

Uses vLLM-Omni's DiffusionAttention for optimized full (bidirectional) attention
via FlashAttention/SageAttention/SDPA backends.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig
from vllm_omni.model_executor.models.omnivoice.profiling import InferenceProfiler

logger = init_logger(__name__)


def _omnivoice_opt_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_OMNIVOICE_OPT") == "1"


def _flash_attn_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_OMNIVOICE_USE_FLASH_ATTN") == "1"


_FLASH_ATTN_VARLEN = None


def _get_flash_attn_varlen():
    """Lazy import flash_attn_varlen_func + helpers (compile lazily)."""
    global _FLASH_ATTN_VARLEN
    if _FLASH_ATTN_VARLEN is None:
        from flash_attn import flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input
        _FLASH_ATTN_VARLEN = (flash_attn_varlen_func, index_first_axis, pad_input)
    return _FLASH_ATTN_VARLEN


# ---------------------------------------------------------------------------
# Unmasking schedule helpers
# ---------------------------------------------------------------------------


def _get_time_steps(
    t_start: float,
    t_end: float,
    num_step: int,
    t_shift: float,
) -> torch.Tensor:
    """Compute the unmasking schedule with time shift.

    Returns cumulative proportions [0, ..., 1] of length num_step.
    Formula: r_n = t_shift * (n/N) / (1 + (t_shift - 1) * (n/N))
    """
    steps = torch.linspace(t_start, t_end, num_step)
    shifted = t_shift * steps / (1.0 + (t_shift - 1.0) * steps)
    return shifted


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for stochastic position selection."""
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-8)))
    return logits / max(temperature, 1e-8) + noise


# ---------------------------------------------------------------------------
# Qwen3-style transformer blocks using DiffusionAttention
# ---------------------------------------------------------------------------


class OmniVoiceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class OmniVoiceAttention(nn.Module):
    """Qwen3-style GQA attention using DiffusionAttention backend."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.hidden_size = config.llm_hidden_size
        self.num_heads = config.llm_num_attention_heads
        self.num_kv_heads = config.llm_num_key_value_heads
        self.head_dim = config.llm_head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Qwen3 uses per-head QK norm
        self.q_norm = OmniVoiceRMSNorm(self.head_dim)
        self.k_norm = OmniVoiceRMSNorm(self.head_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        flash_varlen_indices: torch.Tensor | None = None,
        flash_varlen_cu: torch.Tensor | None = None,
        flash_varlen_max: int = 0,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Per-head QK norm (Qwen3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        if cos is not None and sin is not None:
            q = _apply_rotary_pos_emb(q, cos, sin)
            k = _apply_rotary_pos_emb(k, cos, sin)

        if (_flash_attn_enabled()
                and attention_mask is not None
                and flash_varlen_indices is not None):
            # Flash-attn varlen path. Metadata (indices, cu_seqlens,
            # max_seqlen) is precomputed in _transformer_forward in eager
            # mode and threaded through here as kwargs - this avoids the
            # graph break that would otherwise fire at torch.nonzero +
            # .item() inside each compiled layer (~1ms x 13 layers per step).
            flash_varlen, index_first_axis, pad_input = _get_flash_attn_varlen()
            q_flat = q.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            k_flat = k.reshape(batch_size * seq_len, self.num_kv_heads, self.head_dim)
            v_flat = v.reshape(batch_size * seq_len, self.num_kv_heads, self.head_dim)
            q_unp = index_first_axis(q_flat, flash_varlen_indices)
            k_unp = index_first_axis(k_flat, flash_varlen_indices)
            v_unp = index_first_axis(v_flat, flash_varlen_indices)
            out_unp = flash_varlen(
                q_unp, k_unp, v_unp,
                cu_seqlens_q=flash_varlen_cu,
                cu_seqlens_k=flash_varlen_cu,
                max_seqlen_q=flash_varlen_max,
                max_seqlen_k=flash_varlen_max,
                softmax_scale=self.scale,
                causal=False,
            )  # [T, num_heads, head_dim]
            out = pad_input(out_unp, flash_varlen_indices, batch_size, seq_len)
            out = out.view(batch_size, seq_len, self.num_heads * self.head_dim)
            return self.o_proj(out)

        # SDPA fallback: GQA via repeat_interleave + dense float mask.
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        # Permute to (batch, heads, seq, head_dim) for SDPA
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Convert [B, 1, S, S] bool mask to float mask for SDPA
        sdpa_mask = None
        if attention_mask is not None:
            sdpa_mask = attention_mask.to(dtype=q.dtype)
            sdpa_mask = sdpa_mask.masked_fill(~attention_mask, float("-inf"))
            sdpa_mask = sdpa_mask.masked_fill(attention_mask, 0.0)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            scale=self.scale,
        )

        # Back to (batch, seq, heads * head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class OmniVoiceMLP(nn.Module):
    """Qwen3-style MLP with SwiGLU."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.llm_intermediate_size, config.llm_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OmniVoiceTransformerBlock(nn.Module):
    """Single Qwen3 transformer block with DiffusionAttention."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.input_layernorm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        self.self_attn = OmniVoiceAttention(config)
        self.post_attention_layernorm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        self.mlp = OmniVoiceMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        flash_varlen_indices: torch.Tensor | None = None,
        flash_varlen_cu: torch.Tensor | None = None,
        flash_varlen_max: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
            flash_varlen_indices=flash_varlen_indices,
            flash_varlen_cu=flash_varlen_cu,
            flash_varlen_max=flash_varlen_max,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def _precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tensors."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding. x shape: (B, S, H, D)."""
    seq_len = x.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1)


# ---------------------------------------------------------------------------
# Generator model
# ---------------------------------------------------------------------------


class OmniVoiceGenerator(nn.Module):
    """OmniVoice Stage 0: Iterative unmasking generator.

    Architecture:
    - Text embedding (from Qwen3 vocab) + Audio embedding (8*1025 entries)
    - 28-layer Qwen3 transformer with full bidirectional attention
    - 8-codebook prediction head (single linear: hidden → 8*1025)
    - 32-step iterative unmasking with classifier-free guidance

    Optimizations:
    - DiffusionAttention (FlashAttn/SageAttn/SDPA auto-selected)
    - TeaCache / Cache-DiT compatible (hook-based, non-intrusive)
    - regionally_compile() compatible for torch.compile on repeated blocks
    - Sequence parallelism via SP hooks for multi-GPU
    """

    # For regionally_compile() support
    _repeated_blocks = ["layers"]

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config

        # Text embedding (shared with LLM)
        self.text_embedding = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)

        # Audio embedding: 8 codebooks * 1025 tokens
        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            config.llm_hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList([OmniVoiceTransformerBlock(config) for _ in range(config.llm_num_hidden_layers)])
        self.norm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)

        # Prediction head: hidden → 8 * 1025
        self.audio_heads = nn.Linear(
            config.llm_hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        # Precompute RoPE
        self._rope_cos = None
        self._rope_sin = None

        if _omnivoice_opt_enabled():
            self._enable_compile_optimizations()

    def _enable_compile_optimizations(self) -> None:
        """Apply the engineering-only optimizations from the standalone
        OmniVoice `benchmark_optimized.py` (a039 stack), minus the patches
        that are already covered by vllm-omni's reimplementation:
        - per-layer torch.compile
        - audio_heads torch.compile
        - cuDNN/Flash/MemEff/Math SDPA backends enabled
        - fp16 reduced-precision matmul reduction
        - dynamo cache_size_limit raised so all bucket sizes stay cached
        """
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True

        compile_mode = os.environ.get(
            "VLLM_OMNI_OMNIVOICE_COMPILE_MODE", "default",
        )
        for i in range(len(self.layers)):
            self.layers[i] = torch.compile(
                self.layers[i], mode=compile_mode, dynamic=True,
            )
        self.audio_heads = torch.compile(
            self.audio_heads, mode=compile_mode, dynamic=True,
        )
        logger.info(
            "OmniVoice OPT enabled: per-layer + audio_heads compiled "
            "(mode=%s, dynamic=True), bucketed input lengths in use.",
            compile_mode,
        )

    def _ensure_rope(self, seq_len: int, device: torch.device) -> None:
        """Lazily compute RoPE cos/sin if needed."""
        if self._rope_cos is None or self._rope_cos.shape[0] < seq_len:
            max_len = max(seq_len, 4096)
            self._rope_cos, self._rope_sin = _precompute_rope(
                self.config.llm_head_dim,
                max_len,
                theta=self.config.llm_rope_theta,
                device=device,
            )

    def _prepare_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare mixed text+audio embeddings.

        Args:
            input_ids: [B, 8, S] - text tokens replicated across codebooks,
                       audio positions have per-codebook token IDs
            audio_mask: [B, S] - True for audio positions, False for text

        Returns:
            embeddings: [B, S, hidden_size]
        """
        # Text embeddings from first codebook row (all rows identical for text)
        text_embeds = self.text_embedding(input_ids[:, 0, :])

        # Audio embeddings: offset per codebook, then sum across codebooks
        shifted_ids = (input_ids * audio_mask.unsqueeze(1)) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        # Merge: audio where audio_mask=True, text elsewhere
        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def _transformer_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run through transformer layers.

        Args:
            inputs_embeds: [B, S, hidden_size]
            attention_mask: [B, 1, S, S] or None

        Returns:
            hidden_states: [B, S, hidden_size]
        """
        device = inputs_embeds.device
        seq_len = inputs_embeds.shape[1]
        self._ensure_rope(seq_len, device)

        hidden_states = inputs_embeds
        cos = self._rope_cos.to(device=device, dtype=hidden_states.dtype)
        sin = self._rope_sin.to(device=device, dtype=hidden_states.dtype)

        # Precompute varlen metadata ONCE per step in eager mode, then
        # pass tensors through to each compiled layer. Computing this
        # inside the compiled layer triggers graph breaks (torch.nonzero +
        # .item()) which costs ~1 ms per attention call * 13 layers.
        flash_varlen_indices = None
        flash_varlen_cu = None
        flash_varlen_max = 0
        if _flash_attn_enabled() and attention_mask is not None:
            mask_2d = attention_mask[:, 0, 0, :]
            seqlens = mask_2d.sum(dim=-1, dtype=torch.int32)
            flash_varlen_indices = torch.nonzero(
                mask_2d.flatten(), as_tuple=False
            ).flatten()
            flash_varlen_cu = F.pad(
                torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
            )
            flash_varlen_max = int(seqlens.max().item())

        # The torch.compile wrapper around each layer captures the
        # signature seen at compile time and rejects unknown kwargs at
        # call time. Only pass the varlen kwargs when flash-attn is
        # actually enabled - SDPA path uses the original signature so
        # the legacy compile cache stays valid.
        for layer in self.layers:
            if flash_varlen_indices is not None:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    cos=cos,
                    sin=sin,
                    flash_varlen_indices=flash_varlen_indices,
                    flash_varlen_cu=flash_varlen_cu,
                    flash_varlen_max=flash_varlen_max,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    cos=cos,
                    sin=sin,
                )

        return self.norm(hidden_states)

    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to per-codebook logits.

        Args:
            hidden_states: [B, S, hidden_size]

        Returns:
            logits: [B, 8, S, 1025]
        """
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)  # [B, S, 8*1025]
        return logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)  # [B, 8, S, 1025]

    @staticmethod
    def _resolve_block_size(max_target_len: int) -> int:
        """Resolve temporal block size for block-wise unmasking."""
        block_size_env = os.environ.get("VLLM_OMNI_OMNIVOICE_BLOCK_SIZE", "0")
        try:
            block_size = int(block_size_env)
        except ValueError:
            block_size = 0
        if block_size <= 0:
            block_size = max_target_len
        return block_size

    @staticmethod
    def _resolve_first_block_num_step(num_step: int) -> int:
        first_block_steps_env = os.environ.get(
            "VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP",
        )
        if first_block_steps_env is None:
            return num_step
        try:
            first_block_steps = int(first_block_steps_env)
        except ValueError:
            return num_step
        return max(1, first_block_steps)

    def build_step_plans(
        self,
        target_lens: list[int],
        num_step: int,
        t_shift: float,
    ) -> list[list[tuple[int, int, int]]]:
        """Build per-request frame-block unmasking plans.

        Each step entry is ``(frame_start, frame_end, k)`` where ``k`` is the
        number of codebook/frame positions to unmask for that request.
        ``num_step`` is applied to each temporal block. With block-wise
        streaming enabled, a request with ``N`` blocks therefore runs
        ``N * num_step`` denoising steps.
        """
        max_target_len = max(target_lens)
        block_size = self._resolve_block_size(max_target_len)
        first_block_num_step = self._resolve_first_block_num_step(num_step)
        num_codebooks = self.config.num_audio_codebook

        step_plans: list[list[tuple[int, int, int]]] = []
        for t_len in target_lens:
            num_blocks = (t_len + block_size - 1) // block_size
            num_blocks = max(1, num_blocks)
            eff_block_size = (t_len + num_blocks - 1) // num_blocks
            plan: list[tuple[int, int, int]] = []
            for b in range(num_blocks):
                f_start = b * eff_block_size
                f_end = min(f_start + eff_block_size, t_len)
                b_total = (f_end - f_start) * num_codebooks
                b_steps = first_block_num_step if b == 0 else num_step
                sub_ts = _get_time_steps(0.0, 1.0, b_steps + 1, t_shift).tolist()
                rem = b_total
                for sb in range(b_steps):
                    if sb == b_steps - 1:
                        k_sb = rem
                    else:
                        k_sb = min(
                            math.ceil(b_total * (sub_ts[sb + 1] - sub_ts[sb])),
                            rem,
                        )
                    plan.append((f_start, f_end, int(k_sb)))
                    rem -= int(k_sb)
            expected_steps = first_block_num_step + (num_blocks - 1) * num_step
            assert len(plan) == expected_steps, (
                f"step plan length {len(plan)} != expected_steps {expected_steps} "
                f"for t_len={t_len} block_size={block_size} num_step={num_step} "
                f"first_block_num_step={first_block_num_step}"
            )
            step_plans.append(plan)
        return step_plans

    @staticmethod
    def is_block_complete(step_plan: list[tuple[int, int, int]], step_index: int) -> bool:
        """Return True when ``step_index`` completes its active frame block."""
        if step_index >= len(step_plan) - 1:
            return True
        cur_start, cur_end, _ = step_plan[step_index]
        next_start, next_end, _ = step_plan[step_index + 1]
        return (cur_start, cur_end) != (next_start, next_end)

    def unmask_step(
        self,
        *,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
        target_lens: list[int],
        step_indices: list[int],
        step_plans: list[list[tuple[int, int, int]]],
        c_lens: list[int],
        guidance_scale: float = 2.0,
        layer_penalty_factor: float = 5.0,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
        prof: InferenceProfiler | None = None,
    ) -> None:
        """Run one batched transformer step and mutate ``tokens``/``input_ids``.

        ``step_indices`` can differ per request, which lets the engine batch
        concurrent streams that joined the scheduler at different times.
        """
        B = len(target_lens)
        if prof is None:
            prof = InferenceProfiler.current()

        cudagraphs_enabled = os.environ.get(
            "VLLM_OMNI_OMNIVOICE_COMPILE_MODE", "default",
        ) in ("reduce-overhead", "max-autotune")
        if cudagraphs_enabled:
            torch.compiler.cudagraph_mark_step_begin()

        mask_id = self.config.audio_mask_id
        num_codebooks = self.config.num_audio_codebook
        layer_ids = torch.arange(num_codebooks, device=input_ids.device).view(1, -1, 1)

        with prof.section("gen.transformer"):
            inputs_embeds = self._prepare_embeddings(input_ids, audio_mask)
            hidden_states = self._transformer_forward(inputs_embeds, attention_mask)
            batch_logits = self._get_logits(hidden_states).to(torch.float32)

        with prof.section("gen.log_softmax"):
            cond_log_probs = F.log_softmax(batch_logits[:B], dim=-1)
            uncond_log_probs = (
                F.log_softmax(batch_logits[B:], dim=-1)
                if guidance_scale != 0 else None
            )

        with prof.section("gen.per_i"):
            for i in range(B):
                step_index = step_indices[i]
                if step_index < 0 or step_index >= len(step_plans[i]):
                    continue
                f_start, f_end, k = step_plans[i][step_index]
                if k <= 0:
                    continue

                c_len = c_lens[i]
                t_len = target_lens[i]
                c_log_probs = cond_log_probs[i : i + 1, :, c_len - t_len : c_len, :]

                if guidance_scale != 0:
                    assert uncond_log_probs is not None
                    u_log_probs = uncond_log_probs[i : i + 1, :, :t_len, :]
                    log_probs = torch.log_softmax(
                        c_log_probs + guidance_scale * (c_log_probs - u_log_probs),
                        dim=-1,
                    )
                else:
                    log_probs = c_log_probs

                log_probs[..., mask_id] = -float("inf")
                if class_temperature > 0.0:
                    pred_tokens = _gumbel_sample(log_probs, class_temperature).argmax(dim=-1)
                else:
                    pred_tokens = log_probs.argmax(dim=-1)

                scores = log_probs.max(dim=-1)[0]
                scores = scores - (layer_ids * layer_penalty_factor)
                if position_temperature > 0.0:
                    scores = _gumbel_sample(scores, position_temperature)

                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(sample_tokens != mask_id, -float("inf"))
                if f_start > 0:
                    scores[..., :f_start] = -float("inf")
                if f_end < t_len:
                    scores[..., f_end:] = -float("inf")

                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten().clone()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                tokens[i : i + 1, :, :t_len] = sample_tokens
                input_ids[i, :, c_len - t_len : c_len] = sample_tokens.squeeze(0)
                input_ids[B + i, :, :t_len] = sample_tokens.squeeze(0)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        target_lens: list[int],
        num_step: int = 32,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        layer_penalty_factor: float = 5.0,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
    ) -> torch.Tensor:
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
        """Run the full 32-step iterative unmasking generation.

        Args:
            input_ids: [2*B, 8, S] - conditional (0:B) + unconditional (B:2B)
            audio_mask: [2*B, S] - True for audio positions
            attention_mask: [2*B, 1, S, S] - attention mask
            target_lens: List of target audio lengths per batch item
            num_step: Number of unmasking steps
            guidance_scale: CFG scale
            t_shift: Time shift for schedule
            layer_penalty_factor: Penalty for later codebooks
            position_temperature: Gumbel temperature for position selection
            class_temperature: Temperature for token prediction (0=greedy)

        Returns:
            tokens: [B, 8, max_target_len] - generated audio tokens
        """
        B = len(target_lens)
        _log_batch = os.environ.get("VLLM_OMNI_OMNIVOICE_LOG_BATCH") == "1"
        if _log_batch:
            logger.info("OmniVoiceGenerator.forward: B=%d target_lens=%s", B, list(target_lens))
            _t_start = torch.cuda.Event(enable_timing=True)
            _t_end = torch.cuda.Event(enable_timing=True)
            _t_start.record()
        device = input_ids.device
        max_target_len = max(target_lens)
        mask_id = self.config.audio_mask_id
        num_codebooks = self.config.num_audio_codebook

        # Initialize all target tokens as [MASK]
        tokens = torch.full(
            (B, num_codebooks, max_target_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        step_plans = self.build_step_plans(target_lens, num_step, t_shift)

        # Compute c_lens for extracting target region from full sequence
        c_lens = []
        for i in range(B):
            # Conditional sequence length = number of non-padding positions
            c_len = attention_mask[i, 0, 0].sum().item()
            c_lens.append(int(c_len))

        # Clone the caller's input_ids once so we can mutate in place during
        # the unmasking loop without aliasing. The previous code cloned per
        # (step, i), which allocated B * num_step new tensors per request and
        # added significant Python + CUDA-launch overhead at high B.
        input_ids = input_ids.clone()

        prof = InferenceProfiler.current()

        total_steps = max(len(plan) for plan in step_plans)
        for step in range(total_steps):
            self.unmask_step(
                input_ids=input_ids,
                audio_mask=audio_mask,
                attention_mask=attention_mask,
                tokens=tokens,
                target_lens=target_lens,
                step_indices=[step] * B,
                step_plans=step_plans,
                c_lens=c_lens,
                guidance_scale=guidance_scale,
                layer_penalty_factor=layer_penalty_factor,
                position_temperature=position_temperature,
                class_temperature=class_temperature,
                prof=prof,
            )

        if _log_batch:
            _t_end.record()
            torch.cuda.synchronize()
            _ms = _t_start.elapsed_time(_t_end)
            logger.info("OmniVoiceGenerator.forward: B=%d wall_ms=%.1f", B, _ms)

        # Phase A debug: assert that every active position got unmasked.
        # If any (request, codebook, frame) within target_lens still holds
        # mask_id, the downstream decoder embedding lookup will OOB and
        # CUDA-assert. Fail loudly here so the failure is localized.
        if os.environ.get("VLLM_OMNI_OMNIVOICE_BLOCK_DEBUG") == "1":
            for i in range(B):
                t_len_i = target_lens[i]
                bad = (tokens[i, :, :t_len_i] == mask_id).nonzero(as_tuple=False)
                if bad.numel() > 0:
                    logger.error(
                        "OmniVoiceGenerator: residual mask in request %d "
                        "(t_len=%d) at %d positions; first few=%s; "
                        "step_plan=%s",
                        i, t_len_i, bad.shape[0],
                        bad[:5].tolist(),
                        step_plans[i],
                    )
                    raise RuntimeError(
                        f"residual mask_id in request {i} t_len={t_len_i}"
                    )
        return tokens

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load weights from HuggingFace OmniVoice model.safetensors.

        The HF checkpoint contains:
        - llm.* -> Qwen3 transformer weights
        - audio_embeddings.* -> audio embedding table
        - audio_heads.* -> prediction head
        """
        import os

        from safetensors.torch import load_file

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        state_dict = load_file(weights_path, device=str(device))

        # Map HF weight names to our module names
        loaded_keys = set()

        # 1. Text embedding: llm.embed_tokens.weight -> text_embedding.weight
        text_emb_key = "llm.embed_tokens.weight"
        if text_emb_key in state_dict:
            self.text_embedding.weight.data.copy_(state_dict[text_emb_key])
            loaded_keys.add(text_emb_key)

        # 2. Audio embeddings
        for key in ["audio_embeddings.weight"]:
            if key in state_dict:
                self.audio_embeddings.weight.data.copy_(state_dict[key])
                loaded_keys.add(key)

        # 3. Audio heads
        for key in ["audio_heads.weight"]:
            if key in state_dict:
                self.audio_heads.weight.data.copy_(state_dict[key])
                loaded_keys.add(key)

        # 4. Transformer layers: llm.layers.N.* -> layers.N.*
        for key, value in state_dict.items():
            if key.startswith("llm.layers."):
                # llm.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
                our_key = key.replace("llm.layers.", "layers.")
                parts = our_key.split(".")
                module = self
                try:
                    for part in parts[:-1]:
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    param_name = parts[-1]
                    param = getattr(module, param_name)
                    if isinstance(param, nn.Parameter):
                        param.data.copy_(value)
                    elif isinstance(param, torch.Tensor):
                        param.copy_(value)
                    loaded_keys.add(key)
                except (AttributeError, IndexError, KeyError) as e:
                    logger.warning("Failed to load weight %s: %s", key, e)

        # 5. Final norm: llm.norm.weight -> norm.weight
        norm_key = "llm.norm.weight"
        if norm_key in state_dict:
            self.norm.weight.data.copy_(state_dict[norm_key])
            loaded_keys.add(norm_key)

        unloaded = set(state_dict.keys()) - loaded_keys
        # Filter out audio_tokenizer weights (loaded in decoder stage)
        unloaded = {k for k in unloaded if not k.startswith("audio_tokenizer.")}
        if unloaded:
            logger.info(
                "Generator: %d/%d weights loaded, %d skipped (decoder weights)",
                len(loaded_keys),
                len(state_dict),
                len(unloaded),
            )
        else:
            logger.info("Generator: all %d weights loaded", len(loaded_keys))
