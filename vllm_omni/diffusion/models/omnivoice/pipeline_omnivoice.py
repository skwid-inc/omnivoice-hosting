# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice TTS Pipeline for vLLM-Omni diffusion engine.

Single-stage pipeline that runs the full text-to-speech flow:
  text → tokenize → 32-step iterative unmasking → 8-codebook tokens → DAC decode → 24kHz audio

Uses request-mode execution (all steps in one forward() call).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import soundfile as sf
import torch
from tokenizers import Tokenizer as HFTokenizer
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState, RunnerOutput
from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import OmniVoiceGenerator
from vllm_omni.model_executor.models.omnivoice.profiling import InferenceProfiler

try:
    from transformers import HiggsAudioV2TokenizerModel
except ImportError:
    HiggsAudioV2TokenizerModel = None

import torchaudio

logger = init_logger(__name__)

_REF_AUDIO_MIN_DURATION = 1.0
_REF_AUDIO_MAX_DURATION = 30.0


def _resolve_generator_dtype_from_env() -> torch.dtype | None:
    dtype_name = os.environ.get("VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE")
    if not dtype_name:
        return None

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = dtype_name.strip().lower()
    if key in dtype_map:
        return dtype_map[key]

    logger.warning(
        "Invalid VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE=%r; keeping generator dtype unchanged.",
        dtype_name,
    )
    return None


def get_omnivoice_post_process_func(od_config: OmniDiffusionConfig):
    """Post-processing: convert audio tensor to numpy for WAV encoding."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type == "pt":
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


class OmniVoicePipeline(nn.Module, SupportAudioOutput):
    """OmniVoice text-to-speech pipeline for the diffusion engine.

    Wraps OmniVoiceGenerator (32-step iterative unmasking) and
    OmniVoiceDecoder (HiggsAudioV2 RVQ + DAC) into a single forward() call.
    """

    support_audio_output: ClassVar[bool] = True
    supports_step_execution: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = od_config.model

        # Resolve model path (HF hub ID → local cache)
        if not os.path.isdir(self.model_path):
            from huggingface_hub import snapshot_download

            self.model_path = snapshot_download(self.model_path)

        # Load OmniVoice config
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path) as f:
            hf_config = json.load(f)
        self.config = OmniVoiceConfig(**hf_config)

        # Build generator and decoder
        self.generator = OmniVoiceGenerator(self.config)
        self.decoder = OmniVoiceDecoder(self.config)

        # Tokenizer (low-level, avoids HF tokenizer extra_special_tokens issue)
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
        self.tokenizer = HFTokenizer.from_file(tokenizer_path)

        # Audio tokenizer for voice cloning (requires transformers>=5.3)
        if HiggsAudioV2TokenizerModel is not None:
            audio_tokenizer_path = os.path.join(self.model_path, "audio_tokenizer")
            self.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                audio_tokenizer_path, device_map=self.device
            ).eval()
            logger.info("HiggsAudioV2 tokenizer loaded for voice cloning on %s", self.device)
        else:
            self.audio_tokenizer = None
            logger.warning("Voice cloning disabled (requires transformers>=5.3.0).")

        self.voice_presets = self._load_voice_presets()
        self.default_voice = os.environ.get(
            "VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE", ""
        ).strip().lower()

        # Duration estimator
        self.duration_estimator = RuleDurationEstimator()

        # Generation parameters. NUM_STEP env override lets us trade audio
        # quality for ~2x latency (32 -> 16 = ~50% faster transformer time).
        self.num_step = int(
            os.environ.get("VLLM_OMNI_OMNIVOICE_NUM_STEP", self.config.num_step)
        )
        self.guidance_scale = self.config.guidance_scale
        self.t_shift = self.config.t_shift
        self.layer_penalty_factor = self.config.layer_penalty_factor
        self.position_temperature = self.config.position_temperature
        self.class_temperature = self.config.class_temperature
        self.sample_rate = self.config.sample_rate

    def _encode_ref_audio(self, audio_signal: torch.Tensor, sr: int) -> torch.Tensor:
        """Encode reference audio to 8-codebook tokens for voice cloning."""
        if self.audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not available for voice cloning")
        if audio_signal.dim() == 1:
            audio_signal = audio_signal.unsqueeze(0)
        # Resample to tokenizer's expected sample rate
        target_sr = self.audio_tokenizer.config.sample_rate
        if sr != target_sr:
            audio_signal = torchaudio.functional.resample(audio_signal, sr, target_sr)
        # Ensure mono [B, 1, samples]
        if audio_signal.dim() == 2:
            audio_signal = audio_signal.unsqueeze(1)
        with torch.inference_mode():
            tokens = self.audio_tokenizer.encode(
                audio_signal.to(self.audio_tokenizer.device), return_dict=False
            )  # [B, 8, T_ref]
            tokens = tokens.squeeze(0)  # [8, T_ref]
        return tokens

    def _load_voice_presets(self) -> dict[str, dict[str, Any]]:
        """Load server-side voice cloning presets from local disk.

        ``VLLM_OMNI_OMNIVOICE_VOICE_MAP`` may be either a JSON file path or an
        inline JSON object. Each voice maps to a local reference audio file and
        optional transcript/lang/instruction metadata.
        """
        raw_spec = os.environ.get("VLLM_OMNI_OMNIVOICE_VOICE_MAP", "").strip()
        if not raw_spec:
            return {}
        if self.audio_tokenizer is None:
            raise RuntimeError(
                "VLLM_OMNI_OMNIVOICE_VOICE_MAP requires transformers>=5.3.0 "
                "so the HiggsAudioV2 tokenizer can encode reference voices."
            )

        config, base_dir = self._load_voice_map_config(raw_spec)
        presets: dict[str, dict[str, Any]] = {}
        for voice_name, entry in self._iter_voice_entries(config):
            key = voice_name.strip().lower()
            if not key:
                raise ValueError("OmniVoice voice preset names cannot be empty.")
            audio_path = self._voice_entry_audio_path(entry)
            ref_text = (
                entry.get("ref_text")
                or entry.get("transcript")
                or entry.get("text")
                or ""
            )
            audio_signal, sr = self._load_voice_audio(audio_path, base_dir)
            tokens = self._encode_ref_audio(audio_signal, sr).to(self.device)
            presets[key] = {
                "name": voice_name,
                "ref_audio_tokens": tokens,
                "ref_text": str(ref_text).strip(),
                "lang": entry.get("lang") or entry.get("language"),
                "instruct": entry.get("instruct") or entry.get("instructions"),
                "path": str(audio_path),
            }

        logger.info(
            "Loaded %d OmniVoice voice preset(s): %s",
            len(presets),
            ", ".join(sorted(presets)),
        )
        return presets

    @staticmethod
    def _load_voice_map_config(raw_spec: str) -> tuple[Any, Path | None]:
        expanded = Path(os.path.expandvars(os.path.expanduser(raw_spec)))
        if expanded.exists():
            with open(expanded) as f:
                return json.load(f), expanded.resolve().parent
        try:
            return json.loads(raw_spec), None
        except json.JSONDecodeError as exc:
            raise ValueError(
                "VLLM_OMNI_OMNIVOICE_VOICE_MAP must be a JSON file path or "
                "an inline JSON object."
            ) from exc

    @staticmethod
    def _iter_voice_entries(config: Any) -> Iterable[tuple[str, dict[str, Any]]]:
        if isinstance(config, dict) and "voices" in config:
            config = config["voices"]

        if isinstance(config, list):
            for entry in config:
                if not isinstance(entry, dict):
                    raise ValueError("Voice map list entries must be objects.")
                name = entry.get("name") or entry.get("voice")
                if not name:
                    raise ValueError("Voice map list entries require 'name' or 'voice'.")
                yield str(name), entry
            return

        if not isinstance(config, dict):
            raise ValueError("Voice map must be an object, a list, or contain a 'voices' field.")

        for name, entry in config.items():
            if isinstance(entry, str):
                entry = {"audio": entry}
            if not isinstance(entry, dict):
                raise ValueError(f"Voice preset {name!r} must be an object or audio path string.")
            yield str(name), entry

    @staticmethod
    def _voice_entry_audio_path(entry: dict[str, Any]) -> Path:
        audio_path = (
            entry.get("audio")
            or entry.get("ref_audio")
            or entry.get("path")
            or entry.get("file")
        )
        if not audio_path:
            raise ValueError("Voice preset requires 'audio', 'ref_audio', 'path', or 'file'.")
        return Path(os.path.expandvars(os.path.expanduser(str(audio_path))))

    @staticmethod
    def _load_voice_audio(audio_path: Path, base_dir: Path | None) -> tuple[torch.Tensor, int]:
        if not audio_path.is_absolute() and base_dir is not None:
            audio_path = base_dir / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"OmniVoice reference audio not found: {audio_path}")

        wav_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=-1)
        sr = int(sr)
        duration = len(wav_np) / sr if sr > 0 else 0.0
        if duration < _REF_AUDIO_MIN_DURATION:
            raise ValueError(
                f"Reference audio {audio_path} is too short ({duration:.1f}s). "
                f"At least {_REF_AUDIO_MIN_DURATION:.0f}s of clear speech is required."
            )
        if duration > _REF_AUDIO_MAX_DURATION:
            raise ValueError(
                f"Reference audio {audio_path} is too long ({duration:.1f}s). "
                f"Maximum {_REF_AUDIO_MAX_DURATION:.0f}s supported."
            )
        return torch.from_numpy(np.ascontiguousarray(wav_np)), sr

    def _resolve_voice_preset(self, voice: object | None) -> dict[str, Any] | None:
        if not self.voice_presets:
            return None

        voice_name = str(voice).strip() if voice else ""
        if not voice_name:
            voice_name = self.default_voice or ("default" if "default" in self.voice_presets else "")
        elif (
            voice_name.lower() == "default"
            and "default" not in self.voice_presets
            and self.default_voice
        ):
            voice_name = self.default_voice

        if not voice_name:
            return None

        key = voice_name.lower()
        preset = self.voice_presets.get(key)
        if preset is None:
            available = ", ".join(sorted(self.voice_presets))
            raise ValueError(
                f"Unknown OmniVoice voice {voice_name!r}. Available voices: {available}"
            )
        return preset

    def _prepare_one_request(self, prompt) -> dict | None:
        """Tokenize a single prompt + (optional) reference audio.

        Returns a dict with cond_ids, uncond_ids, cond_len, uncond_len,
        text_len, target_len. Returns None on empty text.
        """
        ref_audio = None
        ref_text = None
        lang = "None"
        instruct = "None"
        voice_preset = None

        if isinstance(prompt, dict):
            text = prompt.get("input", prompt.get("text", str(prompt)))
            ref_audio = prompt.get("ref_audio")
            ref_text = prompt.get("ref_text")
            lang = prompt.get("lang") or "None"
            instruct = prompt.get("instruct") or "None"
            if ref_audio is None:
                voice_preset = self._resolve_voice_preset(prompt.get("voice"))
        else:
            text = str(prompt)
            voice_preset = self._resolve_voice_preset(None)

        if voice_preset is not None and ref_audio is None:
            ref_text = ref_text or voice_preset.get("ref_text")
            lang = "None" if lang == "None" else lang
            lang = lang if lang != "None" else (voice_preset.get("lang") or "None")
            instruct = "None" if instruct == "None" else instruct
            instruct = instruct if instruct != "None" else (
                voice_preset.get("instruct") or "None"
            )

        if not text:
            return None

        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id

        target_len = self.duration_estimator.estimate_duration(text, "Nice to meet you.", 25)
        target_len = max(1, int(target_len))

        style = f"<|denoise|><|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
        full_text = f"{ref_text} {text}" if ref_text else text
        full_prompt = f"{style}<|text_start|>{full_text}<|text_end|>"
        encoding = self.tokenizer.encode(full_prompt)
        text_tokens = torch.tensor(encoding.ids, dtype=torch.long, device=device)
        text_len = text_tokens.shape[0]

        ref_audio_tokens = None
        if ref_audio is not None:
            if self.audio_tokenizer is None:
                raise RuntimeError(
                    "Voice cloning requires transformers>=5.3.0. "
                    "Try: uv pip install 'transformers>=5.3.0'"
                )
            audio_signal, sr = ref_audio
            if isinstance(audio_signal, np.ndarray):
                audio_signal = torch.from_numpy(audio_signal).float()
            ref_audio_tokens = self._encode_ref_audio(audio_signal, int(sr)).to(device)
        elif voice_preset is not None:
            ref_audio_tokens = voice_preset["ref_audio_tokens"].to(device)

        text_ids = text_tokens.unsqueeze(0).repeat(num_cb, 1)
        target_ids = torch.full(
            (num_cb, target_len), mask_id, dtype=torch.long, device=device,
        )

        if ref_audio_tokens is not None:
            cond_ids = torch.cat([text_ids, ref_audio_tokens, target_ids], dim=1)
        else:
            cond_ids = torch.cat([text_ids, target_ids], dim=1)
        cond_len = cond_ids.shape[1]
        uncond_ids = target_ids
        uncond_len = target_len

        return {
            "cond_ids": cond_ids,
            "uncond_ids": uncond_ids,
            "cond_len": cond_len,
            "uncond_len": uncond_len,
            "text_len": text_len,
            "target_len": target_len,
        }

    def _build_batch_tensors(
        self,
        per_req: list[dict],
        device: torch.device,
        num_cb: int,
        mask_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Build generator inputs for a batch of prepared OmniVoice requests."""
        B = len(per_req)
        max_len = max(max(r["cond_len"], r["uncond_len"]) for r in per_req)

        batch_input_ids = torch.full(
            (2 * B, num_cb, max_len), mask_id, dtype=torch.long, device=device,
        )
        batch_audio_mask = torch.zeros(
            (2 * B, max_len), dtype=torch.bool, device=device,
        )
        batch_attn_mask = torch.zeros(
            (2 * B, 1, max_len, max_len), dtype=torch.bool, device=device,
        )

        target_lens: list[int] = []
        for i, r in enumerate(per_req):
            c_len = r["cond_len"]
            u_len = r["uncond_len"]
            text_len = r["text_len"]
            batch_input_ids[i, :, :c_len] = r["cond_ids"]
            batch_input_ids[B + i, :, :u_len] = r["uncond_ids"]
            batch_audio_mask[i, text_len:c_len] = True
            batch_audio_mask[B + i, :u_len] = True
            batch_attn_mask[i, :, :c_len, :c_len] = True
            batch_attn_mask[B + i, :, :u_len, :u_len] = True
            target_lens.append(r["target_len"])

        return batch_input_ids, batch_audio_mask, batch_attn_mask, target_lens

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Generate speech audio for one or more prompts.

        Multi-prompt batching: when ``len(req.prompts) > 1`` the generator
        runs a single ``[2*B, 8, S]`` forward pass that produces tokens for
        all B requests in lockstep. The decoder is invoked per-request to
        avoid mixing in masked/padding positions of shorter targets. The
        result is wrapped as a list of B audio tensors, which the engine
        already knows how to split back to per-HTTP-request outputs.
        """
        prompts = list(req.prompts) if req.prompts else []
        if not prompts:
            return DiffusionOutput(error="Empty prompt list")

        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id

        with InferenceProfiler.scope(batch_size=len(prompts)) as prof:
            return self._forward_inner(prompts, device, num_cb, mask_id, prof)

    def _forward_inner(self, prompts, device, num_cb, mask_id, prof):
        with prof.section("prepare_inputs"):
            per_req = []
            for prompt in prompts:
                prepared = self._prepare_one_request(prompt)
                if prepared is None:
                    return DiffusionOutput(error="Empty text prompt")
                per_req.append(prepared)

        B = len(per_req)
        with prof.section("build_batch_tensors"):
            batch_input_ids, batch_audio_mask, batch_attn_mask, target_lens = (
                self._build_batch_tensors(per_req, device, num_cb, mask_id)
            )

        with prof.section("generator"):
            tokens = self.generator(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attn_mask,
                target_lens=target_lens,
                num_step=self.num_step,
                guidance_scale=self.guidance_scale,
                t_shift=self.t_shift,
                layer_penalty_factor=self.layer_penalty_factor,
                position_temperature=self.position_temperature,
                class_temperature=self.class_temperature,
            )  # [B, 8, max_target_len]

        if B == 1:
            with prof.section("decoder"):
                audio = self.decoder(tokens)  # [1, 1, samples]
            return DiffusionOutput(output=audio)

        # Batched decode: at B>1 calling self.decoder once per row was the
        # dominant cost (e.g. ~50ms x B at B=8 dwarfs the generator gain
        # from batching). Decode the whole [B, 8, max_target_len] tensor in
        # a single call and crop the output per-row.
        #
        # Rows are padded with audio_mask_id beyond their target_len, which
        # would feed garbage into the decoder near the end of each row via
        # the conv receptive field. To get a clean tail we replace each
        # row's padding with that row's last valid token, which the DAC
        # decoder treats as a continuation rather than an OOV token.
        with prof.section("decoder"):
            max_t = tokens.shape[-1]
            if any(t < max_t for t in target_lens):
                for i, t_len in enumerate(target_lens):
                    if t_len < max_t:
                        last_valid = tokens[i : i + 1, :, t_len - 1 : t_len]
                        tokens[i : i + 1, :, t_len:] = last_valid
            audio_batch = self.decoder(tokens)  # [B, 1, max_target_len * spm]
            samples_per_token = audio_batch.shape[-1] // max_t
            audios = []
            for i, t_len in enumerate(target_lens):
                crop = audio_batch[i : i + 1, :, : t_len * samples_per_token]
                audios.append(crop)

        return DiffusionOutput(output=audios)

    def _stream_holdback_frames(self) -> int:
        try:
            return max(
                0,
                int(os.environ.get("VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES", "2")),
            )
        except ValueError:
            return 2

    def prepare_encode(self, state: DiffusionRequestState, **kwargs) -> DiffusionRequestState:
        """Initialize one OmniVoice request for step-wise block streaming."""
        prompts = list(state.prompts or [])
        if len(prompts) != 1:
            raise ValueError(
                f"OmniVoice step streaming expects one prompt per request, got {len(prompts)}"
            )

        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id
        prepared = self._prepare_one_request(prompts[0])
        if prepared is None:
            raise ValueError("Empty text prompt")

        input_ids, audio_mask, attention_mask, target_lens = self._build_batch_tensors(
            [prepared], device, num_cb, mask_id,
        )
        target_len = target_lens[0]
        tokens = torch.full(
            (1, num_cb, target_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        step_plans = self.generator.build_step_plans(
            target_lens,
            self.num_step,
            self.t_shift,
        )
        step_plan = step_plans[0]

        state.step_index = int(state.sampling.step_index or 0)
        state.timesteps = torch.arange(len(step_plan), device=device)
        state.extra["omnivoice"] = {
            "input_ids": input_ids.clone(),
            "audio_mask": audio_mask,
            "attention_mask": attention_mask,
            "tokens": tokens,
            "target_len": target_len,
            "c_len": int(prepared["cond_len"]),
            "step_plan": step_plan,
            "emitted_samples": 0,
            "stream_audio_blocks": (
                isinstance(prompts[0], dict)
                and bool(prompts[0].get("_stream_audio_blocks"))
                and bool(prompts[0].get("_stream_output_enabled"))
            ),
        }
        return state

    def denoise_step(self, state: DiffusionRequestState, **kwargs) -> torch.Tensor | None:
        raise NotImplementedError("OmniVoice uses execute_step_batch() for step execution.")

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor | None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("OmniVoice uses execute_step_batch() for step execution.")

    def post_decode(self, state: DiffusionRequestState, **kwargs) -> DiffusionOutput:
        data = state.extra.get("omnivoice")
        if data is None:
            return DiffusionOutput(error="OmniVoice step state is missing.")
        chunk = self._decode_stream_delta(data, finished=True)
        if data.get("stream_audio_blocks"):
            return DiffusionOutput(output={"audio": chunk, "sr": self.sample_rate})
        return DiffusionOutput(output=chunk)

    def _decode_stream_delta(self, data: dict, finished: bool) -> torch.Tensor:
        target_len = int(data["target_len"])
        step_plan = data["step_plan"]
        current_step = int(data.get("last_step_index", 0))
        if finished:
            decode_frames = target_len
        else:
            _, decode_frames, _ = step_plan[current_step]
        decode_frames = max(0, min(decode_frames, target_len))
        if decode_frames <= 0:
            return torch.empty(0, dtype=torch.float32)

        tokens = data["tokens"][:, :, :decode_frames]
        with InferenceProfiler.current().section("decoder_stream"):
            audio_prefix = self.decoder(tokens)

        samples_per_token = max(1, audio_prefix.shape[-1] // decode_frames)
        emit_end = audio_prefix.shape[-1]
        if not finished:
            holdback = self._stream_holdback_frames() * samples_per_token
            emit_end = max(0, emit_end - holdback)

        emitted_samples = int(data.get("emitted_samples", 0))
        emit_end = max(emitted_samples, emit_end)
        chunk = audio_prefix[..., emitted_samples:emit_end]
        data["emitted_samples"] = emit_end
        return chunk.detach().cpu()

    @torch.inference_mode()
    def execute_step_batch(
        self,
        state_items: list[tuple[DiffusionRequestState, bool]],
    ) -> RunnerOutput:
        """Execute one batched OmniVoice unmasking step across active streams."""
        if not state_items:
            return RunnerOutput(req_id="", step_index=None, finished=True)

        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id
        per_request_results: dict[str, DiffusionOutput] = {}
        per_request_finished: dict[str, bool] = {}
        per_request_step_indices: dict[str, int] = {}

        with InferenceProfiler.scope(batch_size=len(state_items)) as prof:
            for state, is_new_request in state_items:
                if is_new_request or "omnivoice" not in state.extra:
                    self.prepare_encode(state)

            B = len(state_items)
            request_data = [state.extra["omnivoice"] for state, _ in state_items]
            max_seq_len = max(data["input_ids"].shape[-1] for data in request_data)
            max_target_len = max(int(data["target_len"]) for data in request_data)

            with prof.section("build_stream_batch"):
                batch_input_ids = torch.full(
                    (2 * B, num_cb, max_seq_len),
                    mask_id,
                    dtype=torch.long,
                    device=device,
                )
                batch_audio_mask = torch.zeros(
                    (2 * B, max_seq_len),
                    dtype=torch.bool,
                    device=device,
                )
                batch_attn_mask = torch.zeros(
                    (2 * B, 1, max_seq_len, max_seq_len),
                    dtype=torch.bool,
                    device=device,
                )
                batch_tokens = torch.full(
                    (B, num_cb, max_target_len),
                    mask_id,
                    dtype=torch.long,
                    device=device,
                )

                target_lens: list[int] = []
                c_lens: list[int] = []
                step_plans: list[list[tuple[int, int, int]]] = []
                step_indices: list[int] = []
                for i, (state, _) in enumerate(state_items):
                    data = request_data[i]
                    input_ids = data["input_ids"]
                    audio_mask = data["audio_mask"]
                    attention_mask = data["attention_mask"]
                    tokens = data["tokens"]
                    seq_len = input_ids.shape[-1]
                    target_len = int(data["target_len"])

                    batch_input_ids[i, :, :seq_len] = input_ids[0]
                    batch_input_ids[B + i, :, :seq_len] = input_ids[1]
                    batch_audio_mask[i, :seq_len] = audio_mask[0]
                    batch_audio_mask[B + i, :seq_len] = audio_mask[1]
                    batch_attn_mask[i, :, :seq_len, :seq_len] = attention_mask[0]
                    batch_attn_mask[B + i, :, :seq_len, :seq_len] = attention_mask[1]
                    batch_tokens[i : i + 1, :, :target_len] = tokens[:, :, :target_len]
                    target_lens.append(target_len)
                    c_lens.append(int(data["c_len"]))
                    step_plans.append(data["step_plan"])
                    step_indices.append(int(state.step_index))

            self.generator.unmask_step(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attn_mask,
                tokens=batch_tokens,
                target_lens=target_lens,
                step_indices=step_indices,
                step_plans=step_plans,
                c_lens=c_lens,
                guidance_scale=self.guidance_scale,
                layer_penalty_factor=self.layer_penalty_factor,
                position_temperature=self.position_temperature,
                class_temperature=self.class_temperature,
                prof=prof,
            )

            for i, (state, _) in enumerate(state_items):
                data = request_data[i]
                seq_len = data["input_ids"].shape[-1]
                target_len = target_lens[i]
                step_index = step_indices[i]

                data["input_ids"][0].copy_(batch_input_ids[i, :, :seq_len])
                data["input_ids"][1].copy_(batch_input_ids[B + i, :, :seq_len])
                data["tokens"].copy_(batch_tokens[i : i + 1, :, :target_len])
                data["last_step_index"] = step_index

                step_plan = data["step_plan"]
                block_finished = (
                    step_index >= len(step_plan)
                    or self.generator.is_block_complete(step_plan, step_index)
                )
                state.step_index = step_index + 1
                state.sampling.step_index = state.step_index
                finished = state.step_index >= len(step_plan)

                should_emit = bool(data.get("stream_audio_blocks")) and block_finished
                if should_emit or finished:
                    chunk = self._decode_stream_delta(data, finished=finished)
                    if chunk.numel() > 0 or finished:
                        output_payload = (
                            {"audio": chunk, "sr": self.sample_rate}
                            if data.get("stream_audio_blocks")
                            else chunk
                        )
                        per_request_results[state.req_id] = DiffusionOutput(
                            output=output_payload,
                        )

                per_request_finished[state.req_id] = finished
                per_request_step_indices[state.req_id] = state.step_index

        first_req_id = state_items[0][0].req_id
        return RunnerOutput(
            req_id=first_req_id,
            step_index=per_request_step_indices.get(first_req_id),
            finished=all(per_request_finished.values()),
            result=None,
            per_request_results=per_request_results,
            per_request_finished=per_request_finished,
            per_request_step_indices=per_request_step_indices,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from model directory (not from the iterator).

        The diffusion model loader passes HF safetensors weights, but OmniVoice
        has custom weight names (llm.* → generator.*, audio_tokenizer.* → decoder.*).
        We load from model_path directly and return all param names to satisfy
        the loader's "all weights initialized" check.
        """
        # Consume the iterator (required by the loader contract)
        for _ in weights:
            pass

        device = self.device
        self.generator.load_weights(self.model_path, device)
        generator_dtype = _resolve_generator_dtype_from_env()
        if generator_dtype is None:
            self.generator = self.generator.to(device).eval()
        else:
            self.generator = self.generator.to(device=device, dtype=generator_dtype).eval()
            logger.info(
                "OmniVoice generator cast to %s via VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE.",
                generator_dtype,
            )
        self.decoder.load_weights(self.model_path, device)
        logger.info("OmniVoice pipeline loaded on %s", device)

        if os.environ.get("VLLM_OMNI_OMNIVOICE_PREWARM_BUCKETS") == "1":
            self._prewarm_bucket_shapes(device)

        # Return all parameter names to indicate they're initialized
        return {name for name, _ in self.named_parameters()}

    @torch.inference_mode()
    def _prewarm_bucket_shapes(self, device: torch.device) -> None:
        """Run synthetic generator forwards at each bucket size to fill the
        cudagraph cache before the first real request lands.

        With COMPILE_MODE=reduce-overhead, each new (B, S) shape pays a
        one-time graph-capture cost (~100-500 ms). In production this
        capture is paid by the first user request that hits each shape,
        which inflates cold-start latency. Pre-warming at startup pays
        the cost once during init so the first real request runs at
        steady-state speed.

        Pre-warms B=2 (c=1: cond+uncond after CFG split) at every bucket
        in `_FULL_SEQUENCE_BUCKETS`. Higher concurrencies still pay
        capture cost on their first request - extending this to cover
        more (B, S) combos is straightforward but trades startup time
        for first-request latency at those concurrencies.
        """
        # Inlined to avoid pulling in vllm.multimodal which doesn't
        # always resolve. Mirrors omnivoice.py:_FULL_SEQUENCE_BUCKETS.
        bucket_schedule = (640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664)
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id
        # The generator expects input_ids[2*B], with cond at [0:B] and
        # uncond at [B:2B] for CFG. For c=1 (one request) we want B_req=1
        # which means input_ids has dim-0 = 2.
        B_req = 1
        B_input = 2 * B_req

        logger.info(
            "Pre-warming cudagraph for %d bucket sizes at B_req=%d "
            "(VLLM_OMNI_OMNIVOICE_PREWARM_BUCKETS=1)",
            len(bucket_schedule), B_req,
        )
        warm_t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if warm_t0 is not None:
            warm_t0.record()

        for bucket_size in bucket_schedule:
            S = int(bucket_size)
            input_ids = torch.full(
                (B_input, num_cb, S), mask_id, dtype=torch.long, device=device,
            )
            audio_mask = torch.ones(B_input, S, dtype=torch.bool, device=device)
            attention_mask = torch.ones(
                B_input, 1, S, S, dtype=torch.bool, device=device,
            )
            target_lens = [S // 2] * B_req
            try:
                self.generator(
                    input_ids=input_ids,
                    audio_mask=audio_mask,
                    attention_mask=attention_mask,
                    target_lens=target_lens,
                    num_step=2,  # enough to capture the graph; not full quality
                    guidance_scale=0.0,
                    t_shift=self.t_shift,
                )
            except Exception:
                logger.warning(
                    "Pre-warm forward failed for bucket S=%d; "
                    "first real request at this shape will pay capture cost.",
                    S, exc_info=True,
                )

        if warm_t0 is not None:
            warm_t1 = torch.cuda.Event(enable_timing=True)
            warm_t1.record()
            torch.cuda.synchronize()
            elapsed_ms = warm_t0.elapsed_time(warm_t1)
            logger.info("Bucket pre-warm complete in %.0f ms", elapsed_ms)
