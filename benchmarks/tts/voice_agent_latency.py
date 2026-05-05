#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop voice-agent TTS serving benchmark.

This benchmark models live voice agents rather than open-loop request floods.
Each worker repeatedly sends one TTS request, waits for the response, then
idles for a randomized playback/user-turn delay before sending the next one.

Recommended OmniVoice server settings to sweep externally:

    VLLM_OMNI_DIFFUSION_CONCURRENT=1
    VLLM_OMNI_DIFFUSION_BATCH_SIZE=16
    VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS=10
    VLLM_OMNI_OMNIVOICE_OPT=1
    VLLM_OMNI_OMNIVOICE_COMPILE_MODE=max-autotune-no-cudagraphs
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import os
import random
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:
    import pynvml
except Exception:  # pragma: no cover - optional runtime dependency
    pynvml = None

try:
    from vllm_omni.model_executor.models.omnivoice.duration import (
        RuleDurationEstimator,
    )
except Exception:  # pragma: no cover - allows --help outside repo envs
    RuleDurationEstimator = None


DEFAULT_API = "http://127.0.0.1:8091/v1/audio/speech"
DEFAULT_MODEL = "k2-fsa/OmniVoice"
REF_TEXT = "Nice to meet you."
REF_TOKENS = 25.0
TOKENS_PER_SECOND = 25.0

WORDS = (
    "account payment balance appointment update confirmation transfer schedule "
    "policy reminder today tomorrow morning afternoon evening customer service "
    "secure verify information statement request option available process "
    "question support agent message address number recent status due amount "
    "arrangement document notice review completed pending automatic successful "
    "temporary change assistance callback followup expected details"
).split()

CLAUSES = (
    "I can help with that",
    "please confirm the account details",
    "your payment option is available",
    "we can review the recent activity",
    "the next step is ready",
    "I will summarize the current status",
    "please listen carefully to this update",
    "the system has recorded the request",
    "you can choose another option",
    "the confirmation should arrive shortly",
)


@dataclass
class RequestRecord:
    worker_id: int
    request_index: int
    included: bool
    start_s: float
    end_s: float
    latency_s: float
    ttfb_s: float | None
    status_code: int | None
    error: str | None
    target_audio_s: float
    estimated_audio_s: float
    actual_audio_s: float | None
    response_bytes: int
    rtf: float | None
    post_response_wait_s: float
    prompt_chars: int
    prompt_words: int


@dataclass
class InflightSample:
    t_s: float
    in_flight: int


@dataclass
class GPUSample:
    t_s: float
    power_w: float
    sm_pct: float
    mem_pct: float
    clock_mhz: float
    memory_used_mb: float


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def wav_duration_seconds(content: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(content), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        return None
    return None


class PromptFactory:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.estimator = RuleDurationEstimator() if RuleDurationEstimator else None

    def estimate_audio_s(self, text: str) -> float:
        if self.estimator is None:
            # Fallback approximation for --help or non-repo environments.
            return max(1.0, len(text) / 14.0)
        target_tokens = self.estimator.estimate_duration(
            text,
            REF_TEXT,
            REF_TOKENS,
        )
        return target_tokens / TOKENS_PER_SECOND

    def prompt_for_target(self, target_audio_s: float) -> tuple[str, float]:
        words: list[str] = []
        if target_audio_s >= 5.0 and self.rng.random() < 0.65:
            words.extend(self.rng.choice(CLAUSES).lower().split())

        # Grow text until it reaches the same duration estimator used by the
        # OmniVoice pipeline. Add one word at a time so short 2-4s utterances
        # do not get accidentally inflated by a full canned clause.
        while True:
            words.append(self.rng.choice(WORDS))
            text = self._punctuate(words)
            estimate = self.estimate_audio_s(text)
            if estimate >= target_audio_s:
                return text, estimate

    def _punctuate(self, words: list[str]) -> str:
        groups: list[str] = []
        i = 0
        while i < len(words):
            group_len = self.rng.randint(7, 12)
            groups.append(" ".join(words[i : i + group_len]))
            i += group_len
        text = ". ".join(group for group in groups if group).strip()
        if not text.endswith("."):
            text += "."
        return text[0].upper() + text[1:]


class RuntimeSamplers:
    def __init__(self, sample_interval_s: float, enable_gpu: bool, run_t0: float):
        self.sample_interval_s = sample_interval_s
        self.enable_gpu = enable_gpu and pynvml is not None
        self.run_t0 = run_t0
        self.in_flight = 0
        self.inflight_samples: list[InflightSample] = []
        self.gpu_samples: list[GPUSample] = []
        self._stop = asyncio.Event()
        self._nvml_handle = None

    async def __aenter__(self) -> "RuntimeSamplers":
        if self.enable_gpu:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self.enable_gpu:
            pynvml.nvmlShutdown()

    async def sample_loop(self) -> None:
        while not self._stop.is_set():
            t_s = time.perf_counter() - self.run_t0
            self.inflight_samples.append(InflightSample(t_s, self.in_flight))
            if self.enable_gpu and self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self.gpu_samples.append(
                        GPUSample(
                            t_s=t_s,
                            power_w=pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                            / 1000.0,
                            sm_pct=float(util.gpu),
                            mem_pct=float(util.memory),
                            clock_mhz=float(
                                pynvml.nvmlDeviceGetClockInfo(
                                    self._nvml_handle,
                                    pynvml.NVML_CLOCK_SM,
                                ),
                            ),
                            memory_used_mb=mem.used / (1024.0 * 1024.0),
                        ),
                    )
                except Exception:
                    pass
            try:
                await asyncio.wait_for(self._stop.wait(), self.sample_interval_s)
            except TimeoutError:
                pass


async def send_tts_request(
    client: httpx.AsyncClient,
    api: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    language: str | None,
    timeout_s: float,
) -> tuple[int, bytes, float | None]:
    ttfb_s: float | None = None
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "input": prompt,
        "voice": "default",
        "response_format": "wav",
    }
    if language:
        payload["language"] = language

    async with client.stream(
        "POST",
        api,
        json=payload,
        headers=headers,
        timeout=timeout_s,
    ) as response:
        chunks: list[bytes] = []
        async for chunk in response.aiter_bytes():
            if chunk and ttfb_s is None:
                ttfb_s = time.perf_counter() - t0
            chunks.append(chunk)
        if ttfb_s is None:
            ttfb_s = time.perf_counter() - t0
        return response.status_code, b"".join(chunks), ttfb_s


async def voice_worker(
    worker_id: int,
    args: argparse.Namespace,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    prompt_factory: PromptFactory,
    samplers: RuntimeSamplers,
    run_t0: float,
    measure_start_s: float,
    stop_at_s: float,
    records: list[RequestRecord],
) -> None:
    rng = random.Random(args.seed + 1009 * worker_id)
    request_index = 0

    if args.initial_stagger_s > 0:
        await asyncio.sleep(rng.uniform(0.0, args.initial_stagger_s))

    while (time.perf_counter() - run_t0) < stop_at_s:
        target_audio_s = rng.uniform(args.target_audio_min_s, args.target_audio_max_s)
        prompt, estimated_audio_s = prompt_factory.prompt_for_target(target_audio_s)
        post_wait_s = rng.uniform(args.playback_min_s, args.playback_max_s)

        start_s = time.perf_counter() - run_t0
        samplers.in_flight += 1
        status_code: int | None = None
        error: str | None = None
        response_bytes = 0
        actual_audio_s: float | None = None
        ttfb_s: float | None = None

        try:
            status_code, content, ttfb_s = await send_tts_request(
                client,
                args.api,
                headers,
                args.model,
                prompt,
                args.language,
                args.timeout_s,
            )
            response_bytes = len(content)
            if status_code == 200:
                actual_audio_s = wav_duration_seconds(content)
            else:
                error = content[:500].decode("utf-8", errors="replace")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            samplers.in_flight -= 1

        end_s = time.perf_counter() - run_t0
        latency_s = end_s - start_s
        rtf = latency_s / actual_audio_s if actual_audio_s and actual_audio_s > 0 else None
        records.append(
            RequestRecord(
                worker_id=worker_id,
                request_index=request_index,
                included=start_s >= measure_start_s,
                start_s=start_s,
                end_s=end_s,
                latency_s=latency_s,
                ttfb_s=ttfb_s,
                status_code=status_code,
                error=error,
                target_audio_s=target_audio_s,
                estimated_audio_s=estimated_audio_s,
                actual_audio_s=actual_audio_s,
                response_bytes=response_bytes,
                rtf=rtf,
                post_response_wait_s=post_wait_s,
                prompt_chars=len(prompt),
                prompt_words=len(prompt.split()),
            ),
        )
        request_index += 1

        if (time.perf_counter() - run_t0) + post_wait_s > stop_at_s:
            break
        await asyncio.sleep(post_wait_s)


def summarize(
    args: argparse.Namespace,
    worker_count: int,
    records: list[RequestRecord],
    samplers: RuntimeSamplers,
    measure_start_s: float,
    stop_at_s: float,
) -> dict[str, Any]:
    included = [r for r in records if r.included]
    successes = [r for r in included if r.status_code == 200 and r.error is None]
    errors = [r for r in included if r.status_code != 200 or r.error is not None]
    latencies = [r.latency_s for r in successes]
    ttfbs = [r.ttfb_s for r in successes if r.ttfb_s is not None]
    rtfs = [r.rtf for r in successes if r.rtf is not None]
    actual_audio = [r.actual_audio_s for r in successes if r.actual_audio_s is not None]
    estimated_audio = [r.estimated_audio_s for r in successes]
    target_audio = [r.target_audio_s for r in successes]

    measured_duration_s = max(0.001, stop_at_s - measure_start_s)
    total_audio_s = sum(actual_audio)
    inflight_window = [
        s.in_flight
        for s in samplers.inflight_samples
        if measure_start_s <= s.t_s <= stop_at_s
    ]
    gpu_window = [
        s for s in samplers.gpu_samples if measure_start_s <= s.t_s <= stop_at_s
    ]

    summary: dict[str, Any] = {
        "label": args.label,
        "workers": worker_count,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "completed": len(included),
        "successes": len(successes),
        "errors": len(errors),
        "error_rate": len(errors) / len(included) if included else 0.0,
        "req_per_s": len(successes) / measured_duration_s,
        "audio_s_per_wall_s": total_audio_s / measured_duration_s,
        "latency_mean_s": safe_mean(latencies),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p90_s": percentile(latencies, 90),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "ttfb_p50_s": percentile(ttfbs, 50),
        "ttfb_p95_s": percentile(ttfbs, 95),
        "rtf_mean": safe_mean(rtfs),
        "rtf_p50": percentile(rtfs, 50),
        "rtf_p95": percentile(rtfs, 95),
        "rtf_p99": percentile(rtfs, 99),
        "target_audio_mean_s": safe_mean(target_audio),
        "target_audio_p95_s": percentile(target_audio, 95),
        "estimated_audio_mean_s": safe_mean(estimated_audio),
        "actual_audio_mean_s": safe_mean(actual_audio),
        "actual_audio_p95_s": percentile(actual_audio, 95),
        "inflight_mean": safe_mean([float(v) for v in inflight_window]),
        "inflight_p95": percentile([float(v) for v in inflight_window], 95),
        "inflight_max": max(inflight_window) if inflight_window else 0,
        "gpu_power_w_mean": safe_mean([s.power_w for s in gpu_window]),
        "gpu_sm_pct_mean": safe_mean([s.sm_pct for s in gpu_window]),
        "gpu_mem_pct_mean": safe_mean([s.mem_pct for s in gpu_window]),
        "gpu_clock_mhz_mean": safe_mean([s.clock_mhz for s in gpu_window]),
        "gpu_clock_mhz_min": min((s.clock_mhz for s in gpu_window), default=float("nan")),
        "gpu_clock_mhz_max": max((s.clock_mhz for s in gpu_window), default=float("nan")),
        "expected_dtype": args.expected_dtype,
        "expected_compile_mode": args.expected_compile_mode,
        "expected_batch_size": args.expected_batch_size,
        "expected_batch_wait_ms": args.expected_batch_wait_ms,
        "env_VLLM_OMNI_DIFFUSION_CONCURRENT": os.environ.get(
            "VLLM_OMNI_DIFFUSION_CONCURRENT",
        ),
        "env_VLLM_OMNI_DIFFUSION_BATCH_SIZE": os.environ.get(
            "VLLM_OMNI_DIFFUSION_BATCH_SIZE",
        ),
        "env_VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS": os.environ.get(
            "VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS",
        ),
        "env_VLLM_OMNI_DIFFUSION_BATCH_STRATEGY": os.environ.get(
            "VLLM_OMNI_DIFFUSION_BATCH_STRATEGY",
        ),
        "env_VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS": os.environ.get(
            "VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS",
        ),
        "env_VLLM_OMNI_OMNIVOICE_OPT": os.environ.get(
            "VLLM_OMNI_OMNIVOICE_OPT",
        ),
        "env_VLLM_OMNI_OMNIVOICE_COMPILE_MODE": os.environ.get(
            "VLLM_OMNI_OMNIVOICE_COMPILE_MODE",
        ),
        "env_VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE": os.environ.get(
            "VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE",
        ),
    }
    return summary


def write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                row = asdict(row)
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def run_one_worker_count(
    args: argparse.Namespace,
    worker_count: int,
    out_dir: Path,
) -> dict[str, Any]:
    prompt_factory = PromptFactory(args.seed)
    run_t0 = time.perf_counter()
    measure_start_s = args.warmup_s
    stop_at_s = args.warmup_s + args.duration_s
    records: list[RequestRecord] = []

    limits = httpx.Limits(
        max_connections=max(worker_count * 2, 16),
        max_keepalive_connections=max(worker_count, 16),
    )
    timeout = httpx.Timeout(args.timeout_s)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        }
        # Health check with a short timeout before starting many workers.
        health_url = args.api.split("/v1/audio/speech", 1)[0].rstrip("/") + "/health"
        health = await client.get(health_url, timeout=min(5.0, args.timeout_s))
        health.raise_for_status()

        async with RuntimeSamplers(
            args.sample_interval_s,
            args.gpu_samples,
            run_t0,
        ) as samplers:
            sample_task = asyncio.create_task(samplers.sample_loop())
            tasks = [
                asyncio.create_task(
                    voice_worker(
                        worker_id=i,
                        args=args,
                        client=client,
                        headers=headers,
                        prompt_factory=prompt_factory,
                        samplers=samplers,
                        run_t0=run_t0,
                        measure_start_s=measure_start_s,
                        stop_at_s=stop_at_s,
                        records=records,
                    ),
                )
                for i in range(worker_count)
            ]
            await asyncio.gather(*tasks)
            samplers._stop.set()
            await sample_task

            summary = summarize(
                args,
                worker_count,
                records,
                samplers,
                measure_start_s,
                stop_at_s,
            )
            prefix = f"workers_{worker_count}"
            write_jsonl(out_dir / f"{prefix}_requests.jsonl", records)
            write_jsonl(out_dir / f"{prefix}_inflight.jsonl", samplers.inflight_samples)
            if samplers.gpu_samples:
                write_jsonl(out_dir / f"{prefix}_gpu.jsonl", samplers.gpu_samples)
            with (out_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            return summary


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    print()
    print("| workers | req/s | audio/wall | p50 lat | p95 lat | p99 lat | p95 RTF | in-flight p95/max | GPU SM | GPU W |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s['workers']} | {s['req_per_s']:.2f} | "
            f"{s['audio_s_per_wall_s']:.2f}x | "
            f"{s['latency_p50_s']:.3f}s | {s['latency_p95_s']:.3f}s | "
            f"{s['latency_p99_s']:.3f}s | {s['rtf_p95']:.3f} | "
            f"{s['inflight_p95']:.1f}/{s['inflight_max']} | "
            f"{s['gpu_sm_pct_mean']:.1f}% | {s['gpu_power_w_mean']:.0f}W |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=None)
    parser.add_argument("--workers", type=parse_csv_ints, default=[1, 4, 8, 16, 32, 64, 80])
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--warmup-s", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--target-audio-min-s", type=float, default=2.0)
    parser.add_argument("--target-audio-max-s", type=float, default=20.0)
    parser.add_argument("--playback-min-s", type=float, default=1.0)
    parser.add_argument("--playback-max-s", type=float, default=19.0)
    parser.add_argument(
        "--initial-stagger-s",
        type=float,
        default=19.0,
        help="Random initial worker delay. Set 0 for a cold burst.",
    )
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--label", default="voice-agent")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gpu-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-dtype", default="bfloat16")
    parser.add_argument("--expected-compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--expected-batch-size", type=int, default=None)
    parser.add_argument("--expected-batch-wait-ms", type=float, default=None)
    parser.add_argument(
        "--dry-run-prompts",
        type=int,
        default=0,
        help="Print generated prompt examples and exit without contacting the server.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if args.target_audio_min_s <= 0 or args.target_audio_max_s < args.target_audio_min_s:
        raise ValueError("invalid target audio duration range")
    if args.playback_min_s < 0 or args.playback_max_s < args.playback_min_s:
        raise ValueError("invalid playback delay range")

    if args.dry_run_prompts:
        factory = PromptFactory(args.seed)
        rng = random.Random(args.seed)
        for i in range(args.dry_run_prompts):
            target = rng.uniform(args.target_audio_min_s, args.target_audio_max_s)
            prompt, estimate = factory.prompt_for_target(target)
            print(
                json.dumps(
                    {
                        "i": i,
                        "target_audio_s": target,
                        "estimated_audio_s": estimate,
                        "chars": len(prompt),
                        "words": len(prompt.split()),
                        "prompt": prompt,
                    },
                ),
            )
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"outputs/voice_agent_latency_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["workers"] = args.workers
    config["started_utc"] = stamp
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"Writing results to {out_dir}")
    summaries: list[dict[str, Any]] = []
    for worker_count in args.workers:
        print(f"\nRunning workers={worker_count} for {args.duration_s:.1f}s after {args.warmup_s:.1f}s warmup")
        summary = await run_one_worker_count(args, worker_count, out_dir)
        summaries.append(summary)
        print_summary_table([summary])

    write_csv(out_dir / "summary.csv", summaries)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    print_summary_table(summaries)
    print(f"\nSUMMARY_CSV={out_dir / 'summary.csv'}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
