"""A/B audio-quality test for OmniVoice config changes.

Generates 5 fixed prompts with a fixed sampling `seed` so two runs made
under different server configs (e.g. F12 baseline vs Fast-dLLM confidence
decoding) can be blind-compared by ear.

OmniVoice's `voice="default"` is auto-voice mode where the model picks
a latent voice per call — without seeding, every call produces a
different speaker, which makes A/B impossible. This script pins
`seed` per-request so the gumbel sampling RNG starts from the same
state for every prompt, yielding a consistent voice within a run AND
across runs that share the same seed.

Voice cloning via `ref_audio` is the proper fix and the path the
production code expects, but the audio tokenizer
(`HiggsAudioV2TokenizerModel`) requires `transformers>=5.3.0`, which
is not present in the F12 venv (`4.57.6`). The seed approach gets us
a stable voice without an env upgrade — and is what the F12 baseline
itself was measured under, so it's the apples-to-apples comparison
for the experiments queued on this branch.

Two subcommands:

    generate   Hit one server, save 5 WAVs + manifest.json + summary.csv
    compare    Diff two `generate` directories, write side-by-side
               index.html with randomized A/B players + delta metrics

Usage:

    # 1. Run against the F12 baseline (port 8091)
    .venv/bin/python benchmarks/tts/ab_quality.py generate \\
        --api-base http://127.0.0.1:8091 \\
        --label baseline \\
        --out_dir outputs/ab/baseline

    # 2. Restart the server with experiment env vars, then:
    .venv/bin/python benchmarks/tts/ab_quality.py generate \\
        --api-base http://127.0.0.1:8091 \\
        --label fastdllm_thr09_sb2 \\
        --out_dir outputs/ab/fastdllm_thr09_sb2

    # 3. Compare. Open out_dir/index.html in a browser and listen.
    .venv/bin/python benchmarks/tts/ab_quality.py compare \\
        outputs/ab/baseline \\
        outputs/ab/fastdllm_thr09_sb2 \\
        --out_dir outputs/ab/cmp_baseline_vs_fastdllm
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

VOICE = "default"

# Fixed sampling seed for the entire A/B harness. Same seed within a run
# (across the 5 prompts) gives the same voice; same seed across runs
# (baseline vs experiment) gives the same voice for a fair comparison.
# Both sides of the A/B MUST share this value.
DEFAULT_SEED = 42

# Five prompts cover the audio-quality failure modes that matter for
# OmniVoice: prosody on names, digit pronunciation, question intonation,
# typical narrative, and a long subordinate-clause sentence (where
# step-skipping or KV-cache drift tends to break first).
FIXED_PROMPTS: list[dict[str, str]] = [
    {
        "id": "01_greeting",
        "text": "Hello, my name is Taylor and I'm calling from Salient.",
    },
    {
        "id": "02_numbers",
        "text": (
            "Your account number is four eight seven, two zero three, "
            "six one nine, and the balance due is nine hundred and "
            "forty two dollars."
        ),
    },
    {
        "id": "03_question",
        "text": (
            "Are you absolutely sure that's the right answer? "
            "Because if it isn't, we may have a problem."
        ),
    },
    {
        "id": "04_narrative",
        "text": (
            "The quick brown fox jumps over the lazy dog every single "
            "morning before breakfast, rain or shine."
        ),
    },
    {
        "id": "05_complex",
        "text": (
            "In a small village nestled between two mountains, there "
            "lived an old clockmaker who could fix any timepiece, no "
            "matter how broken, ancient, or seemingly beyond repair."
        ),
    },
]


def synth_one(
    client: httpx.Client,
    api_url: str,
    headers: dict[str, str],
    model: str,
    text: str,
    language: str | None,
    seed: int | None,
) -> tuple[np.ndarray, int, float, bytes]:
    """POST one /v1/audio/speech request. Returns (audio, sr, wall_sec, raw_bytes)."""
    payload: dict = {
        "model": model,
        "input": text,
        "voice": VOICE,
        "response_format": "wav",
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if language:
        payload["language"] = language

    t0 = time.perf_counter()
    resp = client.post(api_url, json=payload, headers=headers)
    wall = time.perf_counter() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr), wall, resp.content


def audio_metrics(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Cheap signal stats for sanity-checking obvious regressions."""
    if audio.size == 0:
        return {
            "audio_sec": 0.0, "rms": 0.0, "peak": 0.0, "spectral_centroid_hz": 0.0,
        }
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))

    n = min(len(audio), 1 << 15)
    spec = np.abs(np.fft.rfft(audio[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    centroid = (
        float((freqs * spec).sum() / spec.sum()) if spec.sum() > 0 else 0.0
    )
    return {
        "audio_sec": float(len(audio) / sr),
        "rms": rms,
        "peak": peak,
        "spectral_centroid_hz": centroid,
    }


def cmd_generate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"{args.api_base.rstrip('/')}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    seed = int(args.seed)
    logging.info(
        "voice locked via seed=%d (auto-voice mode, no ref_audio)", seed,
    )

    manifest: dict = {
        "label": args.label,
        "api_base": args.api_base,
        "model": args.model,
        "voice": VOICE,
        "seed": seed,
        "language": args.language,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host_info": {"argv": sys.argv},
        "prompts": [],
    }

    summary_rows = []

    with httpx.Client(timeout=args.http_timeout) as client:
        if args.warmup:
            logging.info("warmup: %s", FIXED_PROMPTS[0]["id"])
            try:
                synth_one(
                    client, api_url, headers, args.model,
                    FIXED_PROMPTS[0]["text"], args.language, seed,
                )
            except Exception as exc:
                logging.warning("warmup failed: %s (continuing)", exc)

        for p in FIXED_PROMPTS:
            logging.info("generate %s (%d chars)", p["id"], len(p["text"]))
            audio, sr, wall, raw = synth_one(
                client, api_url, headers, args.model, p["text"], args.language,
                seed,
            )
            wav_name = f"{p['id']}.wav"
            (out_dir / wav_name).write_bytes(raw)

            metrics = audio_metrics(audio, sr)
            manifest["prompts"].append({
                "id": p["id"],
                "text": p["text"],
                "wav": wav_name,
                "wall_sec": wall,
                "sample_rate": sr,
                **metrics,
            })
            summary_rows.append({
                "id": p["id"],
                "wall_sec": round(wall, 4),
                "audio_sec": round(metrics["audio_sec"], 4),
                "rtf": round(wall / max(metrics["audio_sec"], 1e-6), 4),
                "rms": round(metrics["rms"], 5),
                "peak": round(metrics["peak"], 5),
                "centroid_hz": round(metrics["spectral_centroid_hz"], 1),
            })

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"\nGENERATE  label={args.label}  out_dir={out_dir}")
    print(f"{'id':<14}{'wall':>9}{'audio':>9}{'rtf':>8}{'rms':>9}{'peak':>9}{'centroid':>11}")
    for r in summary_rows:
        print(
            f"{r['id']:<14}"
            f"{r['wall_sec']:>9.3f}{r['audio_sec']:>9.3f}{r['rtf']:>8.3f}"
            f"{r['rms']:>9.4f}{r['peak']:>9.4f}{r['centroid_hz']:>11.1f}"
        )
    return 0


def _load_manifest(d: Path) -> dict:
    f = d / "manifest.json"
    if not f.is_file():
        raise FileNotFoundError(f"missing manifest.json in {d}")
    return json.loads(f.read_text())


def cmd_compare(args: argparse.Namespace) -> int:
    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man_a = _load_manifest(dir_a)
    man_b = _load_manifest(dir_b)
    label_a = man_a.get("label", dir_a.name)
    label_b = man_b.get("label", dir_b.name)

    by_id_a = {p["id"]: p for p in man_a["prompts"]}
    by_id_b = {p["id"]: p for p in man_b["prompts"]}
    common = [p["id"] for p in FIXED_PROMPTS if p["id"] in by_id_a and p["id"] in by_id_b]
    if not common:
        raise SystemExit("no overlapping prompt ids between the two runs")

    rng = random.Random(args.seed)

    delta_rows = []
    items: list[dict] = []

    for pid in common:
        a = by_id_a[pid]
        b = by_id_b[pid]
        len_ratio = a["audio_sec"] and b["audio_sec"] / a["audio_sec"]
        rms_ratio = a["rms"] and b["rms"] / max(a["rms"], 1e-9)
        cent_delta = b["spectral_centroid_hz"] - a["spectral_centroid_hz"]
        wall_speedup = a["wall_sec"] / max(b["wall_sec"], 1e-6)

        delta_rows.append({
            "id": pid,
            "wall_a_sec": round(a["wall_sec"], 4),
            "wall_b_sec": round(b["wall_sec"], 4),
            "wall_speedup_b_over_a": round(wall_speedup, 3),
            "audio_a_sec": round(a["audio_sec"], 3),
            "audio_b_sec": round(b["audio_sec"], 3),
            "len_ratio_b_over_a": round(len_ratio, 4),
            "rms_a": round(a["rms"], 5),
            "rms_b": round(b["rms"], 5),
            "rms_ratio_b_over_a": round(rms_ratio, 4),
            "centroid_a_hz": round(a["spectral_centroid_hz"], 1),
            "centroid_b_hz": round(b["spectral_centroid_hz"], 1),
            "centroid_delta_hz": round(cent_delta, 1),
        })

        # Randomize X/Y assignment per prompt so the listener can't
        # tell which side is the new config without the reveal table.
        flip = bool(rng.getrandbits(1))
        x_label, y_label = ("A", "B") if not flip else ("B", "A")
        # Embed audio as base64 data URLs. file:// resources loaded by a
        # file:// host page render unreliably (some browsers flicker the
        # audio control while retrying the cross-protocol fetch); inline
        # bytes Just Work and make the report self-contained.
        wav_a = (dir_a / a["wav"]).resolve()
        wav_b = (dir_b / b["wav"]).resolve()
        url_a = "data:audio/wav;base64," + base64.b64encode(wav_a.read_bytes()).decode("ascii")
        url_b = "data:audio/wav;base64," + base64.b64encode(wav_b.read_bytes()).decode("ascii")
        x_src, y_src = (url_a, url_b) if not flip else (url_b, url_a)
        items.append({
            "id": pid,
            "text": next(p["text"] for p in FIXED_PROMPTS if p["id"] == pid),
            "x_src": x_src,
            "y_src": y_src,
            "x_is": x_label,
            "y_is": y_label,
        })

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        w.writeheader()
        w.writerows(delta_rows)

    html = _render_html(label_a, label_b, items, delta_rows)
    (out_dir / "index.html").write_text(html)

    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "dir_a": str(dir_a.resolve()),
        "dir_b": str(dir_b.resolve()),
        "n_prompts": len(common),
        "deltas": delta_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nCOMPARE  A={label_a}   B={label_b}")
    hdr = (
        f"{'id':<14}{'wall_A':>9}{'wall_B':>9}{'sup':>7}"
        f"{'len_r':>8}{'rms_r':>8}{'dCent':>9}"
    )
    print(hdr)
    for r in delta_rows:
        print(
            f"{r['id']:<14}"
            f"{r['wall_a_sec']:>9.3f}{r['wall_b_sec']:>9.3f}"
            f"{r['wall_speedup_b_over_a']:>7.2f}"
            f"{r['len_ratio_b_over_a']:>8.3f}"
            f"{r['rms_ratio_b_over_a']:>8.3f}"
            f"{r['centroid_delta_hz']:>9.1f}"
        )
    print(f"\nlisten:  open {out_dir / 'index.html'}")
    return 0


def _render_html(
    label_a: str, label_b: str, items: list[dict], deltas: list[dict],
) -> str:
    rows_html = []
    for it in items:
        rows_html.append(f"""
        <section class="prompt">
          <h3>{it['id']}</h3>
          <p class="text">"{it['text']}"</p>
          <div class="players">
            <div>
              <div class="lab">X</div>
              <audio controls preload="auto" src="file://{it['x_src']}"></audio>
            </div>
            <div>
              <div class="lab">Y</div>
              <audio controls preload="auto" src="file://{it['y_src']}"></audio>
            </div>
          </div>
          <details>
            <summary>reveal mapping</summary>
            <p>X = <b>{label_a if it['x_is'] == 'A' else label_b}</b>,
               Y = <b>{label_a if it['y_is'] == 'A' else label_b}</b></p>
          </details>
        </section>""")

    delta_table = ["<table><thead><tr>"]
    cols = list(deltas[0].keys())
    delta_table.append("".join(f"<th>{c}</th>" for c in cols))
    delta_table.append("</tr></thead><tbody>")
    for r in deltas:
        delta_table.append(
            "<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>"
        )
    delta_table.append("</tbody></table>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>OmniVoice A/B: {label_a} vs {label_b}</title>
<meta name="color-scheme" content="light only">
<style>
  :root {{ color-scheme: light; }}
  html, body {{ background: #ffffff !important; color: #222 !important; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          max-width: 980px; margin: 32px auto; padding: 0 16px; }}
  h1 {{ margin-bottom: 4px; color: #111; }} h2 {{ margin-top: 32px; color: #111; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 24px; }}
  .prompt {{ border: 1px solid #ddd; border-radius: 8px;
             padding: 16px; margin: 16px 0; background: #fafafa; color: #222; }}
  .prompt h3 {{ margin: 0 0 8px; font-size: 15px; color: #555; }}
  .text {{ font-style: italic; color: #333; margin: 8px 0 16px; }}
  .players {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .lab {{ font-weight: 700; font-size: 18px; color: #888; margin-bottom: 4px; }}
  audio {{ width: 100%; }}
  details {{ margin-top: 12px; font-size: 13px; color: #333; }}
  details summary {{ cursor: pointer; color: #0a64f0; }}
  table {{ border-collapse: collapse; font-size: 12px; margin-top: 12px; color: #222; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:first-child, th:first-child {{ text-align: left; }}
</style></head><body>
<h1>OmniVoice A/B</h1>
<p class="meta"><b>A</b> = {label_a} &nbsp;&nbsp;|&nbsp;&nbsp; <b>B</b> = {label_b}
&nbsp;&nbsp;|&nbsp;&nbsp; players X / Y are randomized per prompt; expand
"reveal mapping" only after you've decided which one sounds better.</p>

<h2>Listen</h2>
{"".join(rows_html)}

<h2>Automated metrics (sanity only — listening is the real test)</h2>
<p class="meta">
  <code>wall_speedup_b_over_a</code>: B faster than A means &gt; 1.<br>
  <code>len_ratio_b_over_a</code>: ≈ 1.0 means audio length unchanged
  (drift &gt; ±10% suggests the model is generating different content).<br>
  <code>rms_ratio_b_over_a</code>: ≈ 1.0 means loudness preserved.<br>
  <code>centroid_delta_hz</code>: shift in mean spectral centroid; large
  shifts often correlate with audible timbre changes.
</p>
{"".join(delta_table)}
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Run 5 fixed prompts and save WAVs")
    g.add_argument("--api-base", default="http://127.0.0.1:8091",
                   help="bare host (no /v1)")
    g.add_argument("--api-key", default="EMPTY")
    g.add_argument("--model", default="k2-fsa/OmniVoice")
    g.add_argument("--label", required=True,
                   help="short tag for this run, e.g. baseline / fastdllm_sb2")
    g.add_argument("--language", default=None)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--http_timeout", type=float, default=120.0)
    g.add_argument("--warmup", action="store_true", default=True,
                   help="(default on) fire one warmup request before measurement")
    g.add_argument("--no-warmup", action="store_false", dest="warmup")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=("Sampling seed; locks the voice. Both sides of an "
                         "A/B MUST use the same value (default %(default)s)."))

    c = sub.add_parser("compare", help="Side-by-side HTML + delta CSV")
    c.add_argument("dir_a", help="generate output dir for run A (e.g. baseline)")
    c.add_argument("dir_b", help="generate output dir for run B (experiment)")
    c.add_argument("--out_dir", required=True)
    c.add_argument("--seed", type=int, default=1234,
                   help="RNG seed for X/Y player randomization")

    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)

    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
