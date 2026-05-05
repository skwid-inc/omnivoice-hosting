"""Aggregate Phase A sweep CSVs into a side-by-side comparison.

Reads every `outputs/phase_a/<label>_*/sweep/concurrency_sweep.csv` plus
the F12 baseline at outputs/baseline_rerun_seedhook_*/concurrency_sweep.csv
and prints two tables: req/s by (config, c) and wall_avg by (config, c).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def load_csv(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            c = int(r["concurrency"])
            rows[c] = {k: float(v) for k, v in r.items() if k != "concurrency"}
    return rows


def find_latest(glob: str, root: Path = Path(".")) -> Path | None:
    matches = sorted(root.glob(glob))
    return matches[-1] if matches else None


def main() -> int:
    repo = Path(".")
    configs: dict[str, Path] = {}

    baseline = find_latest("outputs/baseline_rerun_seedhook_*")
    if baseline is not None:
        c = baseline / "concurrency_sweep.csv"
        if c.exists():
            configs["F12_baseline"] = c

    for d in sorted(repo.glob("outputs/phase_a/bs*_*")):
        c = d / "sweep" / "concurrency_sweep.csv"
        if c.exists():
            label = d.name.split("_")[0]  # bs0, bs32, ...
            # If there are multiple runs at the same label, prefer the latest dir.
            configs[label] = c

    if not configs:
        print("No CSVs found.")
        return 1

    data = {label: load_csv(p) for label, p in configs.items()}
    cs = sorted({c for d in data.values() for c in d.keys()})

    def table(metric: str, fmt: str, title: str) -> None:
        print(f"\n## {title}\n")
        header = "| c   | " + " | ".join(f"{l:>14}" for l in data.keys()) + " |"
        sep = "|" + "|".join(["-" * (len(seg) + 2) for seg in header.split("|")[1:-1]]) + "|"
        print(header)
        print(sep)
        for c in cs:
            cells = []
            for label, rows in data.items():
                if c in rows:
                    cells.append(f"{rows[c][metric]:>{fmt}}")
                else:
                    cells.append(f"{'-':>14}")
            print(f"| {c:<3} | " + " | ".join(cells) + " |")

    table("req_per_sec", "14.3f", "req/s by config and concurrency")
    table("wall_avg", "14.3f", "wall_avg (s) by config and concurrency")
    table("wall_p95", "14.3f", "wall_p95 (s) by config and concurrency")

    print("\n## delta vs F12_baseline (relative req/s)\n")
    if "F12_baseline" not in data:
        print("(no F12_baseline available; skipping deltas)")
        return 0
    base = data["F12_baseline"]
    header = "| c   | " + " | ".join(f"{l:>14}" for l in data.keys() if l != "F12_baseline") + " |"
    print(header)
    sep = "|" + "|".join(["-" * (len(seg) + 2) for seg in header.split("|")[1:-1]]) + "|"
    print(sep)
    for c in cs:
        cells = []
        for label in data.keys():
            if label == "F12_baseline":
                continue
            if c in data[label] and c in base:
                ratio = data[label][c]["req_per_sec"] / base[c]["req_per_sec"]
                cells.append(f"{(ratio - 1.0) * 100:>13.1f}%")
            else:
                cells.append(f"{'-':>14}")
        print(f"| {c:<3} | " + " | ".join(cells) + " |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
