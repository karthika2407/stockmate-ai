"""Run every benchmark and write a manifest of what produced which file.

The report quotes no number that is not in results/, and this script is how
results/ is produced.  It records the command, the exit status and the wall
clock for each benchmark, so a reviewer can tell at a glance whether a figure
came from a full run or a truncated one.

Run:
    python scripts/run_benchmarks.py            (full, takes a while)
    python scripts/run_benchmarks.py --quick    (smaller, for a smoke test)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FULL = [
    ("forecast_tuning", ["benchmarks/bench_tuning.py"]),
    ("forecast", ["benchmarks/bench_forecast.py", "--skus", "300", "--origins", "8"]),
    ("audit", ["benchmarks/bench_audit.py", "--splits", "test", "validation", "train"]),
    ("price_creep", ["benchmarks/bench_price_creep.py", "--skus", "400"]),
    ("replenishment", ["benchmarks/bench_replenishment.py", "--skus", "150"]),
    ("gst_rules", ["benchmarks/bench_gst_rules.py"]),
    ("extraction", ["benchmarks/bench_extraction.py", "--limit", "100"]),
]

QUICK = [
    ("forecast", ["benchmarks/bench_forecast.py", "--skus", "80", "--origins", "3"]),
    ("audit", ["benchmarks/bench_audit.py", "--splits", "test"]),
    ("price_creep", ["benchmarks/bench_price_creep.py", "--skus", "120"]),
    ("replenishment", ["benchmarks/bench_replenishment.py", "--skus", "40"]),
    ("gst_rules", ["benchmarks/bench_gst_rules.py"]),
    ("extraction", ["benchmarks/bench_extraction.py", "--limit", "20"]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these benchmark names")
    args = ap.parse_args()

    plan = QUICK if args.quick else FULL
    if args.only:
        plan = [(n, c) for n, c in plan if n in args.only]

    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "quick" if args.quick else "full",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "runs": [],
    }

    for name, cmd in plan:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        elapsed = time.time() - t0
        manifest["runs"].append({
            "name": name,
            "command": " ".join(["python", *cmd]),
            "returncode": proc.returncode,
            "seconds": round(elapsed, 1),
            "ok": proc.returncode == 0,
        })
        print(f"-- {name}: exit {proc.returncode} in {elapsed:.0f}s")

    out = ROOT / "results" / "MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    failed = [r["name"] for r in manifest["runs"] if not r["ok"]]
    print(f"\nmanifest: {out}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all benchmarks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
