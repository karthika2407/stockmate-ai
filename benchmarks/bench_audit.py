"""Benchmark: can the verifier find a fault that was put there on purpose?

Protocol
--------
1. Load every annotated CORD-v2 receipt and build a reference bill from the
   annotation, so extraction error plays no part in this measurement.
2. Split the receipts into those whose own arithmetic closes under at least one
   standard reading and those that do not.  The second group is reported but
   never perturbed: a detector that fires on a receipt which was already
   inconsistent has told us nothing.
3. For each clean receipt and each fault kind, inject one labelled fault whose
   size and location are known, and record whether the checks fired, whether
   they fired on the right line, and whether the rupee amount they reported
   matches the amount actually injected.
4. Report the false-alarm rate separately, on the clean receipts with nothing
   injected.  Recall without that number means nothing.

Run:
    python benchmarks/bench_audit.py --splits test validation
Writes results/audit_detection.json and results/audit_per_case.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockmate.datasets.perturb import FAULT_KINDS, inject_fault, seeded_random  # noqa: E402
from stockmate.datasets.receipts import load_cord  # noqa: E402
from stockmate.verify.suite import verify_bill  # noqa: E402

#: Rule tiers scored separately, so the table shows what each one is worth.
ABLATIONS = {
    "arithmetic_only": ("arithmetic",),
    "full_verifier": ("arithmetic", "duplicate"),
}


def failed_checks(bill, tiers=("arithmetic", "duplicate")):
    return verify_bill(bill).failures_in(tiers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT.parent / "_datasets" / "cord_v2"))
    ap.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    t0 = time.time()
    _, bills, stats = load_cord(
        args.data, splits=tuple(args.splits), limit=args.limit, with_images=False
    )
    load_s = time.time() - t0
    print(f"{stats.receipts} receipts, {stats.line_count} lines  ({load_s:.1f}s)")

    clean, inconsistent = [], []
    for b in bills:
        (clean if not failed_checks(b) else inconsistent).append(b)
    print(f"  arithmetic closes on {len(clean)}; "
          f"{len(inconsistent)} are inconsistent as issued and are not perturbed")

    # False alarms: the clean receipts, untouched.  By construction this should
    # be zero, and it is asserted rather than assumed.
    false_alarms = sum(1 for b in clean if failed_checks(b))

    rows: list[dict] = []
    per_kind: dict[str, dict] = defaultdict(
        lambda: {"applicable": 0, "detected": 0, "localised": 0,
                 "amount_exact": 0, "injected_paise": 0, "reported_paise": 0}
    )
    ablation: dict[str, dict[str, dict]] = {
        name: defaultdict(lambda: {"applicable": 0, "detected": 0}) for name in ABLATIONS
    }

    t0 = time.time()
    for bill in clean:
        for kind in FAULT_KINDS:
            made = inject_fault(bill, kind, rng=seeded_random(bill.bill_id, kind.value))
            if made is None:
                continue                      # receipt cannot carry this fault
            dirty, fault = made
            report = verify_bill(dirty)
            for name, tiers in ABLATIONS.items():
                cell = ablation[name][kind.value]
                cell["applicable"] += 1
                cell["detected"] += int(bool(report.failures_in(tiers)))

            fails = report.failures_in(ABLATIONS["full_verifier"])
            detected = bool(fails)

            lines_flagged = {c.line_index for c in fails if c.line_index is not None}
            localised = (
                fault.line_index in lines_flagged
                if (detected and fault.line_index is not None)
                else None
            )

            # The largest rupee difference any failed check reports is what the
            # owner would be shown as the amount at stake.
            reported = max((abs(c.delta.paise) for c in fails if c.delta), default=0)
            amount_exact = detected and abs(reported - abs(fault.amount.paise)) <= 100

            s = per_kind[kind.value]
            s["applicable"] += 1
            s["detected"] += int(detected)
            s["localised"] += int(bool(localised))
            s["amount_exact"] += int(amount_exact)
            s["injected_paise"] += abs(fault.amount.paise)
            s["reported_paise"] += reported if detected else 0

            rows.append({
                "bill_id": bill.bill_id,
                "fault_kind": kind.value,
                "fault_line": fault.line_index,
                "injected_paise": abs(fault.amount.paise),
                "detected": int(detected),
                "localised": "" if localised is None else int(localised),
                "reported_paise": reported,
                "amount_exact": int(amount_exact),
                "rules_fired": "|".join(sorted({c.rule_id for c in fails})),
            })
    bench_s = time.time() - t0

    summary = {}
    for kind, s in per_kind.items():
        n = s["applicable"]
        line_cases = sum(1 for r in rows if r["fault_kind"] == kind and r["localised"] != "")
        summary[kind] = {
            "applicable_cases": n,
            "detected": s["detected"],
            "recall": round(s["detected"] / n, 4) if n else None,
            "localisation_rate": round(s["localised"] / line_cases, 4) if line_cases else None,
            "amount_exact_rate": round(s["amount_exact"] / n, 4) if n else None,
            "injected_total_rupees": round(s["injected_paise"] / 100, 2),
            "reported_total_rupees": round(s["reported_paise"] / 100, 2),
        }

    total_cases = sum(s["applicable_cases"] for s in summary.values())
    total_detected = sum(s["detected"] for s in summary.values())

    payload = {
        "protocol": (
            "labelled faults injected into real annotated receipts that balance "
            "as issued; detection scored against the known fault"
        ),
        "dataset": {
            "source": "CORD-v2 (naver-clova-ix/cord-v2), CC BY 4.0",
            "splits": args.splits,
            "receipts": stats.receipts,
            "lines": stats.line_count,
            "receipts_balancing_as_issued": len(clean),
            "receipts_inconsistent_as_issued": len(inconsistent),
            "inconsistent_share": round(len(inconsistent) / stats.receipts, 4) if stats.receipts else None,
        },
        "false_alarms_on_clean_receipts": false_alarms,
        "overall": {
            "cases": total_cases,
            "detected": total_detected,
            "recall": round(total_detected / total_cases, 4) if total_cases else None,
        },
        "by_fault_kind": summary,
        "ablation": {
            name: {
                "tiers": list(ABLATIONS[name]),
                "by_fault_kind": {
                    k: {
                        "applicable": v["applicable"],
                        "detected": v["detected"],
                        "recall": round(v["detected"] / v["applicable"], 4) if v["applicable"] else None,
                    }
                    for k, v in kinds.items()
                },
                "overall_recall": round(
                    sum(v["detected"] for v in kinds.values())
                    / max(sum(v["applicable"] for v in kinds.values()), 1), 4
                ),
            }
            for name, kinds in ablation.items()
        },
        "runtime_seconds": round(bench_s, 2),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_detection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(out / "audit_per_case.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'fault kind':<22} {'cases':>6} {'recall':>8} {'localise':>9} {'amount':>8}")
    print("-" * 58)
    for kind, s in sorted(summary.items()):
        loc = f"{s['localisation_rate']:.1%}" if s["localisation_rate"] is not None else "   n/a"
        print(f"{kind:<22} {s['applicable_cases']:>6} {s['recall']:>8.1%} "
              f"{loc:>9} {s['amount_exact_rate']:>8.1%}")
    print("-" * 58)
    print(f"{'overall':<22} {total_cases:>6} {total_detected / total_cases:>8.1%}")

    print(f"\nablation ({'  '.join(ABLATIONS)}):")
    kinds = sorted(summary)
    print(f"  {'fault kind':<22}" + "".join(f"{n:>18}" for n in ABLATIONS))
    for k in kinds:
        cells = ""
        for name in ABLATIONS:
            c = ablation[name][k]
            cells += f"{c['detected'] / c['applicable']:>17.1%} " if c["applicable"] else f"{'n/a':>18}"
        print(f"  {k:<22}" + cells)
    print(f"  {'overall':<22}" + "".join(
        f"{payload['ablation'][n]['overall_recall']:>17.1%} " for n in ABLATIONS
    ))
    print(f"\nfalse alarms on {len(clean)} untouched clean receipts: {false_alarms}")
    print(f"written: {out / 'audit_detection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
