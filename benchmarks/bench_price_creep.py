"""Benchmark: price creep against real price history.

Arithmetic cannot see a quietly raised unit price, because a raised price still
multiplies correctly.  Only history can.  The question this benchmark answers is
what that history is worth in practice, and what it costs the owner in alerts.

The honest framing matters here.  A genuine price rise is a price rise: when
StockMate says "this costs more than before" on real data it is not wrong, it is
telling the owner something true that they may not want told every week.  So the
two numbers reported are not precision and recall against a ground truth of
fraud, which does not exist.  They are:

  * detection rate: of injected rises of a known size, how many are reported;
  * alert burden: on the untouched real series, what share of purchases trigger
    an alert anyway, because prices really did move.

A threshold that catches everything and fires on a third of all purchases is
useless in a shop.  The sweep exists to find where those two meet.

Data
----
Online Retail II (UCI 502, CC BY 4.0).  Each transaction line carries the unit
price actually charged, so the price series is real market movement across two
years, including promotions, repricing and data-entry noise.  The dataset has a
single seller, so "supplier" is constant throughout and the comparison is
per-SKU over time, which is the comparison the engine makes.

Run:
    python benchmarks/bench_price_creep.py
Writes results/price_creep.json and results/price_creep_sweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockmate.datasets.perturb import seeded_random  # noqa: E402
from stockmate.datasets.retail import load_online_retail  # noqa: E402
from stockmate.engines.bill_auditor import MIN_OBSERVATIONS  # noqa: E402
from stockmate.money import Money  # noqa: E402

THRESHOLDS = [Decimal(x) for x in ("2", "5", "10", "15", "25")]
INJECTED_RISES = [Decimal(x) for x in ("3", "5", "8", "12", "20", "35")]

#: Floors on the rupee value of an alert, swept rather than fixed.
#:
#: The evaluation data is a GBP catalogue whose unit prices run from about 0.04
#: to 12.75, so an absolute floor tuned for Indian grocery bills suppresses
#: almost every alert here and tells us nothing about the rule.  The floor is
#: therefore swept in the data's own units and reported as such.  An operator
#: setting this for a real shop must pick it against their own line values, and
#: the shipped default is stated in rupees for that reason.
IMPACT_FLOORS = [Decimal(x) for x in ("0", "0.25", "1.00", "5.00")]


def fires(now: Money, reference: Money, qty: Decimal,
          creep_pct: Decimal, min_impact: Money) -> bool:
    """The engine's decision rule, isolated so the sweep tests exactly it."""
    if reference.is_zero():
        return False
    rise = now - reference
    if rise.paise <= 0:
        return False
    pct = (Decimal(rise.paise) / Decimal(reference.paise)) * 100
    impact = rise * qty
    return pct >= creep_pct and impact.paise >= min_impact.paise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT.parent / "_datasets" / "uci_online_retail_ii" / "online_retail_II.xlsx"))
    ap.add_argument("--skus", type=int, default=400, help="SKUs with the most purchase events")
    ap.add_argument("--min-impact", type=float, default=20.0, help="rupee floor on an alert")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    min_impact = Money.from_rupees(str(args.min_impact))

    t0 = time.time()
    data = load_online_retail(args.data)
    tx = data.transactions
    tx = tx[~tx["is_cancellation"]].sort_values("ts")
    counts = tx["sku_id"].value_counts().head(args.skus)
    tx = tx[tx["sku_id"].isin(counts.index)]
    print(f"{len(tx):,} purchase events across {tx['sku_id'].nunique()} SKUs  "
          f"({time.time() - t0:.1f}s)")

    # Walk each SKU forward in time.  At every event with enough history, record
    # the observed rise against the running median of what came before.
    events: list[dict] = []
    for sku, grp in tx.groupby("sku_id", sort=False):
        prices = [Money.from_rupees(str(p)) for p in grp["unit_price"].tolist()]
        quantities = [Decimal(str(abs(q))) for q in grp["units"].tolist()]
        for i in range(MIN_OBSERVATIONS, len(prices)):
            reference = Money(int(round(statistics.median(p.paise for p in prices[:i]))))
            if reference.is_zero():
                continue
            events.append({
                "sku_id": sku,
                "now_paise": prices[i].paise,
                "ref_paise": reference.paise,
                "qty": quantities[i],
            })
    print(f"{len(events):,} comparison points with at least "
          f"{MIN_OBSERVATIONS} prior purchases")

    rng = seeded_random("price_creep_bench", len(events))
    sample = events if len(events) <= 120_000 else rng.sample(events, 120_000)

    rows = []
    print(f"\n{'floor':>7} {'thresh':>7} {'burden':>8}  " +
          "  ".join(f"{int(r)}%" for r in INJECTED_RISES))
    print("-" * 60)
    for floor in IMPACT_FLOORS:
        floor_money = Money.from_rupees(floor)
        for thr in THRESHOLDS:
            natural = sum(
                1 for e in sample
                if fires(Money(e["now_paise"]), Money(e["ref_paise"]), e["qty"], thr, floor_money)
            )
            row = {
                "impact_floor": float(floor),
                "threshold_pct": float(thr),
                "comparison_points": len(sample),
                "natural_alerts": natural,
                "alert_burden": round(natural / len(sample), 4) if sample else None,
            }
            for rise in INJECTED_RISES:
                detected = applicable = 0
                for e in sample:
                    ref = Money(e["ref_paise"])
                    # Inject on top of the reference, so the measured rise is
                    # exactly the injected one and not confounded by real movement.
                    injected = ref * (1 + rise / 100)
                    if injected.paise <= ref.paise:
                        continue
                    applicable += 1
                    detected += int(fires(injected, ref, e["qty"], thr, floor_money))
                row[f"detect_{rise}pct"] = round(detected / applicable, 4) if applicable else None
            rows.append(row)
            print(f"{float(floor):>7.2f} {float(thr):>6.1f}% {row['alert_burden']:>8.2%}  " +
                  "  ".join(f"{row[f'detect_{r}pct']:.0%}" for r in INJECTED_RISES))

    payload = {
        "protocol": (
            "per-SKU running-median price comparison on real transaction prices; "
            "detection measured on rises injected on top of the reference price, "
            "alert burden measured on the untouched real series"
        ),
        "dataset": {
            "source": "UCI Online Retail II (502), DOI 10.24432/C5CG6D, CC BY 4.0",
            "purchase_events": int(len(tx)),
            "skus": int(tx["sku_id"].nunique()),
            "comparison_points": len(events),
            "sampled_points": len(sample),
            "note": "single-seller dataset, so the comparison is per SKU over time",
        },
        "impact_floor_note": (
            "floors are in the dataset's own currency units (GBP), swept rather "
            "than fixed, because an absolute rupee floor does not transfer to a "
            "catalogue whose unit prices run from 0.04 to 12.75"
        ),
        "min_observations": MIN_OBSERVATIONS,
        "thresholds": rows,
        "runtime_seconds": round(time.time() - t0, 2),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "price_creep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(out / "price_creep_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {out / 'price_creep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
