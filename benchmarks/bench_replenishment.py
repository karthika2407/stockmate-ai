"""Benchmark: replenishment policies replayed against real demand.

Every policy sees the same demand traces, the same opening stock, the same lead
time and the same cash per cycle.  What differs is only the decision rule, so
the gap between them is attributable to the rule and to nothing else.

The ablation matters more than the headline here.  StockMate changes two things
at once against a classical reorder point: the demand rate it uses, and how it
cuts the order when the cash runs short.  Both are run on their own so the
credit can be assigned honestly, and if the allocation is doing all the work
then that is what the table will say.

Run:
    python benchmarks/bench_replenishment.py --skus 150
Writes results/replenishment.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockmate.datasets.retail import daily_demand, load_online_retail  # noqa: E402
from stockmate.engines.policies import (  # noqa: E402
    FixedCoverPolicy, ForecastReorderPolicy, StockMatePolicy,
)
from stockmate.engines.simulate import SimConfig, simulate, summarise  # noqa: E402


def build_policies(cfg: SimConfig):
    common = dict(lead_time_days=cfg.lead_time_days, review_days=cfg.review_days)
    return [
        FixedCoverPolicy(cover_days=14.0),
        ForecastReorderPolicy(method="seasonal_naive", name="reorder_seasonal_naive", **common),
        ForecastReorderPolicy(method="croston", name="reorder_croston", **common),
        StockMatePolicy(smart_allocation=False, name="stockmate_forecast_only", **common),
        # floor_days=0 is the ranked allocation with no protection for the tail.
        # It is kept in the comparison because it scores well on the headline
        # fill rate while emptying most of the shop, and that is worth showing.
        StockMatePolicy(smart_allocation=True, floor_days=0.0,
                        name="stockmate_rank_only", **common),
        StockMatePolicy(smart_allocation=True, floor_days=3.0,
                        name="stockmate_full", **common),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT.parent / "_datasets" / "uci_online_retail_ii" / "online_retail_II.xlsx"))
    ap.add_argument("--skus", type=int, default=150)
    ap.add_argument("--lead-time", type=int, default=3)
    ap.add_argument("--review-days", type=int, default=7)
    ap.add_argument("--budget-days", type=float, default=6.0,
                    help="cash per cycle as days of cost of goods sold")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = SimConfig(
        lead_time_days=args.lead_time,
        review_days=args.review_days,
        budget_days_of_cogs=args.budget_days,
    )

    t0 = time.time()
    data = load_online_retail(args.data)
    panel = daily_demand(data, top_n=args.skus)
    print(f"{panel['sku_id'].nunique()} SKUs over {panel['day'].nunique()} days "
          f"({time.time() - t0:.1f}s)")
    print(f"lead time {cfg.lead_time_days}d, review every {cfg.review_days}d, "
          f"cash per cycle {cfg.budget_days_of_cogs} days of COGS")

    results = []
    for policy in build_policies(cfg):
        t = time.time()
        r = simulate(panel, policy, cfg)
        results.append(r)
        print(f"  {r.policy:<28} fill {r.fill_rate:6.2%}  "
              f"median SKU {r.median_sku_fill:6.2%}  "
              f"ok {r.well_served_skus:>3}  dead {r.abandoned_skus:>3}  "
              f"held {r.mean_inventory_value:9,.0f}  ({time.time() - t:.0f}s)")

    payload = summarise(
        results,
        budget_note=(
            f"cash per review cycle set to {cfg.budget_days_of_cogs} days of cost "
            "of goods sold, so the budget binds and the allocation rule matters"
        ),
    )
    payload["config"] = {
        "skus": int(panel["sku_id"].nunique()),
        "days_simulated": results[0].days if results else 0,
        "lead_time_days": cfg.lead_time_days,
        "review_days": cfg.review_days,
        "opening_cover_days": cfg.opening_cover_days,
        "budget_days_of_cogs": cfg.budget_days_of_cogs,
        "warmup_days": cfg.warmup_days,
        "slow_mover_days": cfg.slow_mover_days,
    }
    payload["dataset"] = data.report
    payload["runtime_seconds"] = round(time.time() - t0, 2)
    payload["environment"] = {"python": platform.python_version(), "platform": platform.platform()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replenishment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out / 'replenishment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
