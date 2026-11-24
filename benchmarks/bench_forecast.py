"""Benchmark: demand forecasting accuracy on Online Retail II.

Run:
    python benchmarks/bench_forecast.py --skus 300 --origins 8

Writes results/forecast_backtest.json and results/forecast_per_forecast.csv.
Every number quoted in the report comes from those two files.
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
from stockmate.forecast.backtest import rolling_origin_backtest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT.parent / "_datasets" / "uci_online_retail_ii" / "online_retail_II.xlsx"))
    ap.add_argument("--skus", type=int, default=300, help="top-N SKUs by volume")
    ap.add_argument("--horizon", type=int, default=7, help="lead time in days")
    ap.add_argument("--origins", type=int, default=8)
    ap.add_argument("--gap", type=int, default=21, help="days between origins")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.data}")
    t0 = time.time()
    data = load_online_retail(args.data)
    panel = daily_demand(data, top_n=args.skus)
    load_s = time.time() - t0
    print(f"  {data.report['clean_rows']:,} clean rows -> "
          f"{panel['sku_id'].nunique()} SKUs x {len(panel):,} SKU-days  ({load_s:.1f}s)")

    print(f"backtesting: horizon={args.horizon}d, {args.origins} origins every {args.gap}d")
    t0 = time.time()
    result = rolling_origin_backtest(
        panel,
        horizon=args.horizon,
        n_origins=args.origins,
        origin_gap_days=args.gap,
    )
    total_s = time.time() - t0

    payload = result.to_dict()
    payload["dataset"] = data.report
    payload["runtime_seconds"] = round(total_s, 2)
    payload["load_seconds"] = round(load_s, 2)
    payload["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    payload["zero_demand_day_share"] = round(float((panel["units"] == 0).mean()), 4)

    (out_dir / "forecast_backtest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result.per_forecast.to_csv(out_dir / "forecast_per_forecast.csv", index=False)

    ref = result.summary["seasonal_naive"]
    print(f"\n{'method':<22} {'WAPE':>8} {'MAE':>8} {'RMSE':>8} {'bias':>8}  skill vs seasonal-naive")
    print("-" * 78)
    for name, m in sorted(result.summary.items(), key=lambda kv: kv[1]["wape"]):
        skill = (ref["wape"] - m["wape"]) / ref["wape"] if ref["wape"] else float("nan")
        print(f"{name:<22} {m['wape']:>8.4f} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
              f"{m['bias']:>8.3f}  {skill:+7.2%}")
    print(f"\n{len(result.per_forecast):,} forecasts over {len(result.origins)} origins "
          f"in {total_s:.1f}s")
    print(f"written: {out_dir / 'forecast_backtest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
