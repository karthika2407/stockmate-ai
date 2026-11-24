"""Hyperparameter selection for the global demand model, on a disjoint SKU set.

The reported accuracy benchmark evaluates the top-N SKUs by volume.  Choosing
the model's settings on those same SKUs would make the headline number a
best-of-sweep rather than an honest estimate, so the sweep runs here on a
separate band of SKUs ranked below the evaluation set.  The winning
configuration is then frozen and used unchanged in bench_forecast.py.

Run:
    python benchmarks/bench_tuning.py
Writes results/forecast_tuning.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockmate.datasets.retail import daily_demand, load_online_retail  # noqa: E402
from stockmate.forecast.backtest import rolling_origin_backtest  # noqa: E402
from stockmate.forecast.global_model import GlobalDemandModel  # noqa: E402

GRID = [
    ("poisson_d6", dict(loss="poisson", max_depth=6, max_iter=300, min_samples_leaf=40)),
    ("squared_d6", dict(loss="squared_error", max_depth=6, max_iter=300, min_samples_leaf=40)),
    ("absolute_d6", dict(loss="absolute_error", max_depth=6, max_iter=300, min_samples_leaf=40)),
    ("absolute_deep", dict(loss="absolute_error", max_depth=None, max_iter=400, min_samples_leaf=20)),
    ("absolute_slow", dict(loss="absolute_error", max_depth=None, max_iter=600,
                           min_samples_leaf=20, learning_rate=0.03)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT.parent / "_datasets" / "uci_online_retail_ii" / "online_retail_II.xlsx"))
    ap.add_argument("--eval-skus", type=int, default=300,
                    help="size of the evaluation set this tuning must avoid")
    ap.add_argument("--tune-skus", type=int, default=200, help="SKUs to tune on, ranked below it")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    data = load_online_retail(args.data)
    wide = daily_demand(data, top_n=args.eval_skus + args.tune_skus)
    volume = wide.groupby("sku_id")["units"].sum().sort_values(ascending=False)
    eval_skus = set(volume.head(args.eval_skus).index)
    tune_skus = set(volume.iloc[args.eval_skus: args.eval_skus + args.tune_skus].index)
    assert not (eval_skus & tune_skus), "tuning and evaluation SKUs must not overlap"

    panel = wide[wide["sku_id"].isin(tune_skus)].copy()
    print(f"tuning on {panel['sku_id'].nunique()} SKUs disjoint from the "
          f"{len(eval_skus)} evaluation SKUs ({len(panel):,} SKU-days)")

    rows = []
    for name, kw in GRID:
        t0 = time.time()
        r = rolling_origin_backtest(
            panel, horizon=7, n_origins=3, origin_gap_days=21,
            model_factory=lambda kw=kw: GlobalDemandModel(horizon=7, **kw),
        )
        m = r.summary["stockmate_global"]
        b = r.summary["stockmate_blend"]
        rows.append({
            "config": name, "params": kw,
            "global_wape": round(m["wape"], 4), "global_mae": round(m["mae"], 3),
            "global_bias": round(m["bias"], 3),
            "blend_wape": round(b["wape"], 4),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  {name:<16} global WAPE {m['wape']:.4f}  blend WAPE {b['wape']:.4f}  "
              f"({time.time() - t0:.0f}s)")

    best = min(rows, key=lambda r: r["blend_wape"])
    payload = {
        "protocol": "hyperparameters chosen on SKUs disjoint from the evaluation set",
        "eval_sku_count": len(eval_skus),
        "tune_sku_count": len(tune_skus),
        "tune_panel_rows": int(len(panel)),
        "grid": rows,
        "selected": best["config"],
        "selected_params": best["params"],
        "baselines_on_tuning_set": {
            k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
            for k, v in r.summary.items() if k in ("croston_sba", "seasonal_naive")
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forecast_tuning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nselected: {best['config']}  {best['params']}")
    print(f"written: {out / 'forecast_tuning.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
