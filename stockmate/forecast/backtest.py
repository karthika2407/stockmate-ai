"""Rolling-origin backtest.

The evaluation question is narrow: standing at a particular day, knowing only
what had happened by then, how well does each method predict the average daily
demand over the next lead time?  That is the quantity the replenishment decision
consumes, so it is the quantity worth scoring.

The harness refits the global model at every origin from scratch.  It is slower
than fitting once and sliding the window, and it is the only version that cannot
leak future information backwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import BASELINES
from .global_model import GlobalDemandModel

__all__ = ["BacktestResult", "rolling_origin_backtest", "metrics", "skill_vs"]


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Error measures for one method over one pooled set of forecasts.

    WAPE is the headline: it is scale-free across SKUs that differ by three
    orders of magnitude in volume, and unlike MAPE it does not explode on the
    zero-demand days that make up nearly half this data.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - actual
    abs_err = np.abs(err)

    denom = np.abs(actual).sum()
    wape = float(abs_err.sum() / denom) if denom > 0 else float("nan")

    smape_denom = (np.abs(actual) + np.abs(pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_terms = np.where(smape_denom > 0, abs_err / smape_denom, 0.0)

    return {
        "n": int(len(actual)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": wape,
        "smape": float(smape_terms.mean()),
        "bias": float(err.mean()),
        "bias_pct_of_mean": (
            float(err.mean() / actual.mean()) if actual.mean() > 0 else float("nan")
        ),
    }


def skill_vs(method: dict[str, float], reference: dict[str, float], key: str = "wape") -> float:
    """Fractional improvement of ``method`` over ``reference``. Positive is better."""
    ref = reference.get(key)
    own = method.get(key)
    if ref in (None, 0) or own is None or np.isnan(ref) or np.isnan(own):
        return float("nan")
    return float((ref - own) / ref)


#: Weight used at the first origin, before any error history exists.
DEFAULT_BLEND_WEIGHT = 0.5


def _fit_blend_weight(
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    grid: np.ndarray | None = None,
) -> float:
    """Pick the booster's share of the blend from past origins only.

    ``history`` holds (actual, booster, croston) triples for origins already
    scored.  Returns the weight minimising pooled absolute error over them, or
    the default when there is nothing to learn from yet.
    """
    if not history:
        return DEFAULT_BLEND_WEIGHT
    actual = np.concatenate([h[0] for h in history])
    booster = np.concatenate([h[1] for h in history])
    croston = np.concatenate([h[2] for h in history])
    if len(actual) == 0:
        return DEFAULT_BLEND_WEIGHT

    grid = np.arange(0.0, 1.001, 0.05) if grid is None else grid
    errors = [np.abs(w * booster + (1 - w) * croston - actual).sum() for w in grid]
    return float(grid[int(np.argmin(errors))])


@dataclass
class BacktestResult:
    per_forecast: pd.DataFrame
    summary: dict[str, dict[str, float]]
    origins: list[str]
    config: dict = field(default_factory=dict)
    fit_seconds: float = 0.0
    blend_weights: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "origins": self.origins,
            "blend_weights_per_origin": self.blend_weights,
            "fit_seconds": round(self.fit_seconds, 2),
            "methods": self.summary,
            "skill_vs_seasonal_naive": {
                name: round(skill_vs(m, self.summary["seasonal_naive"]), 4)
                for name, m in self.summary.items()
            },
        }


def rolling_origin_backtest(
    panel: pd.DataFrame,
    *,
    horizon: int = 7,
    n_origins: int = 8,
    origin_gap_days: int = 21,
    min_train_days: int = 240,
    model_factory=None,
) -> BacktestResult:
    """Score the global model and every baseline over several cut-off days.

    ``panel`` is the dense (sku_id, day, units) frame from the retail loader.
    """
    panel = panel.sort_values(["sku_id", "day"]).reset_index(drop=True)
    all_days = np.sort(panel["day"].unique())

    last_usable = len(all_days) - horizon - 1
    first_usable = min_train_days
    if last_usable <= first_usable:
        raise ValueError("not enough history for the requested horizon and training window")

    origins_idx = [
        last_usable - i * origin_gap_days
        for i in range(n_origins)
        if last_usable - i * origin_gap_days > first_usable
    ][::-1]
    origin_days = [pd.Timestamp(all_days[i]) for i in origins_idx]

    rows: list[dict] = []
    fit_seconds = 0.0
    #: Errors seen at origins already passed, used to set the blend weight for
    #: the next one.  Nothing from the current or any later origin enters this.
    blend_history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    blend_weights: list[dict] = []

    for origin in origin_days:
        train = panel[panel["day"] <= origin]
        future = panel[(panel["day"] > origin) & (panel["day"] <= origin + pd.Timedelta(days=horizon))]

        # Actual mean daily demand over the horizon, per SKU.
        actual = (
            future.groupby("sku_id")["units"].agg(["mean", "count"]).rename(columns={"mean": "actual"})
        )
        actual = actual[actual["count"] == horizon]          # full horizons only
        if actual.empty:
            continue

        # Baselines, from each SKU's own history up to the origin.
        preds: dict[str, dict[str, float]] = {name: {} for name in BASELINES}
        for sku, grp in train.groupby("sku_id", sort=False):
            if sku not in actual.index:
                continue
            hist = grp["units"].to_numpy(dtype=float)
            for name, fn in BASELINES.items():
                preds[name][sku] = float(np.mean(fn(hist, horizon)))

        # The global model, refitted on this origin's training data only.
        t0 = time.time()
        factory = model_factory or (lambda: GlobalDemandModel(horizon=horizon))
        model = factory().fit(train)
        fitted = model.predict_rate(train).set_index("sku_id")["forecast_daily"]
        fit_seconds += time.time() - t0
        preds["stockmate_global"] = fitted.to_dict()

        # The shipped forecaster is a blend of the booster and Croston.  Its
        # weight is fitted on the origins already evaluated, so the first origin
        # uses an even split and every later one uses only the past.
        weight = _fit_blend_weight(blend_history)
        shared = [s for s in actual.index
                  if s in preds["croston_sba"] and s in preds["stockmate_global"]]
        preds["stockmate_blend"] = {
            s: weight * preds["stockmate_global"][s] + (1 - weight) * preds["croston_sba"][s]
            for s in shared
        }
        blend_weights.append({"origin": origin.date().isoformat(), "weight": round(weight, 3)})
        blend_history.append((
            np.array([actual.loc[s, "actual"] for s in shared], dtype=float),
            np.array([preds["stockmate_global"][s] for s in shared], dtype=float),
            np.array([preds["croston_sba"][s] for s in shared], dtype=float),
        ))

        for sku in actual.index:
            base = {"origin": origin.date().isoformat(), "sku_id": sku, "actual": float(actual.loc[sku, "actual"])}
            for name, mapping in preds.items():
                if sku in mapping:
                    rows.append({**base, "method": name, "pred": float(mapping[sku])})

    per_forecast = pd.DataFrame(rows)
    summary = {
        name: metrics(grp["actual"].to_numpy(), grp["pred"].to_numpy())
        for name, grp in per_forecast.groupby("method")
    }

    return BacktestResult(
        per_forecast=per_forecast,
        summary=summary,
        origins=[d.date().isoformat() for d in origin_days],
        config={
            "horizon_days": horizon,
            "n_origins": len(origin_days),
            "origin_gap_days": origin_gap_days,
            "min_train_days": min_train_days,
            "skus": int(panel["sku_id"].nunique()),
            "panel_rows": int(len(panel)),
        },
        fit_seconds=fit_seconds,
        blend_weights=blend_weights,
    )
