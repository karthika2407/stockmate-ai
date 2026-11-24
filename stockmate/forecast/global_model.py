"""A single gradient-boosted model fitted across every SKU at once.

Fitting one model per SKU is the obvious thing to do and the wrong thing to do
here: the median SKU has under two years of daily history, nearly half of it
zeros, which is not enough to learn a weekly shape per product.  Pooling the
SKUs lets the model learn the calendar effects once, from all of them together,
and use the per-SKU lag features to tell the products apart.  This is the
standard global-model setup from the forecasting-competition literature, and on
this data it is also the only one that fits in the CPU budget.

The features are deliberately plain: lags, rolling statistics, time since the
last sale, and calendar flags.  Nothing here looks into the future, and the
backtest refits at every origin so no information crosses the forecast boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

__all__ = ["GlobalDemandModel", "build_features", "FEATURE_LAGS", "ROLL_WINDOWS"]

FEATURE_LAGS = (1, 2, 3, 7, 14, 28)
ROLL_WINDOWS = (7, 14, 28, 56)

#: Floor on the per-SKU scale, so a dormant product cannot divide by zero.
SCALE_FLOOR = 0.25


def _croston_rolling(s: pd.Series, alpha: float = 0.1) -> pd.Series:
    """Croston/SBA rate recomputed online, one value per day, using past data only."""
    out = np.full(len(s), np.nan)
    z = p = None
    gap = 0
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if z is not None and p is not None and p > 0:
            out[i] = (1.0 - alpha / 2.0) * (z / p)
        if np.isnan(v):
            continue
        gap += 1
        if v > 0:
            if z is None:
                z, p = v, float(gap)
            else:
                z += alpha * (v - z)
                p += alpha * (gap - p)
            gap = 0
    return pd.Series(out, index=s.index)


def build_features(
    panel: pd.DataFrame,
    *,
    horizon: int,
    with_target: bool = True,
) -> pd.DataFrame:
    """Lag, rolling and calendar features for a (sku_id, day, units) panel.

    The target is the mean daily demand over the next ``horizon`` days, because
    that is what the replenishment decision actually needs: how much will move
    between now and the next delivery.  Predicting a single day ahead would
    optimise the wrong quantity.
    """
    df = panel.sort_values(["sku_id", "day"]).copy()
    g = df.groupby("sku_id", sort=False)["units"]

    for lag in FEATURE_LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    for win in ROLL_WINDOWS:
        shifted = g.shift(1)
        df[f"roll_mean_{win}"] = shifted.groupby(df["sku_id"]).transform(
            lambda s, w=win: s.rolling(w, min_periods=max(2, w // 4)).mean()
        )
        df[f"roll_std_{win}"] = shifted.groupby(df["sku_id"]).transform(
            lambda s, w=win: s.rolling(w, min_periods=max(2, w // 4)).std()
        )
        df[f"roll_nonzero_{win}"] = shifted.groupby(df["sku_id"]).transform(
            lambda s, w=win: (s > 0).rolling(w, min_periods=max(2, w // 4)).mean()
        )

    # Days since the last non-zero sale: the single most informative feature on
    # intermittent demand, and the one the baselines cannot use.
    def _since_last_sale(s: pd.Series) -> pd.Series:
        sold = s.shift(1) > 0
        idx = np.arange(len(s))
        last = np.where(sold.to_numpy(), idx, np.nan)
        last = pd.Series(last, index=s.index).ffill()
        return pd.Series(idx, index=s.index) - last

    df["days_since_sale"] = g.transform(_since_last_sale)

    # Hand the model what the classical methods would have said.  Croston's
    # rate and the weekly repeat are cheap to compute and carry most of the
    # signal on intermittent series, so the booster should be correcting them
    # rather than rediscovering them.
    shifted = g.shift(1)
    df["croston_rate"] = shifted.groupby(df["sku_id"]).transform(_croston_rolling)
    df["seasonal_naive_rate"] = g.shift(7)
    df["seasonal_naive_mean"] = shifted.groupby(df["sku_id"]).transform(
        lambda s: s.rolling(7, min_periods=3).mean()
    )

    day = pd.to_datetime(df["day"])
    df["dow"] = day.dt.dayofweek
    df["dom"] = day.dt.day
    df["month"] = day.dt.month
    df["week_of_year"] = day.dt.isocalendar().week.astype(int)
    df["is_month_end"] = day.dt.is_month_end.astype(int)
    df["days_from_start"] = (day - day.min()).dt.days

    # Pooling SKUs that differ by three orders of magnitude in volume lets the
    # biggest sellers dominate the loss.  Learning the ratio to each SKU's own
    # recent level instead puts every product on the same footing; the
    # prediction is multiplied back by the scale afterwards.
    df["scale"] = df["roll_mean_28"].fillna(df["roll_mean_7"]).fillna(SCALE_FLOOR)
    df["scale"] = df["scale"].clip(lower=SCALE_FLOOR)

    if with_target:
        # Mean demand over the next `horizon` days, aligned so that row t only
        # ever sees data up to t.
        fwd = (
            df.groupby("sku_id", sort=False)["units"]
            .transform(lambda s: s.shift(-1).rolling(horizon, min_periods=horizon).mean())
        )
        df["target"] = fwd

    return df


@dataclass
class GlobalDemandModel:
    """Gradient-boosted regressor over the pooled SKU panel."""

    horizon: int = 7
    #: WAPE scores absolute error, which is minimised by the conditional median,
    #: while squared and Poisson losses chase the conditional mean.  On demand
    #: this skewed the two differ a lot, so the loss is a deliberate choice and
    #: is reported alongside the result rather than left at a default.
    loss: str = "absolute_error"
    # Settings chosen by benchmarks/bench_tuning.py on a band of SKUs ranked
    # below the evaluation set, so the reported accuracy is not a best-of-sweep
    # on its own test data.  See results/forecast_tuning.json.
    max_iter: int = 600
    learning_rate: float = 0.03
    max_depth: int | None = None
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_state: int = 20261124
    model: HistGradientBoostingRegressor | None = field(default=None, repr=False)
    feature_names: list[str] = field(default_factory=list, repr=False)

    #: Ratios above this are treated as outliers during fitting.  A product that
    #: sold forty times its usual rate did so for a reason the features do not
    #: contain, and letting those rows into the loss wrecks the ordinary days.
    ratio_clip: float = 8.0

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        drop = {
            "sku_id", "day", "units", "target", "name",
            "unit_price", "units_raw", "scale", "target_ratio",
        }
        return [c for c in df.columns if c not in drop]

    def fit(self, panel: pd.DataFrame) -> "GlobalDemandModel":
        feats = build_features(panel, horizon=self.horizon, with_target=True)
        feats = feats.dropna(subset=["target"]).copy()

        feats["target_ratio"] = (feats["target"] / feats["scale"]).clip(0.0, self.ratio_clip)
        self.feature_names = self._feature_columns(feats)

        X = feats[self.feature_names].to_numpy(dtype=float)
        y = feats["target_ratio"].to_numpy(dtype=float)

        self.model = HistGradientBoostingRegressor(
            loss=self.loss,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.model.fit(X, y)
        return self

    def predict_rate(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Predicted mean daily demand for the last day of each SKU's history."""
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_rate()")
        feats = build_features(panel, horizon=self.horizon, with_target=False)
        last = feats.groupby("sku_id", sort=False).tail(1).copy()
        X = last[self.feature_names].to_numpy(dtype=float)
        ratio = np.clip(self.model.predict(X), 0.0, self.ratio_clip)
        last["forecast_daily"] = ratio * last["scale"].to_numpy(dtype=float)
        return last[["sku_id", "day", "forecast_daily"]].reset_index(drop=True)
