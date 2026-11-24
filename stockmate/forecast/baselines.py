"""Forecasting baselines.

Any claim that a model helps a shopkeeper has to be measured against what the
shopkeeper could do without it.  These are the policies a competent owner
already applies by eye: repeat last week, average the recent past, or carry on
as before.  A model that cannot beat them is not worth the compute.

Croston's method is included because nearly half the SKU-days in the evaluation
data have zero demand, and the usual smoothing methods are known to be biased on
intermittent series.  The Syntetos-Boylan correction is applied, since plain
Croston is biased upward.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "naive",
    "seasonal_naive",
    "moving_average",
    "croston_sba",
    "zero_forecast",
    "BASELINES",
]


def zero_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Order nothing. The floor every other method has to clear."""
    return np.zeros(horizon, dtype=float)


def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Tomorrow looks like today."""
    if len(history) == 0:
        return np.zeros(horizon, dtype=float)
    return np.full(horizon, float(history[-1]))


def seasonal_naive(history: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """This Tuesday looks like last Tuesday.

    Retail demand is strongly weekly, so this is a much stronger baseline than
    the plain naive one and is the honest comparison to beat.
    """
    if len(history) < period:
        return naive(history, horizon)
    season = history[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(season, reps)[:horizon].astype(float)


def moving_average(history: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    """Average of the last ``window`` days, held flat across the horizon."""
    if len(history) == 0:
        return np.zeros(horizon, dtype=float)
    w = min(window, len(history))
    return np.full(horizon, float(np.mean(history[-w:])))


def croston_sba(
    history: np.ndarray,
    horizon: int,
    alpha: float = 0.1,
) -> np.ndarray:
    """Croston's method with the Syntetos-Boylan bias correction.

    Demand size and the gap between demands are smoothed separately, and the
    rate is their ratio scaled by ``1 - alpha/2``.  Returns a flat rate across
    the horizon, which is what this family of methods produces.
    """
    nz = np.nonzero(history)[0]
    if len(nz) == 0:
        return np.zeros(horizon, dtype=float)
    if len(nz) == 1:
        rate = float(history[nz[0]]) / max(len(history), 1)
        return np.full(horizon, rate)

    sizes = history[nz].astype(float)
    intervals = np.diff(np.concatenate(([nz[0] - 1], nz))).astype(float)

    z = sizes[0]        # smoothed demand size
    p = intervals[0]    # smoothed interval between demands
    for size, gap in zip(sizes[1:], intervals[1:]):
        z += alpha * (size - z)
        p += alpha * (gap - p)

    if p <= 0:
        return np.zeros(horizon, dtype=float)
    rate = (1.0 - alpha / 2.0) * (z / p)
    return np.full(horizon, max(rate, 0.0))


#: Name -> callable(history, horizon) used by the backtest harness.
BASELINES = {
    "zero": zero_forecast,
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average_28": moving_average,
    "croston_sba": croston_sba,
}
