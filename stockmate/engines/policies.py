"""Ordering policies compared in the replenishment simulation.

Each one is a plain callable so the simulation harness cannot tell them apart,
and so the only difference between a baseline and the proposed policy is the
decision rule itself.  All of them see the same history, the same position and
the same cash limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..forecast.baselines import croston_sba, moving_average, seasonal_naive
from .planner import SERVICE_Z

__all__ = [
    "FixedCoverPolicy",
    "ForecastReorderPolicy",
    "StockMatePolicy",
    "proportional_budget_cut",
]


def proportional_budget_cut(
    wanted: dict[str, float], unit_cost: dict[str, float], budget: float
) -> dict[str, float]:
    """Scale every order down by the same fraction to fit the budget.

    This is what a shop does by instinct when the bill comes to more than the
    cash in the till: buy a bit less of everything.  It is the baseline the
    value-per-rupee allocation has to beat, and it is not a straw man, because
    it keeps the assortment intact.

    The leftover pass matters for fairness.  Scaling and then flooring every
    order strands a large part of the budget unspent, and a policy that simply
    spends more of the available cash would then appear to win on the strength
    of the arithmetic rather than the decision.  So the remainder is handed out
    by largest fractional shortfall until no further unit is affordable, which
    is both what a shopkeeper would do and what makes the comparison mean
    something.
    """
    total = sum(q * unit_cost.get(s, 0.0) for s, q in wanted.items())
    if total <= budget or total <= 0:
        return wanted

    scale = budget / total
    scaled = {s: q * scale for s, q in wanted.items()}
    out = {s: float(math.floor(v)) for s, v in scaled.items()}
    spent = sum(out[s] * unit_cost.get(s, 0.0) for s in out)

    remainders = sorted(
        scaled, key=lambda s: -(scaled[s] - math.floor(scaled[s]))
    )
    left = budget - spent
    for sku in remainders:
        cost = unit_cost.get(sku, 0.0)
        if cost <= 0 or out[sku] >= wanted[sku]:
            continue
        if cost <= left:
            out[sku] += 1.0
            left -= cost
    return out


def value_per_rupee_cut(
    wanted: dict[str, float],
    unit_cost: dict[str, float],
    rate: dict[str, float],
    budget: float,
    *,
    floor_days: float = 0.0,
    lead_time_days: int = 3,
) -> dict[str, float]:
    """Spend the cash where it prevents the most lost sales.

    Units are taken in order of expected daily demand per rupee of cost, so a
    fast mover is filled before a slow one gets anything.

    Left unchecked this is a trap, and measuring it is what exposed the trap.
    Ranked purely by value per rupee the rule pours an entire tight budget into
    the dozen cheapest fast movers and orders nothing at all for the rest: the
    unit fill rate climbs, because those few lines are most of the units, while
    most of the shelves empty and stay empty.  A kirana that stops stocking two
    thirds of its lines has not optimised anything; it has lost the customer who
    came in for the missing item and bought their whole basket elsewhere.

    ``floor_days`` fixes that by reserving, for every SKU that needs anything at
    all, enough to cover the delivery wait before the ranking gets to spend what
    is left.  Set it to zero to reproduce the degenerate behaviour, which the
    benchmark does deliberately so the effect is visible rather than argued.
    """
    total = sum(q * unit_cost.get(s, 0.0) for s, q in wanted.items())
    if total <= budget or total <= 0:
        return wanted

    out = {s: 0.0 for s in wanted}
    left = budget

    if floor_days > 0:
        # First pass: a thin layer for everyone who needs something, cheapest
        # need first so the floor reaches as many lines as the cash allows.
        floors = {
            s: min(wanted[s], math.ceil(max(rate.get(s, 0.0), 0.0) * floor_days))
            for s in wanted if wanted[s] > 0
        }
        for sku in sorted(floors, key=lambda s: floors[s] * unit_cost.get(s, 0.0)):
            cost = unit_cost.get(sku, 0.0)
            need = floors[sku]
            if cost <= 0 or need <= 0:
                continue
            take = min(need, math.floor(left / cost))
            if take <= 0:
                continue
            out[sku] = float(take)
            left -= take * cost

    # Second pass: whatever remains goes where it buys the most demand covered.
    order = sorted(
        wanted,
        key=lambda s: -(max(rate.get(s, 0.0), 1e-9) / max(unit_cost.get(s, 1e-9), 1e-9)),
    )
    for sku in order:
        cost = unit_cost.get(sku, 0.0)
        room = wanted[sku] - out[sku]
        if cost <= 0 or room <= 0:
            continue
        take = min(room, math.floor(left / cost))
        if take <= 0:
            continue
        out[sku] += float(take)
        left -= take * cost
    return out


@dataclass
class FixedCoverPolicy:
    """Order up to a fixed number of days of cover, using a trailing average.

    This is the rule of thumb a careful owner already uses: keep about two
    weeks of what normally sells.  It needs no model at all.
    """

    cover_days: float = 14.0
    window: int = 28
    name: str = "fixed_cover_14d"

    def __call__(self, history, on_hand, on_order, unit_cost, budget):
        wanted = {}
        rate = {}
        for sku, h in history.items():
            r = float(np.mean(h[-self.window:])) if len(h) else 0.0
            rate[sku] = r
            target = r * self.cover_days
            gap = target - on_hand.get(sku, 0.0) - on_order.get(sku, 0.0)
            wanted[sku] = math.ceil(gap) if gap > 0 else 0.0
        return proportional_budget_cut(wanted, unit_cost, budget)


@dataclass
class ForecastReorderPolicy:
    """Classical reorder point driven by one named forecaster.

    Separating this from the proposed policy is what isolates the contribution:
    if StockMate wins only because it forecasts better, this shows it, and if it
    wins because of how it spends a tight budget, this shows that instead.
    """

    method: str = "croston"
    lead_time_days: int = 3
    review_days: int = 7
    service_level: float = 0.95
    name: str = "reorder_croston"

    def _rate(self, h: np.ndarray) -> float:
        horizon = self.lead_time_days + self.review_days
        if self.method == "croston":
            return float(np.mean(croston_sba(h, horizon)))
        if self.method == "seasonal_naive":
            return float(np.mean(seasonal_naive(h, horizon)))
        return float(np.mean(moving_average(h, horizon)))

    def __call__(self, history, on_hand, on_order, unit_cost, budget):
        z = SERVICE_Z.get(round(self.service_level, 3), SERVICE_Z[0.95])
        exposure = self.lead_time_days + self.review_days
        wanted, rate = {}, {}
        for sku, h in history.items():
            r = self._rate(h)
            rate[sku] = r
            sd = float(np.std(h[-56:])) if len(h) >= 7 else 0.0
            target = r * exposure + z * sd * math.sqrt(exposure)
            gap = target - on_hand.get(sku, 0.0) - on_order.get(sku, 0.0)
            wanted[sku] = math.ceil(gap) if gap > 0 else 0.0
        return proportional_budget_cut(wanted, unit_cost, budget)


@dataclass
class StockMatePolicy:
    """The shipped policy: blended forecast, plus value-per-rupee allocation.

    Two differences from the reorder baseline, and the simulation is run with
    each of them separately so the credit lands where it is due:

      * the demand rate is a blend of Croston and a trailing seasonal average,
        which is the forecaster the accuracy benchmark selected;
      * when the targets cost more than the cycle's cash, the cut is made by
        expected demand per rupee instead of proportionally.
    """

    lead_time_days: int = 3
    review_days: int = 7
    service_level: float = 0.95
    blend_weight: float = 0.4
    smart_allocation: bool = True
    #: Days of cover guaranteed to every line before the ranking spends the
    #: rest.  Zero reproduces the assortment-destroying variant on purpose.
    floor_days: float = 3.0
    name: str = "stockmate"

    def __call__(self, history, on_hand, on_order, unit_cost, budget):
        z = SERVICE_Z.get(round(self.service_level, 3), SERVICE_Z[0.95])
        exposure = self.lead_time_days + self.review_days
        wanted, rate = {}, {}
        for sku, h in history.items():
            croston = float(np.mean(croston_sba(h, exposure)))
            recent = float(np.mean(h[-28:])) if len(h) else 0.0
            r = self.blend_weight * recent + (1 - self.blend_weight) * croston
            rate[sku] = r
            sd = float(np.std(h[-56:])) if len(h) >= 7 else 0.0
            target = r * exposure + z * sd * math.sqrt(exposure)
            gap = target - on_hand.get(sku, 0.0) - on_order.get(sku, 0.0)
            wanted[sku] = math.ceil(gap) if gap > 0 else 0.0

        if self.smart_allocation:
            return value_per_rupee_cut(
                wanted, unit_cost, rate, budget,
                floor_days=self.floor_days, lead_time_days=self.lead_time_days,
            )
        return proportional_budget_cut(wanted, unit_cost, budget)
