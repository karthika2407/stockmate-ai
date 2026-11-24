"""Replay a shop against real demand and see what a policy would have done.

A forecast benchmark answers "how close was the number".  It does not answer the
question the owner asked, which is whether the shelf had stock when a customer
wanted it and how much cash was sitting in boxes at the back meanwhile.  Those
two pull against each other, and a policy can only be judged on both at once.

So each policy is replayed day by day over the real demand traces: orders are
placed on review days, arrive after the lead time, and are consumed by the
demand that actually occurred.  Demand that arrives with the shelf empty is lost
and recorded as lost, not back-ordered, because a kirana customer walks to the
next shop.

Every policy sees the same demand, the same opening stock, the same lead time
and the same cash limit per cycle.  The only thing that differs is how it
decides, which is the only way the comparison means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..money import Money

__all__ = ["SimConfig", "SimResult", "Policy", "simulate", "summarise"]


class Policy(Protocol):
    """Decides order quantities for every SKU at one review point.

    Receives the demand history strictly before ``day``, the current position,
    and the cash available this cycle.  It must not look at anything dated on or
    after ``day``; the harness slices the history so that it cannot.
    """

    name: str

    def __call__(
        self,
        history: dict[str, np.ndarray],
        on_hand: dict[str, float],
        on_order: dict[str, float],
        unit_cost: dict[str, float],
        budget: float,
    ) -> dict[str, float]: ...


@dataclass
class SimConfig:
    lead_time_days: int = 3
    review_days: int = 7
    #: Opening stock as a multiple of each SKU's mean daily demand over the
    #: warm-up window.  Every policy starts from the same position, so this
    #: choice shifts all results together and does not favour any of them.
    opening_cover_days: float = 7.0
    #: Cash released per review cycle, as a multiple of what the whole shop
    #: sells in a day at cost. Set low to make the budget bind.
    budget_days_of_cogs: float = 6.0
    warmup_days: int = 120
    slow_mover_days: int = 30


@dataclass
class SimResult:
    policy: str
    days: int
    demanded: float = 0.0
    served: float = 0.0
    lost: float = 0.0
    stockout_sku_days: int = 0
    sku_days: int = 0
    on_hand_value_sum: float = 0.0
    on_hand_units_sum: float = 0.0
    ordered_value: float = 0.0
    ordered_units: float = 0.0
    orders: int = 0
    trapped_value_sum: float = 0.0
    budget_bound_cycles: int = 0
    cycles: int = 0
    daily: list[dict] = field(default_factory=list)
    #: Per-SKU demanded and served units.  A shop is not only its best sellers:
    #: a policy can lift the unit fill rate while quietly emptying most of the
    #: shelves, and only a per-SKU view shows that.
    per_sku_demanded: dict[str, float] = field(default_factory=dict)
    per_sku_served: dict[str, float] = field(default_factory=dict)

    def sku_fill_rates(self) -> dict[str, float]:
        return {
            s: (self.per_sku_served.get(s, 0.0) / d)
            for s, d in self.per_sku_demanded.items() if d > 0
        }

    @property
    def median_sku_fill(self) -> float:
        rates = sorted(self.sku_fill_rates().values())
        if not rates:
            return float("nan")
        mid = len(rates) // 2
        return rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2

    @property
    def abandoned_skus(self) -> int:
        """SKUs that were wanted but served less than a tenth of the time."""
        return sum(1 for r in self.sku_fill_rates().values() if r < 0.10)

    @property
    def well_served_skus(self) -> int:
        return sum(1 for r in self.sku_fill_rates().values() if r >= 0.90)

    @property
    def fill_rate(self) -> float:
        return self.served / self.demanded if self.demanded else float("nan")

    @property
    def stockout_rate(self) -> float:
        return self.stockout_sku_days / self.sku_days if self.sku_days else float("nan")

    @property
    def mean_inventory_value(self) -> float:
        return self.on_hand_value_sum / self.days if self.days else float("nan")

    @property
    def mean_trapped_value(self) -> float:
        return self.trapped_value_sum / self.days if self.days else float("nan")

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "days": self.days,
            "fill_rate": round(self.fill_rate, 4),
            "median_sku_fill_rate": round(self.median_sku_fill, 4),
            "skus_well_served": self.well_served_skus,
            "skus_abandoned": self.abandoned_skus,
            "skus_tracked": len(self.sku_fill_rates()),
            "stockout_sku_day_rate": round(self.stockout_rate, 4),
            "units_demanded": round(self.demanded, 1),
            "units_lost": round(self.lost, 1),
            "mean_inventory_value": round(self.mean_inventory_value, 2),
            "mean_inventory_units": round(self.on_hand_units_sum / self.days, 1) if self.days else None,
            "units_ordered": round(self.ordered_units, 1),
            "mean_cost_per_unit_ordered": (
                round(self.ordered_value / self.ordered_units, 4) if self.ordered_units else None
            ),
            "mean_trapped_value": round(self.mean_trapped_value, 2),
            "trapped_share_of_inventory": (
                round(self.mean_trapped_value / self.mean_inventory_value, 4)
                if self.mean_inventory_value else None
            ),
            "ordered_value": round(self.ordered_value, 2),
            "orders_placed": self.orders,
            "cycles": self.cycles,
            "budget_bound_cycles": self.budget_bound_cycles,
        }


def simulate(panel: pd.DataFrame, policy: Policy, config: SimConfig) -> SimResult:
    """Run one policy over the whole panel.

    ``panel`` is the dense (sku_id, day, units, unit_price) frame.  The first
    ``warmup_days`` are history only: no ordering and no scoring happen there,
    so every policy starts with the same amount to learn from.
    """
    pivot = panel.pivot_table(index="day", columns="sku_id", values="units", fill_value=0.0)
    pivot = pivot.sort_index()
    skus = list(pivot.columns)
    demand = pivot.to_numpy(dtype=float)
    days = list(pivot.index)

    costs = (
        panel.groupby("sku_id")["unit_price"].median().reindex(skus).fillna(0.0).to_dict()
    )

    warm = min(config.warmup_days, len(days) - config.review_days - 1)
    if warm < 30:
        raise ValueError("not enough days to warm up and then simulate")

    warm_mean = demand[:warm].mean(axis=0)
    on_hand = {s: float(warm_mean[i] * config.opening_cover_days) for i, s in enumerate(skus)}
    on_order: dict[str, float] = {s: 0.0 for s in skus}
    arrivals: dict[int, dict[str, float]] = {}

    daily_cogs = float(sum(warm_mean[i] * costs[s] for i, s in enumerate(skus)))
    budget = daily_cogs * config.budget_days_of_cogs

    result = SimResult(policy=getattr(policy, "name", "policy"), days=len(days) - warm)

    for t in range(warm, len(days)):
        for sku, qty in arrivals.pop(t, {}).items():
            on_hand[sku] += qty
            on_order[sku] = max(0.0, on_order[sku] - qty)

        # Serve the day's real demand from what is on the shelf.
        for i, sku in enumerate(skus):
            want = float(demand[t, i])
            have = on_hand[sku]
            got = min(want, have)
            on_hand[sku] = have - got
            result.demanded += want
            result.served += got
            result.lost += want - got
            result.sku_days += 1
            result.per_sku_demanded[sku] = result.per_sku_demanded.get(sku, 0.0) + want
            result.per_sku_served[sku] = result.per_sku_served.get(sku, 0.0) + got
            if want > 0 and got < want:
                result.stockout_sku_days += 1

        inv_value = sum(on_hand[s] * costs[s] for s in skus)
        result.on_hand_value_sum += inv_value
        result.on_hand_units_sum += sum(on_hand[s] for s in skus)

        # Cash trapped: value sitting in SKUs that have not sold recently.
        lo = max(0, t - config.slow_mover_days)
        recent = demand[lo:t].sum(axis=0)
        result.trapped_value_sum += float(
            sum(on_hand[s] * costs[s] for i, s in enumerate(skus) if recent[i] <= 0)
        )

        if (t - warm) % config.review_days == 0:
            history = {s: demand[:t, i] for i, s in enumerate(skus)}
            wanted = policy(history, dict(on_hand), dict(on_order), costs, budget)
            spend = 0.0
            placed = 0
            for sku, qty in wanted.items():
                if qty <= 0:
                    continue
                cost = qty * costs[sku]
                spend += cost
                result.ordered_units += qty
                placed += 1
                on_order[sku] += qty
                arrivals.setdefault(t + config.lead_time_days, {})
                arrivals[t + config.lead_time_days][sku] = (
                    arrivals[t + config.lead_time_days].get(sku, 0.0) + qty
                )
            result.ordered_value += spend
            result.orders += placed
            result.cycles += 1
            if spend > budget * 0.999:
                result.budget_bound_cycles += 1

    return result


def summarise(results: list[SimResult], budget_note: str = "") -> dict:
    """Side-by-side comparison, with the trade-off made explicit."""
    best_fill = max((r.fill_rate for r in results), default=float("nan"))
    return {
        "budget_note": budget_note,
        "policies": [r.to_dict() for r in results],
        "best_fill_rate": round(best_fill, 4),
        "note": (
            "fill rate and inventory value move together; a policy is only "
            "better if it raises fill rate without raising held value, or "
            "holds less for the same fill rate"
        ),
    }
