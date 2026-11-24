"""The replenishment planner: Buy, Buy Less, or Don't Buy, inside a cash limit.

A forecast is not a decision.  The decision a shop owner actually faces on a
Tuesday morning is which of forty things to reorder with the eleven thousand
rupees in the till, knowing that whatever is not bought may sell out and
whatever is bought may sit.  The planner turns a demand rate into that decision
and, crucially, respects the cash limit rather than pretending it does not
exist.

Two parts:

  * a per-SKU target, which is ordinary inventory control: cover the demand
    expected over the lead time and the review period, plus a safety buffer
    sized from how variable that demand has been;
  * an allocation across SKUs, which is where the cash limit bites.  When the
    targets cost more than the shop has, something must be cut, and the planner
    cuts by expected unmet demand per rupee rather than alphabetically or by
    whoever shouted loudest.

The allocation is a bounded knapsack and is solved as a linear program when PuLP
is available, with a greedy fallback that is used when it is not.  Both are
reported, because the greedy is what runs if the solver is missing and an owner
deserves to know which one produced their order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from ..models import OrderSuggestion, PlanDecision, StockItem
from ..money import Money
from ..store.ledger import InMemoryLedger, Ledger

__all__ = [
    "SERVICE_Z",
    "PlanResult",
    "reorder_point",
    "plan_sku",
    "allocate_under_budget",
    "build_plan",
]

#: Normal quantiles for the service levels an owner might ask for.  Stated as a
#: table rather than computed so the chosen level is visible in the code review
#: and in the report, instead of hidden inside a distribution call.
SERVICE_Z = {
    0.50: 0.0000,
    0.80: 0.8416,
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
}


def reorder_point(
    forecast_daily: float,
    demand_sd: float,
    lead_time_days: int,
    review_days: int,
    service_level: float = 0.95,
) -> tuple[float, float]:
    """Level at which to reorder, and the safety stock inside it.

    Cover is taken over the lead time plus the review period, because an order
    placed today is not reviewed again until the next review: the shop is
    exposed for both stretches, not just the delivery wait.  The buffer scales
    with the square root of that exposure, which is the standard result for
    independent daily demand and is the assumption most likely to be wrong here;
    real demand is autocorrelated around festivals, so the buffer understates
    the risk precisely when it matters most.  That limitation is real and is
    stated rather than hidden.
    """
    exposure = max(lead_time_days + review_days, 1)
    z = SERVICE_Z.get(round(service_level, 3), SERVICE_Z[0.95])
    cycle_demand = forecast_daily * exposure
    safety = z * demand_sd * math.sqrt(exposure)
    return cycle_demand + safety, safety


def plan_sku(
    item: StockItem,
    forecast_daily: float,
    demand_sd: float,
    *,
    review_days: int = 7,
    service_level: float = 0.95,
    on_order: Decimal = Decimal(0),
    ledger: Ledger | None = None,
) -> OrderSuggestion:
    """Decide what, if anything, to order for one SKU, ignoring the cash limit."""
    ledger = InMemoryLedger() if ledger is None else ledger
    target, safety = reorder_point(
        forecast_daily, demand_sd, item.lead_time_days, review_days, service_level
    )
    position = float(item.on_hand) + float(on_order)
    shortfall = target - position

    cover_days = (position / forecast_daily) if forecast_daily > 1e-9 else None

    if shortfall <= 0:
        decision = PlanDecision.DONT_BUY
        qty = Decimal(0)
        reason = (
            f"{item.on_hand} in stock"
            + (f" plus {on_order} on the way" if on_order else "")
            + f" covers the next {item.lead_time_days + review_days} days at the "
            f"current rate of {forecast_daily:.1f} a day."
        )
    else:
        pack = float(item.pack_size) if item.pack_size else 1.0
        packs = math.ceil(shortfall / pack) if pack > 0 else math.ceil(shortfall)
        qty = Decimal(str(packs)) * item.pack_size
        decision = PlanDecision.BUY
        reason = (
            f"At {forecast_daily:.1f} a day this needs about {target:.0f} to cover "
            f"{item.lead_time_days} days of delivery and {review_days} days until "
            f"the next review, including {safety:.0f} as a buffer. "
            f"There {'is' if position == 1 else 'are'} {position:.0f} on hand."
        )

    cash = item.unit_cost * qty
    cid = ledger.record(
        kind="order_cash",
        formula="unit cost x quantity to order",
        inputs={
            "unit_cost": item.unit_cost,
            "quantity": str(qty),
            "forecast_daily": round(forecast_daily, 4),
            "target_level": round(target, 2),
            "safety_stock": round(safety, 2),
            "on_hand": str(item.on_hand),
            "on_order": str(on_order),
        },
        output=cash,
        sku_id=item.sku_id,
    )

    return OrderSuggestion(
        sku_id=item.sku_id,
        name=item.name,
        decision=decision,
        suggest_qty=qty,
        reorder_point=Decimal(str(round(target, 2))),
        cover_days=cover_days,
        forecast_daily=forecast_daily,
        cash_required=cash,
        reason=reason,
        computation_id=cid,
    )


@dataclass
class PlanResult:
    suggestions: list[OrderSuggestion]
    budget: Money
    spend: Money
    method: str
    cut_count: int

    @property
    def headroom(self) -> Money:
        return self.budget - self.spend

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "budget_paise": self.budget.paise,
            "spend_paise": self.spend.paise,
            "headroom_paise": self.headroom.paise,
            "lines": len(self.suggestions),
            "lines_cut": self.cut_count,
            "suggestions": [s.to_dict() for s in self.suggestions],
        }


def allocate_under_budget(
    suggestions: list[OrderSuggestion],
    budget: Money,
    *,
    prefer_solver: bool = True,
) -> PlanResult:
    """Trim the wanted order down to what the shop can pay for.

    Value is expected unmet demand avoided, approximated as the forecast daily
    rate times the units bought: a unit of something that sells twice a day is
    worth more than a unit of something that sells twice a month.  Quantities
    may be reduced but not increased, which is why the result can say Buy Less.
    """
    wanted = [s for s in suggestions if s.suggest_qty > 0]
    if not wanted:
        return PlanResult(list(suggestions), budget, Money.zero(), "no_order", 0)

    total = Money(sum(s.cash_required.paise for s in wanted))
    if total.paise <= budget.paise:
        return PlanResult(list(suggestions), budget, total, "unconstrained", 0)

    chosen: dict[str, Decimal] | None = None
    method = "greedy"
    if prefer_solver:
        chosen = _solve_lp(wanted, budget)
        if chosen is not None:
            method = "linear_program"
    if chosen is None:
        chosen = _solve_greedy(wanted, budget)

    out: list[OrderSuggestion] = []
    spend = 0
    cuts = 0
    for s in suggestions:
        if s.suggest_qty <= 0:
            out.append(s)
            continue
        qty = chosen.get(s.sku_id, Decimal(0))
        unit = Money(round(s.cash_required.paise / float(s.suggest_qty))) if s.suggest_qty else Money.zero()
        cash = Money(int(round(unit.paise * float(qty))))
        if qty == s.suggest_qty:
            out.append(s)
            spend += s.cash_required.paise
            continue
        cuts += 1
        decision = PlanDecision.BUY_LESS if qty > 0 else PlanDecision.DONT_BUY
        note = (
            f" Cut from {s.suggest_qty} to {qty} to stay inside the "
            f"{budget.format()} budget."
            if qty > 0 else
            f" Held back entirely this cycle: the {budget.format()} budget is "
            f"better spent on faster movers."
        )
        out.append(
            OrderSuggestion(
                sku_id=s.sku_id, name=s.name, decision=decision, suggest_qty=qty,
                reorder_point=s.reorder_point, cover_days=s.cover_days,
                forecast_daily=s.forecast_daily, cash_required=cash,
                reason=s.reason + note, computation_id=s.computation_id,
            )
        )
        spend += cash.paise

    return PlanResult(out, budget, Money(spend), method, cuts)


def _solve_lp(wanted: list[OrderSuggestion], budget: Money) -> dict[str, Decimal] | None:
    """Bounded knapsack over order quantities. Returns None if PuLP is absent."""
    try:
        import pulp
    except ImportError:
        return None

    prob = pulp.LpProblem("replenishment", pulp.LpMaximize)
    vars_: dict[str, pulp.LpVariable] = {}
    for s in wanted:
        vars_[s.sku_id] = pulp.LpVariable(
            f"q_{s.sku_id}", lowBound=0, upBound=float(s.suggest_qty), cat="Integer"
        )

    def unit_cost(s: OrderSuggestion) -> float:
        return s.cash_required.paise / float(s.suggest_qty) if s.suggest_qty else 0.0

    prob += pulp.lpSum(max(s.forecast_daily, 1e-6) * vars_[s.sku_id] for s in wanted)
    prob += pulp.lpSum(unit_cost(s) * vars_[s.sku_id] for s in wanted) <= budget.paise

    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
    except Exception:
        return None
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return {sku: Decimal(str(int(round(v.value() or 0)))) for sku, v in vars_.items()}


def _solve_greedy(wanted: list[OrderSuggestion], budget: Money) -> dict[str, Decimal]:
    """Fill the budget by value per rupee, one unit batch at a time."""
    order = sorted(
        wanted,
        key=lambda s: -(
            max(s.forecast_daily, 1e-6)
            / max(s.cash_required.paise / max(float(s.suggest_qty), 1.0), 1e-6)
        ),
    )
    remaining = budget.paise
    chosen: dict[str, Decimal] = {s.sku_id: Decimal(0) for s in wanted}
    for s in order:
        unit = s.cash_required.paise / float(s.suggest_qty) if s.suggest_qty else 0
        if unit <= 0:
            continue
        affordable = int(remaining // unit)
        take = min(int(s.suggest_qty), affordable)
        if take <= 0:
            continue
        chosen[s.sku_id] = Decimal(str(take))
        remaining -= int(round(unit * take))
    return chosen


def build_plan(
    items: list[StockItem],
    forecasts: dict[str, tuple[float, float]],
    budget: Money,
    *,
    review_days: int = 7,
    service_level: float = 0.95,
    on_order: dict[str, Decimal] | None = None,
    ledger: Ledger | None = None,
) -> tuple[PlanResult, Ledger]:
    """Plan every SKU, then fit the result to the cash available."""
    ledger = InMemoryLedger() if ledger is None else ledger
    on_order = on_order or {}
    suggestions = [
        plan_sku(
            item,
            *forecasts.get(item.sku_id, (0.0, 0.0)),
            review_days=review_days,
            service_level=service_level,
            on_order=on_order.get(item.sku_id, Decimal(0)),
            ledger=ledger,
        )
        for item in items
    ]
    return allocate_under_budget(suggestions, budget), ledger
