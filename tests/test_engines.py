"""Audit, cash-trap and planner behaviour, including the cases where the right
answer is to say nothing.
"""

from datetime import date, timedelta
from decimal import Decimal

from stockmate.engines.bill_auditor import PriceHistory, audit_bill
from stockmate.engines.cash_trap import CashTrapConfig, scan_cash_traps
from stockmate.engines.planner import (
    PlanDecision, allocate_under_budget, build_plan, plan_sku, reorder_point,
)
from stockmate.models import Bill, BillLine, StockItem
from stockmate.money import Money
from stockmate.store.ledger import InMemoryLedger, guard


def rupees(x):
    return Money.from_rupees(x)


def bill_at(price: str, qty=10, bill_id="b") -> Bill:
    up = rupees(price)
    return Bill(
        bill_id=bill_id, supplier_name="Anand Distributors",
        lines=[BillLine(index=0, description="Sona Masoori Rice 5kg",
                        quantity=Decimal(str(qty)), unit_price=up,
                        line_total=up * Decimal(str(qty)))],
    )


def history_at(price: str, times=3) -> PriceHistory:
    h = PriceHistory()
    for i in range(times):
        h.add_bill(bill_at(price, bill_id=f"past-{i}"))
    return h


# --------------------------------------------------------------- bill auditor

def test_price_creep_is_found_and_valued():
    findings, ledger = audit_bill(bill_at("348.00"), history_at("310.00"))
    assert len(findings) == 1
    assert findings[0].amount == rupees("380.00")        # 38 x 10
    guard(findings, ledger)                              # every amount is backed


def test_a_steady_price_raises_nothing():
    findings, _ = audit_bill(bill_at("310.00"), history_at("310.00"))
    assert findings == []


def test_a_price_fall_raises_nothing():
    findings, _ = audit_bill(bill_at("290.00"), history_at("310.00"))
    assert findings == []


def test_a_rise_under_the_threshold_is_ignored():
    findings, _ = audit_bill(bill_at("318.00"), history_at("310.00"))   # 2.6 per cent
    assert findings == []


def test_a_large_percentage_on_a_trivial_amount_is_ignored():
    # Twenty per cent of two rupees on one unit is not worth an owner's day.
    h = PriceHistory()
    for i in range(3):
        up = rupees("2.00")
        h.add_bill(Bill(bill_id=f"p{i}", supplier_name="S",
                        lines=[BillLine(index=0, description="Matchbox",
                                        quantity=Decimal(1), unit_price=up,
                                        line_total=up)]))
    up = rupees("2.40")
    now = Bill(bill_id="n", supplier_name="S",
               lines=[BillLine(index=0, description="Matchbox", quantity=Decimal(1),
                               unit_price=up, line_total=up)])
    assert audit_bill(now, h)[0] == []


def test_one_prior_purchase_is_not_a_reference_price():
    h = PriceHistory()
    h.add_bill(bill_at("310.00", bill_id="only"))
    assert audit_bill(bill_at("400.00"), h)[0] == []


def test_the_reference_resists_one_bad_record():
    # A median, not a mean: a single mistyped bill must not move the baseline.
    h = history_at("310.00", times=4)
    h.add_bill(bill_at("31000.00", bill_id="typo"))
    ref, n = h.reference_price("Anand Distributors", "Sona Masoori Rice 5kg")
    assert ref == rupees("310.00")
    assert n == 5


# ------------------------------------------------------------------ cash trap

def item(sku="X", on_hand="10", cost="100.00", **kw) -> StockItem:
    return StockItem(sku_id=sku, name=sku, on_hand=Decimal(on_hand),
                     unit_cost=rupees(cost), **kw)


def test_a_fast_mover_is_not_a_trap():
    traps, _ = scan_cash_traps([item(on_hand="20")], {"X": 5.0})
    assert traps.findings == []


def test_slow_cover_is_valued_at_the_excess_only():
    traps, ledger = scan_cash_traps([item(on_hand="100", cost="50.00")], {"X": 1.0})
    assert len(traps.findings) == 1
    assert traps.findings[0].kind == "slow_mover"
    assert traps.findings[0].amount == rupees("2750.00")     # (100 - 45) x 50
    guard(traps.findings, ledger)


def test_an_item_with_no_demand_is_dead_not_slow():
    traps, _ = scan_cash_traps([item(on_hand="30")], {"X": 0.0})
    assert traps.findings[0].kind == "dead_stock"
    assert traps.dead_value == rupees("3000.00")


def test_expiry_counts_only_what_will_not_sell_in_time():
    today = date(2026, 11, 24)
    it = item(on_hand="100", cost="10.00", expiry=today + timedelta(days=10))
    traps, _ = scan_cash_traps([it], {"X": 2.0}, today=today)
    assert traps.findings[0].kind == "expiry_risk"
    assert traps.findings[0].amount == rupees("800.00")       # 100 - 20 sellable


def test_small_values_are_below_the_noise_floor():
    traps, _ = scan_cash_traps([item(on_hand="1", cost="10.00")], {"X": 0.0})
    assert traps.findings == []


# -------------------------------------------------------------------- planner

def test_reorder_point_covers_lead_time_plus_review():
    level, safety = reorder_point(10.0, 0.0, lead_time_days=3, review_days=7)
    assert level == 100.0
    assert safety == 0.0


def test_variability_raises_the_buffer():
    _, calm = reorder_point(10.0, 0.0, 3, 7)
    _, choppy = reorder_point(10.0, 4.0, 3, 7)
    assert choppy > calm


def test_enough_stock_means_do_not_buy():
    s = plan_sku(item(on_hand="200", cost="10.00"), 1.0, 0.0)
    assert s.decision is PlanDecision.DONT_BUY
    assert s.suggest_qty == 0


def test_an_empty_shelf_means_buy():
    s = plan_sku(item(on_hand="0", cost="10.00"), 5.0, 0.0)
    assert s.decision is PlanDecision.BUY
    assert s.suggest_qty > 0


def test_orders_round_up_to_whole_packs():
    s = plan_sku(item(on_hand="0", cost="10.00", pack_size=Decimal(12)), 5.0, 0.0)
    assert Decimal(s.suggest_qty) % 12 == 0


def test_a_tight_budget_produces_buy_less_not_overspend():
    items = [item(sku=f"S{i}", on_hand="0", cost="100.00") for i in range(10)]
    forecasts = {f"S{i}": (5.0, 1.0) for i in range(10)}
    budget = rupees("2000.00")
    result, ledger = build_plan(items, forecasts, budget)
    assert result.spend.paise <= budget.paise
    assert result.cut_count > 0
    assert any(s.decision is PlanDecision.BUY_LESS or s.decision is PlanDecision.DONT_BUY
               for s in result.suggestions)
    guard([], ledger)


def test_a_generous_budget_is_left_alone():
    items = [item(sku="S", on_hand="0", cost="10.00")]
    result, _ = build_plan(items, {"S": (1.0, 0.0)}, rupees("100000.00"))
    assert result.method == "unconstrained"
    assert result.cut_count == 0


def test_the_greedy_fallback_also_respects_the_budget():
    items = [item(sku=f"S{i}", on_hand="0", cost="100.00") for i in range(8)]
    suggestions = [plan_sku(i, 5.0, 1.0) for i in items]
    budget = rupees("1500.00")
    result = allocate_under_budget(suggestions, budget, prefer_solver=False)
    assert result.method == "greedy"
    assert result.spend.paise <= budget.paise
