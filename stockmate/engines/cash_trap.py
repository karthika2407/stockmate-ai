"""Cash-Trap Radar: the money that is already spent and not coming back soon.

A shop's working capital does not usually disappear through theft or a bad
month.  It goes quietly, a few hundred rupees at a time, into stock that seemed
sensible to buy and then did not move.  The owner cannot see it because it is
not a transaction; it is the absence of one.  Nothing on any report says "this
is forty days of your cash sitting on the bottom shelf".

This engine says exactly that, in rupees, for three separate traps:

  * slow movers, where cover runs far past any sensible reorder horizon;
  * expiry risk, where stock will not sell before it has to be written off;
  * dead stock, where nothing has sold for long enough that the demand estimate
    itself is no longer believable.

Each finding carries the computation that produced its figure, so an owner who
disagrees can be shown the working rather than asked to accept a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from ..models import Finding, Severity, StockItem
from ..money import Money
from ..store.ledger import InMemoryLedger, Ledger

__all__ = ["CashTrapConfig", "CashTrapSummary", "scan_cash_traps"]


@dataclass
class CashTrapConfig:
    #: Cover beyond this many days is capital doing nothing useful.
    slow_cover_days: float = 45.0
    #: No sale for this long and the item is treated as dead rather than slow.
    dead_days: int = 60
    #: Flag expiry when stock will not clear before this many days remain.
    expiry_horizon_days: int = 30
    #: Ignore anything smaller than this; a shop cannot act on eleven rupees.
    min_value: Money = Money.from_rupees("50.00")


@dataclass
class CashTrapSummary:
    findings: list[Finding]
    slow_value: Money
    expiry_value: Money
    dead_value: Money

    @property
    def total(self) -> Money:
        return self.slow_value + self.expiry_value + self.dead_value

    def to_dict(self) -> dict:
        return {
            "slow_value_paise": self.slow_value.paise,
            "expiry_value_paise": self.expiry_value.paise,
            "dead_value_paise": self.dead_value.paise,
            "total_paise": self.total.paise,
            "findings": [f.to_dict() for f in self.findings],
        }


def scan_cash_traps(
    items: list[StockItem],
    forecasts: dict[str, float],
    *,
    today: date | None = None,
    config: CashTrapConfig | None = None,
    ledger: Ledger | None = None,
) -> tuple[CashTrapSummary, Ledger]:
    """Find the capital that is stuck, and say how much and why.

    ``forecasts`` maps a SKU to its expected daily demand.  A SKU with no
    forecast is reported as dead rather than silently skipped, because an item
    nobody can forecast is usually an item nobody is buying.
    """
    today = today or date.today()
    config = config or CashTrapConfig()
    ledger = InMemoryLedger() if ledger is None else ledger

    findings: list[Finding] = []
    slow = expiry = dead = 0

    for item in items:
        if item.on_hand <= 0:
            continue
        value = item.value_locked()
        if value.paise < config.min_value.paise:
            continue

        rate = forecasts.get(item.sku_id, 0.0)
        idle_days = (today - item.last_sold).days if item.last_sold else None

        # Dead stock first: if nothing has sold, cover is infinite and the
        # slow-mover arithmetic would divide by a rate that means nothing.
        if rate <= 1e-9 or (idle_days is not None and idle_days >= config.dead_days):
            cid = ledger.record(
                kind="dead_stock",
                formula="unit cost x units on hand",
                inputs={
                    "unit_cost": item.unit_cost,
                    "on_hand": str(item.on_hand),
                    "days_since_last_sale": idle_days,
                    "forecast_daily": round(rate, 4),
                },
                output=value,
                sku_id=item.sku_id,
            )
            dead += value.paise
            findings.append(Finding(
                finding_id=f"trap:dead:{item.sku_id}",
                kind="dead_stock",
                title=f"{item.name} has stopped selling",
                detail=(
                    f"{item.on_hand} units are on the shelf, worth {value.format()}. "
                    + (f"Nothing has sold for {idle_days} days. "
                       if idle_days is not None else "There is no recent sale on record. ")
                    + "Clearing this at a discount returns cash that is currently idle."
                ),
                amount=value, bucket="released", severity=Severity.HIGH,
                sku_id=item.sku_id, computation_id=cid, checks=["TRAP.DEAD"],
            ))
            continue

        cover_days = float(item.on_hand) / rate

        if item.expiry is not None:
            days_left = (item.expiry - today).days
            if days_left <= config.expiry_horizon_days:
                sellable = Decimal(str(min(float(item.on_hand), rate * max(days_left, 0))))
                at_risk_units = item.on_hand - sellable
                if at_risk_units > 0:
                    at_risk = item.unit_cost * at_risk_units
                    if at_risk.paise >= config.min_value.paise:
                        cid = ledger.record(
                            kind="expiry_risk",
                            formula="unit cost x (units on hand - units expected to sell before expiry)",
                            inputs={
                                "unit_cost": item.unit_cost,
                                "on_hand": str(item.on_hand),
                                "days_to_expiry": days_left,
                                "forecast_daily": round(rate, 4),
                                "expected_to_sell": str(sellable),
                            },
                            output=at_risk,
                            sku_id=item.sku_id,
                        )
                        expiry += at_risk.paise
                        findings.append(Finding(
                            finding_id=f"trap:expiry:{item.sku_id}",
                            kind="expiry_risk",
                            title=f"{item.name} will not clear before it expires",
                            detail=(
                                f"{item.on_hand} units expire in {days_left} days. At "
                                f"{rate:.1f} a day only about {sellable} will sell, "
                                f"leaving {at_risk_units} units worth {at_risk.format()}. "
                                "Discounting now recovers more than writing it off later."
                            ),
                            amount=at_risk, bucket="protected", severity=Severity.HIGH,
                            sku_id=item.sku_id, computation_id=cid, checks=["TRAP.EXPIRY"],
                        ))
                        continue

        if cover_days > config.slow_cover_days:
            excess_units = item.on_hand - Decimal(str(rate * config.slow_cover_days))
            if excess_units <= 0:
                continue
            excess = item.unit_cost * excess_units
            if excess.paise < config.min_value.paise:
                continue
            cid = ledger.record(
                kind="slow_mover",
                formula=(
                    "unit cost x (units on hand - units that sell within the "
                    "slow-mover horizon)"
                ),
                inputs={
                    "unit_cost": item.unit_cost,
                    "on_hand": str(item.on_hand),
                    "forecast_daily": round(rate, 4),
                    "cover_days": round(cover_days, 1),
                    "horizon_days": config.slow_cover_days,
                },
                output=excess,
                sku_id=item.sku_id,
            )
            slow += excess.paise
            findings.append(Finding(
                finding_id=f"trap:slow:{item.sku_id}",
                kind="slow_mover",
                title=f"{item.name} is {cover_days:.0f} days of stock",
                detail=(
                    f"{item.on_hand} units at {rate:.1f} a day is {cover_days:.0f} days "
                    f"of cover. Everything beyond {config.slow_cover_days:.0f} days, "
                    f"about {excess.format()}, is cash that could be buying something "
                    "that moves."
                ),
                amount=excess, bucket="released", severity=Severity.MEDIUM,
                sku_id=item.sku_id, computation_id=cid, checks=["TRAP.SLOW"],
            ))

    findings.sort(key=lambda f: -(f.amount.paise if f.amount else 0))
    return CashTrapSummary(findings, Money(slow), Money(expiry), Money(dead)), ledger
