"""The Bill Auditor: what the arithmetic cannot see.

Three of the ways a shop loses money on a supplier bill leave the arithmetic
perfectly intact.  A unit price raised five per cent still multiplies correctly.
A line charged twice still adds up.  A scheme discount that was agreed verbally
and then not applied leaves no trace on the paper at all.  Catching these needs
something the bill does not contain: what this supplier charged last time.

Every rupee figure produced here is written to the ledger first and carries the
id of the computation that produced it, so the owner can always be shown the
subtraction rather than asked to trust it.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from ..models import Bill, Finding, Severity
from ..money import Money
from ..store.ledger import InMemoryLedger, Ledger
from ..verify.duplicate import normalise_description

__all__ = ["PriceHistory", "PriceObservation", "audit_bill", "DEFAULT_CREEP_PCT"]

#: A price rise smaller than this is normal trade movement, not a finding.
DEFAULT_CREEP_PCT = Decimal("5")

#: And a rise worth less than this in rupees is not worth an owner's attention,
#: whatever its percentage. Flagging a four rupee difference trains people to
#: ignore the alerts that matter.
DEFAULT_MIN_IMPACT = Money.from_rupees("20.00")

#: Below this many past purchases the median is not a reference price.
MIN_OBSERVATIONS = 2


@dataclass(frozen=True)
class PriceObservation:
    supplier: str
    item_key: str
    unit_price: Money
    observed_on: date | None = None
    bill_id: str | None = None


@dataclass
class PriceHistory:
    """What each supplier has charged for each item, as recorded from past bills."""

    observations: dict[tuple[str, str], list[PriceObservation]] = field(default_factory=lambda: defaultdict(list))

    @staticmethod
    def key(supplier: str | None, description: str) -> tuple[str, str]:
        return ((supplier or "").strip().upper(), normalise_description(description))

    def add_bill(self, bill: Bill) -> int:
        """Record every priced line on a bill. Returns how many were recorded."""
        added = 0
        for line in bill.lines:
            if line.unit_price is None or not normalise_description(line.description):
                continue
            k = self.key(bill.supplier_name or bill.supplier_gstin, line.description)
            self.observations[k].append(
                PriceObservation(
                    supplier=k[0],
                    item_key=k[1],
                    unit_price=line.unit_price,
                    observed_on=bill.invoice_date,
                    bill_id=bill.bill_id,
                )
            )
            added += 1
        return added

    def reference_price(self, supplier: str | None, description: str) -> tuple[Money | None, int]:
        """Median of what has been paid before, and how many purchases that is.

        The median rather than the mean: one mistyped bill in the history should
        not move the reference price that every future bill is judged against.
        """
        obs = self.observations.get(self.key(supplier, description), [])
        if len(obs) < MIN_OBSERVATIONS:
            return None, len(obs)
        median = statistics.median(o.unit_price.paise for o in obs)
        return Money(int(round(median))), len(obs)

    def __len__(self) -> int:
        return sum(len(v) for v in self.observations.values())


def audit_bill(
    bill: Bill,
    history: PriceHistory,
    *,
    ledger: Ledger | None = None,
    creep_pct: Decimal = DEFAULT_CREEP_PCT,
    min_impact: Money = DEFAULT_MIN_IMPACT,
) -> tuple[list[Finding], Ledger]:
    """Compare each line against what this supplier charged before.

    Returns the findings and the ledger that backs them.  A caller that passes
    its own ledger gets the computations appended to it.
    """
    ledger = InMemoryLedger() if ledger is None else ledger
    findings: list[Finding] = []

    for line in bill.lines:
        if line.unit_price is None or line.quantity is None:
            continue
        reference, n_obs = history.reference_price(
            bill.supplier_name or bill.supplier_gstin, line.description
        )
        if reference is None or reference.is_zero():
            continue

        rise = line.unit_price - reference
        if rise.paise <= 0:
            continue

        pct = (Decimal(rise.paise) / Decimal(reference.paise)) * 100
        impact = rise * line.quantity
        if pct < creep_pct or impact.paise < min_impact.paise:
            continue

        cid = ledger.record(
            kind="price_creep",
            formula="(unit price now - median price paid before) x quantity",
            inputs={
                "unit_price_now": line.unit_price,
                "median_price_before": reference,
                "quantity": str(line.quantity),
                "past_purchases": n_obs,
                "rise_percent": f"{pct:.2f}",
            },
            output=impact,
            bill_id=bill.bill_id,
            sku_id=line.sku_id,
        )
        findings.append(
            Finding(
                finding_id=f"{bill.bill_id}:creep:{line.index}",
                kind="price_creep",
                title=f"{line.description} costs more than before",
                detail=(
                    f"This bill charges {line.unit_price.format()} each. Across "
                    f"{n_obs} earlier purchases the usual price was "
                    f"{reference.format()}, so this is {pct:.1f} per cent higher. "
                    f"On {line.quantity} units that is {impact.format()}."
                ),
                amount=impact,
                bucket="found",
                severity=Severity.HIGH if pct >= 15 else Severity.MEDIUM,
                bill_id=bill.bill_id,
                line_index=line.index,
                sku_id=line.sku_id,
                computation_id=cid,
                checks=["AUDIT.PRICE_CREEP"],
            )
        )

    return findings, ledger
