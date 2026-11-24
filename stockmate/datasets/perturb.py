"""Controlled, labelled faults injected into real receipts.

There is no public corpus of supplier bills annotated with the overcharges they
contain, and there is unlikely ever to be one: a shop that spots an overcharge
disputes it, and nobody publishes the file.  The standard way to measure a
detector without such a corpus is to take genuine documents and introduce faults
whose location and size are known, then ask what the detector finds.

That is what this module does.  It is an evaluation instrument, not part of the
running system, and two properties keep it honest:

  * Only receipts whose own arithmetic already closes are perturbed.  Injecting
    a fault into a receipt that was already inconsistent would make the result
    unreadable, since the detector would be right for the wrong reason.
  * Faults are drawn with a seeded generator derived from the receipt id, so
    the same receipt always receives the same fault and the whole benchmark
    replays exactly.  Nothing here uses Python's ``hash``, which is salted per
    process and would silently make every run different.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from random import Random

from ..models import Bill, BillLine
from ..money import Money

__all__ = ["FaultKind", "InjectedFault", "seeded_random", "inject_fault", "FAULT_KINDS"]


class FaultKind(str, Enum):
    """The overcharge patterns a shop actually meets on supplier bills."""

    PRICE_CREEP = "price_creep"              # unit price quietly raised
    LINE_MATH = "line_math"                  # qty x price does not equal the line total
    SUBTOTAL_INFLATED = "subtotal_inflated"  # subtotal above the sum of the lines
    TOTAL_INFLATED = "total_inflated"        # grand total above the rebuilt total
    DUPLICATE_LINE = "duplicate_line"        # the same item charged twice
    DROPPED_DISCOUNT = "dropped_discount"    # an agreed discount not applied


FAULT_KINDS = tuple(FaultKind)


def seeded_random(*parts: object) -> Random:
    """A generator whose seed depends only on the given parts.

    blake2b rather than ``hash``: the built-in is salted per interpreter run, so
    a benchmark seeded from it produces different data every time it is run and
    the published numbers cannot be reproduced.
    """
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return Random(int.from_bytes(digest, "big"))


@dataclass
class InjectedFault:
    """What was done to a receipt, so the detector's answer can be scored."""

    kind: FaultKind
    line_index: int | None
    amount: Money
    detail: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "line_index": self.line_index,
            "amount_paise": self.amount.paise,
            "detail": self.detail,
        }


def _priced_lines(bill: Bill) -> list[BillLine]:
    return [
        l for l in bill.lines
        if l.line_total is not None and l.unit_price is not None and l.quantity
    ]


def inject_fault(
    bill: Bill,
    kind: FaultKind,
    *,
    rng: Random | None = None,
) -> tuple[Bill, InjectedFault] | None:
    """Return a copy of ``bill`` carrying one labelled fault.

    Returns None when the receipt cannot carry that fault, for example a
    single-line receipt asked for a duplicate.  Callers must treat None as
    "not applicable" and exclude it from the denominator rather than counting
    it as a miss.
    """
    rng = rng or seeded_random(bill.bill_id, kind.value)
    out = copy.deepcopy(bill)
    lines = _priced_lines(out)
    if not lines:
        return None

    target = rng.choice(lines)

    if kind is FaultKind.PRICE_CREEP:
        # A credible creep: between 3% and 15%, the band that gets missed.
        pct = Decimal(str(round(rng.uniform(0.03, 0.15), 4)))
        before = target.line_total
        target.unit_price = target.unit_price * (1 + pct)
        target.line_total = target.unit_price * target.quantity
        delta = target.line_total - before
        if delta.is_zero():
            return None
        _reflow_totals(out, delta)
        return out, InjectedFault(
            kind, target.index, delta,
            f"unit price raised {float(pct) * 100:.1f}% on line {target.index + 1}",
        )

    if kind is FaultKind.LINE_MATH:
        delta = Money.from_rupees(Decimal(str(round(rng.uniform(5, 200), 2))))
        target.line_total = target.line_total + delta
        _reflow_totals(out, delta)
        return out, InjectedFault(
            kind, target.index, delta,
            f"line {target.index + 1} total raised without changing quantity or price",
        )

    if kind is FaultKind.SUBTOTAL_INFLATED:
        if out.subtotal is None:
            return None
        delta = Money.from_rupees(Decimal(str(round(rng.uniform(10, 300), 2))))
        out.subtotal = out.subtotal + delta
        if out.grand_total is not None:
            out.grand_total = out.grand_total + delta
        return out, InjectedFault(kind, None, delta, "subtotal raised above the sum of the lines")

    if kind is FaultKind.TOTAL_INFLATED:
        if out.grand_total is None:
            return None
        delta = Money.from_rupees(Decimal(str(round(rng.uniform(10, 300), 2))))
        out.grand_total = out.grand_total + delta
        return out, InjectedFault(kind, None, delta, "grand total raised above the rebuilt total")

    if kind is FaultKind.DUPLICATE_LINE:
        if len(out.lines) < 2:
            return None
        clone = copy.deepcopy(target)
        clone.index = len(out.lines)
        out.lines.append(clone)
        delta = clone.line_total
        _reflow_totals(out, delta)
        return out, InjectedFault(
            kind, clone.index, delta,
            f"line {target.index + 1} ({target.description}) charged a second time",
        )

    if kind is FaultKind.DROPPED_DISCOUNT:
        if out.discount_total is None or out.discount_total.is_zero():
            return None
        delta = out.discount_total
        out.discount_total = None
        if out.grand_total is not None:
            out.grand_total = out.grand_total + delta
        return out, InjectedFault(
            kind, None, delta, "an agreed discount was not applied to the total",
        )

    raise ValueError(f"unknown fault kind: {kind}")


def _reflow_totals(bill: Bill, delta: Money) -> None:
    """Push a line-level change through the subtotal and the grand total.

    Without this the injected fault would show up as three failed checks instead
    of one, and the detector would look better than it is: a bill whose totals
    still match the original lines is an easier target than one a supplier
    actually issued.
    """
    if bill.subtotal is not None:
        bill.subtotal = bill.subtotal + delta
    if bill.grand_total is not None:
        bill.grand_total = bill.grand_total + delta
