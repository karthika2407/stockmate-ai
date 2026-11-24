"""Duplicate detection, within a bill and across bills.

A line charged twice is arithmetically perfect: the quantities multiply, the
subtotal adds up and the total closes.  Nothing in the arithmetic checks can see
it, which is exactly why it is worth a rule of its own.

The same is true one level up.  A supplier who sends the same invoice twice, or
a shop that photographs one bill on two days, produces a second document that is
internally flawless.  Only memory catches it.

Both rules are deliberately cautious.  Buying two crates of the same oil on one
bill is ordinary, and a shop that gets accused of double payment every week will
stop reading the alerts.  So a repeat is raised for confirmation, at medium
severity, with the matching lines shown; it is never asserted as an overcharge.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..models import Bill, Check, CheckStatus, Severity

__all__ = ["check_duplicate_lines", "check_duplicate_bill", "normalise_description"]

_WS = re.compile(r"[^a-z0-9]+")


def normalise_description(text: str) -> str:
    """Fold case, punctuation and spacing so two printings of one item match."""
    return _WS.sub(" ", (text or "").lower()).strip()


def check_duplicate_lines(bill: Bill) -> list[Check]:
    """Lines on this bill that charge for the same thing at the same price."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for line in bill.lines:
        if line.line_total is None:
            continue
        key = (
            normalise_description(line.description),
            str(line.quantity),
            line.unit_price.paise if line.unit_price else None,
            line.line_total.paise,
        )
        if not key[0]:
            continue                      # an unreadable description matches everything
        groups[key].append(line.index)

    checks: list[Check] = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        repeats = len(indices) - 1
        first = bill.lines[indices[0]]
        amount = first.line_total * repeats
        human = ", ".join(str(i + 1) for i in indices)
        checks.append(
            Check(
                rule_id="DUP.LINE",
                status=CheckStatus.FAIL,
                message=(
                    f"Lines {human} charge for the same item at the same price "
                    f"({first.description}, {first.quantity} at "
                    f"{first.unit_price.format() if first.unit_price else 'unknown'}). "
                    f"If that is one delivery charged twice, "
                    f"{amount.format()} is at stake. Please confirm."
                ),
                expected=f"one charge for {first.description}",
                observed=f"{len(indices)} identical charges",
                delta=amount,
                line_index=indices[-1],
                severity=Severity.MEDIUM,
                evidence={
                    "line_indices": indices,
                    "repeat_count": repeats,
                    "requires_owner_confirmation": True,
                },
            )
        )

    if not checks:
        checks.append(
            Check(
                rule_id="DUP.LINE",
                status=CheckStatus.PASS,
                message="No line on this bill repeats another at the same quantity and price.",
                severity=Severity.INFO,
            )
        )
    return checks


def check_duplicate_bill(bill: Bill, seen_hashes: dict[str, str]) -> Check:
    """Has a bill with this supplier, number, date and total been seen before?

    ``seen_hashes`` maps a content hash to the id of the bill that produced it.
    The caller owns that store, so the rule works the same against a database or
    against an in-memory dictionary in a test.
    """
    key = bill.content_hash()
    previous = seen_hashes.get(key)
    if previous and previous != bill.bill_id:
        return Check(
            rule_id="DUP.BILL",
            status=CheckStatus.FAIL,
            message=(
                f"This bill matches one already recorded ({previous}): same "
                f"supplier, same invoice number, same date and same total"
                + (f" of {bill.grand_total.format()}" if bill.grand_total else "")
                + ". Check whether it has already been paid."
            ),
            expected="a bill not seen before",
            observed=f"identical to {previous}",
            delta=bill.grand_total,
            severity=Severity.HIGH,
            evidence={
                "content_hash": key,
                "previous_bill_id": previous,
                "requires_owner_confirmation": True,
            },
        )
    return Check(
        rule_id="DUP.BILL",
        status=CheckStatus.PASS,
        message="This bill has not been recorded before.",
        severity=Severity.INFO,
        evidence={"content_hash": key},
    )
