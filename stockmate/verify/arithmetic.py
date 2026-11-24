"""Arithmetic checks on a supplier bill.

These are the cheapest and most reliable findings the system produces: they need
no history, no catalogue and no model, only the numbers already printed on the
bill.  A line whose quantity times unit price does not equal its printed total is
wrong on its own terms, and that is true whoever issued it.

Every check tolerates a small rounding window, because suppliers round at
different points in the calculation and a one paisa disagreement is not a
dispute worth sending to anybody.  The window is explicit and configurable
rather than hidden in the comparisons.
"""

from __future__ import annotations

from decimal import Decimal

from ..models import Bill, Check, CheckStatus, Severity
from ..money import Money, total as money_total

__all__ = [
    "DEFAULT_TOLERANCE",
    "check_line_math",
    "check_subtotal",
    "check_grand_total",
    "run_arithmetic_checks",
]

#: Accept this much disagreement before calling a line wrong.  One rupee covers
#: the usual per-line rounding; anything larger is a real difference.
DEFAULT_TOLERANCE = Money.from_rupees("1.00")


def _within(a: Money, b: Money, tolerance: Money) -> bool:
    return abs((a - b).paise) <= tolerance.paise


def check_line_math(bill: Bill, tolerance: Money = DEFAULT_TOLERANCE) -> list[Check]:
    """quantity x unit price (less any line discount) against the printed total."""
    checks: list[Check] = []
    for line in bill.lines:
        expected = line.expected_total()
        if expected is None or line.line_total is None:
            checks.append(
                Check(
                    rule_id="ARITH.LINE",
                    status=CheckStatus.SKIPPED,
                    message=(
                        f"Line {line.index + 1} ({line.description}): not enough "
                        "printed values to check the multiplication."
                    ),
                    line_index=line.index,
                    severity=Severity.INFO,
                )
            )
            continue

        if line.discount:
            expected = expected - line.discount
        delta = line.line_total - expected
        ok = _within(line.line_total, expected, tolerance)
        checks.append(
            Check(
                rule_id="ARITH.LINE",
                status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                message=(
                    f"Line {line.index + 1} ({line.description}): "
                    f"{line.quantity} x {line.unit_price.format()}"
                    + (f" less {line.discount.format()}" if line.discount else "")
                    + f" = {expected.format()}, bill says {line.line_total.format()}."
                    + ("" if ok else f" Difference {delta.format()}.")
                ),
                expected=expected,
                observed=line.line_total,
                delta=delta,
                line_index=line.index,
                severity=Severity.HIGH if not ok else Severity.INFO,
                evidence={
                    "quantity": str(line.quantity),
                    "unit_price_paise": line.unit_price.paise,
                    "discount_paise": line.discount.paise if line.discount else 0,
                    "tolerance_paise": tolerance.paise,
                },
            )
        )
    return checks


def check_subtotal(bill: Bill, tolerance: Money | None = None) -> Check:
    """Sum of the line totals against the printed subtotal."""
    priced = [l for l in bill.lines if l.line_total is not None]
    if not priced or bill.subtotal is None:
        return Check(
            rule_id="ARITH.SUBTOTAL",
            status=CheckStatus.SKIPPED,
            message="No printed subtotal, or no priced lines, to compare.",
            severity=Severity.INFO,
        )

    # The subtotal accumulates each line's rounding, so the window grows with the
    # number of lines rather than staying at a flat rupee.
    tol = tolerance or Money.from_paise(max(100, 50 * len(priced)))
    computed = money_total(l.line_total for l in priced)
    delta = bill.subtotal - computed
    ok = _within(bill.subtotal, computed, tol)
    return Check(
        rule_id="ARITH.SUBTOTAL",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        message=(
            f"{len(priced)} lines add up to {computed.format()}; "
            f"the bill's subtotal is {bill.subtotal.format()}."
            + ("" if ok else f" Difference {delta.format()}.")
        ),
        expected=computed,
        observed=bill.subtotal,
        delta=delta,
        severity=Severity.HIGH if not ok else Severity.INFO,
        evidence={"line_count": len(priced), "tolerance_paise": tol.paise},
    )


def check_grand_total(bill: Bill, tolerance: Money | None = None) -> Check:
    """subtotal - discount + tax + round off against the printed grand total."""
    if bill.grand_total is None:
        return Check(
            rule_id="ARITH.GRAND_TOTAL",
            status=CheckStatus.SKIPPED,
            message="The bill has no printed grand total.",
            severity=Severity.INFO,
        )

    base = bill.subtotal
    if base is None:
        priced = [l for l in bill.lines if l.line_total is not None]
        if not priced:
            return Check(
                rule_id="ARITH.GRAND_TOTAL",
                status=CheckStatus.SKIPPED,
                message="No subtotal and no priced lines, so the total cannot be rebuilt.",
                severity=Severity.INFO,
            )
        base = money_total(l.line_total for l in priced)

    tax = bill.tax_total
    if tax is None:
        parts = [t for t in (bill.cgst, bill.sgst, bill.igst) if t is not None]
        tax = money_total(parts) if parts else None

    # Printed bills do not all mean the same thing by the figures they print.
    # A discount may already be inside the subtotal or be deducted after it; a
    # tax line may be inclusive or added on.  Accusing a supplier because the
    # system assumed one convention and they used another is a false alarm, and
    # false alarms are what stop an owner from using an auditor at all.  So
    # every standard reading is rebuilt, and the bill is only reported wrong
    # when none of them lands on the printed total.
    candidates: list[tuple[str, Money]] = []

    def add(label: str, value: Money) -> None:
        if bill.round_off:
            candidates.append((f"{label} plus round off", value + bill.round_off))
        candidates.append((label, value))

    add("subtotal", base)
    if tax is not None and not tax.is_zero():
        add("subtotal plus tax", base + tax)
    if bill.discount_total and not bill.discount_total.is_zero():
        add("subtotal less discount", base - bill.discount_total)
        if tax is not None and not tax.is_zero():
            add("subtotal less discount plus tax", base - bill.discount_total + tax)

    tol = tolerance or Money.from_rupees("1.00")
    match = next(
        ((label, value) for label, value in candidates if _within(bill.grand_total, value, tol)),
        None,
    )

    # When nothing matches, report against the fullest reading, which is the one
    # the owner would have worked out by hand.
    primary_label, primary_value = candidates[0]
    for label, value in candidates:
        if "discount" in label and tax is not None:
            primary_label, primary_value = label, value
            break

    if match is not None:
        label, value = match
        return Check(
            rule_id="ARITH.GRAND_TOTAL",
            status=CheckStatus.PASS,
            message=(
                f"The printed total {bill.grand_total.format()} is consistent "
                f"with the bill read as {label} ({value.format()})."
            ),
            expected=value,
            observed=bill.grand_total,
            delta=bill.grand_total - value,
            severity=Severity.INFO,
            evidence={
                "reading": label,
                "readings_tried": [l for l, _ in candidates],
                "tolerance_paise": tol.paise,
            },
        )

    delta = bill.grand_total - primary_value
    return Check(
        rule_id="ARITH.GRAND_TOTAL",
        status=CheckStatus.FAIL,
        message=(
            f"No standard reading of this bill reaches the printed total. "
            f"Read as {primary_label} it comes to {primary_value.format()}, "
            f"but the bill says {bill.grand_total.format()}, a difference of "
            f"{delta.format()}."
        ),
        expected=primary_value,
        observed=bill.grand_total,
        delta=delta,
        severity=Severity.HIGH,
        evidence={
            "readings_tried": {l: v.paise for l, v in candidates},
            "subtotal_paise": base.paise,
            "discount_paise": bill.discount_total.paise if bill.discount_total else 0,
            "tax_paise": tax.paise if tax is not None else None,
            "round_off_paise": bill.round_off.paise if bill.round_off else 0,
            "tolerance_paise": tol.paise,
        },
    )


def check_tax_split(bill: Bill) -> Check:
    """CGST and SGST should be equal halves, and should not coexist with IGST."""
    if bill.cgst is None or bill.sgst is None:
        return Check(
            rule_id="ARITH.TAX_SPLIT",
            status=CheckStatus.SKIPPED,
            message="The bill does not print CGST and SGST separately.",
            severity=Severity.INFO,
        )
    if bill.igst is not None and not bill.igst.is_zero():
        return Check(
            rule_id="ARITH.TAX_SPLIT",
            status=CheckStatus.FAIL,
            message=(
                "The bill charges IGST as well as CGST and SGST. A supply is "
                "either within the state or across it, not both."
            ),
            expected="CGST+SGST or IGST",
            observed="all three",
            severity=Severity.HIGH,
        )
    delta = bill.cgst - bill.sgst
    ok = abs(delta.paise) <= 1          # one paisa for the odd-rupee split
    return Check(
        rule_id="ARITH.TAX_SPLIT",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        message=(
            f"CGST {bill.cgst.format()} and SGST {bill.sgst.format()}"
            + (" match." if ok else f" differ by {delta.format()}, which they should not.")
        ),
        expected=bill.cgst,
        observed=bill.sgst,
        delta=delta,
        severity=Severity.MEDIUM if not ok else Severity.INFO,
    )


def run_arithmetic_checks(bill: Bill, tolerance: Money = DEFAULT_TOLERANCE) -> list[Check]:
    """Every arithmetic rule, in the order an owner would read them."""
    checks = check_line_math(bill, tolerance)
    checks.append(check_subtotal(bill))
    checks.append(check_tax_split(bill))
    checks.append(check_grand_total(bill))
    return checks
