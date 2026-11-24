"""Verification rules: they must fire on real faults and stay quiet otherwise.

The silence tests matter more than the detection tests.  A rule that cries wolf
on an ordinary bill gets switched off within a week, and then it detects nothing
at all.
"""

from datetime import date
from decimal import Decimal

from stockmate.models import Bill, BillLine, CheckStatus
from stockmate.money import Money
from stockmate.verify.arithmetic import run_arithmetic_checks
from stockmate.verify.duplicate import check_duplicate_bill, check_duplicate_lines
from stockmate.verify.suite import verify_bill


def rupees(x):
    return Money.from_rupees(x)


def line(i, desc, qty, unit, total=None):
    up = rupees(unit)
    return BillLine(
        index=i, description=desc, quantity=Decimal(str(qty)),
        unit_price=up, line_total=rupees(total) if total else up * Decimal(str(qty)),
    )


def clean_bill() -> Bill:
    lines = [
        line(0, "Sona Masoori Rice 5kg", 10, "310.00"),
        line(1, "Sunflower Oil 1L", 12, "139.00"),
        line(2, "Sugar 1kg", 20, "46.00"),
    ]
    sub = Money(sum(l.line_total.paise for l in lines))
    return Bill(
        bill_id="t1", supplier_name="Anand Distributors",
        invoice_number="AD/1", invoice_date=date(2026, 11, 24),
        lines=lines, subtotal=sub, grand_total=sub,
    )


def failures(bill, **kw):
    return [c for c in run_arithmetic_checks(bill, **kw) if c.status is CheckStatus.FAIL]


def test_clean_bill_raises_nothing():
    assert failures(clean_bill()) == []


def test_line_multiplication_error_is_caught_and_localised():
    bill = clean_bill()
    bill.lines[1].line_total = rupees("1700.00")      # should be 1668
    bill.subtotal = Money(sum(l.line_total.paise for l in bill.lines))
    bill.grand_total = bill.subtotal
    fails = failures(bill)
    line_fails = [c for c in fails if c.rule_id == "ARITH.LINE"]
    assert len(line_fails) == 1
    assert line_fails[0].line_index == 1
    assert line_fails[0].delta == rupees("32.00")


def test_subtotal_that_does_not_match_the_lines():
    bill = clean_bill()
    bill.subtotal = bill.subtotal + rupees("500.00")
    bill.grand_total = bill.subtotal
    assert any(c.rule_id == "ARITH.SUBTOTAL" for c in failures(bill))


def test_grand_total_inflation_is_caught():
    bill = clean_bill()
    bill.grand_total = bill.grand_total + rupees("250.00")
    fails = [c for c in failures(bill) if c.rule_id == "ARITH.GRAND_TOTAL"]
    assert len(fails) == 1
    assert fails[0].delta == rupees("250.00")


def test_rounding_inside_a_rupee_is_not_a_dispute():
    bill = clean_bill()
    bill.grand_total = bill.grand_total + rupees("0.80")
    assert not [c for c in failures(bill) if c.rule_id == "ARITH.GRAND_TOTAL"]


def test_discount_applied_or_not_both_read_as_consistent():
    # Real bills differ on whether a printed discount is already inside the
    # subtotal. Either reading must pass rather than accuse the supplier.
    bill = clean_bill()
    bill.discount_total = rupees("100.00")
    assert not [c for c in failures(bill) if c.rule_id == "ARITH.GRAND_TOTAL"]

    bill.grand_total = bill.subtotal - rupees("100.00")
    assert not [c for c in failures(bill) if c.rule_id == "ARITH.GRAND_TOTAL"]


def test_igst_cannot_sit_beside_cgst_and_sgst():
    bill = clean_bill()
    bill.cgst, bill.sgst, bill.igst = rupees("90.00"), rupees("90.00"), rupees("180.00")
    assert any(c.rule_id == "ARITH.TAX_SPLIT" for c in failures(bill))


def test_missing_values_are_skipped_not_failed():
    bill = clean_bill()
    bill.lines[0].unit_price = None
    statuses = {c.status for c in run_arithmetic_checks(bill) if c.line_index == 0}
    assert CheckStatus.SKIPPED in statuses
    assert CheckStatus.FAIL not in statuses


def test_duplicate_line_is_flagged_although_arithmetic_closes():
    bill = clean_bill()
    dup = line(3, "Sugar 1kg", 20, "46.00")
    bill.lines.append(dup)
    bill.subtotal = Money(sum(l.line_total.paise for l in bill.lines))
    bill.grand_total = bill.subtotal
    assert failures(bill) == []                       # arithmetic is perfect
    dup_fails = [c for c in check_duplicate_lines(bill) if c.status is CheckStatus.FAIL]
    assert len(dup_fails) == 1
    assert dup_fails[0].delta == rupees("920.00")


def test_distinct_lines_are_not_called_duplicates():
    assert not [c for c in check_duplicate_lines(clean_bill()) if c.status is CheckStatus.FAIL]


def test_the_same_bill_sent_twice_is_recognised():
    bill = clean_bill()
    seen: dict[str, str] = {}
    first = check_duplicate_bill(bill, seen)
    assert first.status is CheckStatus.PASS
    seen[bill.content_hash()] = bill.bill_id

    again = Bill(
        bill_id="t2", supplier_name=bill.supplier_name,
        invoice_number=bill.invoice_number, invoice_date=bill.invoice_date,
        lines=bill.lines, subtotal=bill.subtotal, grand_total=bill.grand_total,
    )
    assert check_duplicate_bill(again, seen).status is CheckStatus.FAIL


def test_suite_tiers_can_be_scored_separately():
    bill = clean_bill()
    bill.lines.append(line(3, "Sugar 1kg", 20, "46.00"))
    bill.subtotal = Money(sum(l.line_total.paise for l in bill.lines))
    bill.grand_total = bill.subtotal
    report = verify_bill(bill)
    assert report.failures_in(("arithmetic",)) == []
    assert report.failures_in(("duplicate",))
