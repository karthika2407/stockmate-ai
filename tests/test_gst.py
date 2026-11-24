"""Tests for the GST rate schedule and the GST verification rules.

The point of these tests is not coverage for its own sake.  Each one pins down a
behaviour that a shopkeeper's dispute would turn on: that a malformed schedule
refuses to load rather than judging bills against a hole, that an 8 digit HSN
falls back the way the published schedule is organised, that each rule fires when
it should and stays quiet when it should not, and that an odd paisa in the
CGST/SGST split adds back to exactly the stated tax.
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stockmate.models import Bill, BillLine, CheckStatus
from stockmate.money import Money, total as money_total
from stockmate.reference.gst_rates import (
    DEFAULT_CSV,
    GstSchedule,
    GstScheduleError,
    default_schedule,
)
from stockmate.verify.gst import make_gst_checker, run_gst_checks
from stockmate.verify.suite import verify_bill

HEADER = (
    "hsn_prefix,description,rate_percent,effective_from,effective_to,source,confidence\n"
)
SRC = "test fixture (not a legal citation)"
AS_ON = date(2026, 4, 1)


# ------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def shipped() -> GstSchedule:
    return default_schedule()


def write_schedule(tmp_path: Path, rows: list[str], name: str = "rates.csv") -> Path:
    path = tmp_path / name
    path.write_text(HEADER + "".join(r if r.endswith("\n") else r + "\n" for r in rows),
                    encoding="utf-8")
    return path


def one_line_bill(
    hsn: str | None,
    rate: Decimal | None,
    *,
    unit_paise: int = 10_000,
    qty: int = 3,
    invoice_date: date = AS_ON,
    tax_override: Money | None = None,
    split: bool = True,
    igst: Money | None = None,
) -> Bill:
    """A bill that is correct by construction unless a caller overrides a figure."""
    unit = Money.from_paise(unit_paise)
    line = BillLine(
        index=0,
        description="test item",
        quantity=Decimal(qty),
        unit_price=unit,
        line_total=unit * Decimal(qty),
        hsn=hsn,
        gst_rate=rate,
    )
    subtotal = line.line_total
    tax = tax_override
    if tax is None:
        tax = subtotal.percent(rate) if rate is not None else Money.zero()
    bill = Bill(
        bill_id="t1",
        invoice_number="t1",
        invoice_date=invoice_date,
        lines=[line],
        subtotal=subtotal,
        tax_total=tax,
        grand_total=subtotal + tax,
        igst=igst,
    )
    if split:
        bill.cgst, bill.sgst = tax.split_half()
    return bill


def rule(checks, rule_id):
    found = [c for c in checks if c.rule_id == rule_id]
    assert found, f"{rule_id} did not run at all"
    return found[0]


# ------------------------------------------------------- schedule validation


def test_shipped_schedule_loads_and_is_all_unverified(shipped: GstSchedule):
    assert len(shipped.entries) > 0
    assert shipped.verified_rows == 0
    assert shipped.unverified_rows == len(shipped.entries)
    assert "unverified" in shipped.source_note
    assert shipped.schedule_version.startswith(DEFAULT_CSV.name + "@")


def test_rejects_rate_outside_the_legal_slabs(tmp_path: Path):
    path = write_schedule(
        tmp_path,
        [f'1006,Rice,14,2025-09-22,,"{SRC}",unverified'],
    )
    with pytest.raises(GstScheduleError) as exc:
        GstSchedule.load(path)
    # The row number a text editor shows, so the operator can go and fix it.
    assert "row 2" in str(exc.value)
    assert "legal slab" in str(exc.value)


def test_rejects_rate_that_was_legal_only_in_another_era(tmp_path: Path):
    # 12 percent was a slab before 22 September 2025 and not after it.
    path = write_schedule(
        tmp_path,
        [f'2106,Namkeen,12,2025-09-22,,"{SRC}",unverified'],
    )
    with pytest.raises(GstScheduleError):
        GstSchedule.load(path)


def test_rejects_overlapping_date_ranges_for_one_prefix(tmp_path: Path):
    path = write_schedule(
        tmp_path,
        [
            f'1006,Rice,5,2017-07-01,2025-09-30,"{SRC}",unverified',
            f'1006,Rice,0,2025-09-22,,"{SRC}",unverified',
        ],
    )
    with pytest.raises(GstScheduleError) as exc:
        GstSchedule.load(path)
    assert "overlaps" in str(exc.value)


def test_accepts_adjacent_ranges_for_one_prefix(tmp_path: Path):
    path = write_schedule(
        tmp_path,
        [
            f'1006,Rice,5,2017-07-01,2025-09-21,"{SRC}",unverified',
            f'1006,Rice,0,2025-09-22,,"{SRC}",unverified',
        ],
    )
    schedule = GstSchedule.load(path)
    assert schedule.lookup("1006", date(2025, 9, 21)).rate_percent == Decimal(5)
    assert schedule.lookup("1006", date(2025, 9, 22)).rate_percent == Decimal(0)


def test_rejects_malformed_rows_with_the_row_number(tmp_path: Path):
    bad = [
        f'1006,Rice,five,2025-09-22,,"{SRC}",unverified',
        f'100,Rice,5,2025-09-22,,"{SRC}",unverified',
        f'1006,Rice,5,22-09-2025,,"{SRC}",unverified',
        f'1006,Rice,5,2025-09-22,2025-01-01,"{SRC}",unverified',
        f'1006,Rice,5,2025-09-22,,"{SRC}",probably',
        "1006,Rice,5,2025-09-22,,,unverified",
    ]
    for row in bad:
        path = write_schedule(tmp_path, [row], name=f"bad{bad.index(row)}.csv")
        with pytest.raises(GstScheduleError) as exc:
            GstSchedule.load(path)
        assert "row 2" in str(exc.value), row


def test_bad_header_is_refused(tmp_path: Path):
    path = tmp_path / "wrong.csv"
    path.write_text("hsn,rate\n1006,5\n", encoding="utf-8")
    with pytest.raises(GstScheduleError):
        GstSchedule.load(path)


# ------------------------------------------------------------ longest prefix


def test_longest_prefix_fallback(tmp_path: Path):
    path = write_schedule(
        tmp_path,
        [
            f'33,Chapter 33 fallback,18,2025-09-22,,"{SRC}",unverified',
            f'3305,Hair preparations,18,2025-09-22,,"{SRC}",unverified',
            f'330510,Shampoo,5,2025-09-22,,"{SRC}",unverified',
            f'33051011,Shampoo sachet,0,2025-09-22,,"{SRC}",unverified',
        ],
    )
    schedule = GstSchedule.load(path)
    assert schedule.lookup("33051011", AS_ON).hsn_prefix == "33051011"
    assert schedule.lookup("33051099", AS_ON).hsn_prefix == "330510"
    assert schedule.lookup("33059011", AS_ON).hsn_prefix == "3305"
    assert schedule.lookup("33069900", AS_ON).hsn_prefix == "33"
    assert schedule.lookup("1006", AS_ON) is None


def test_lookup_ignores_printed_separators(shipped: GstSchedule):
    assert shipped.lookup("1006 30 10", AS_ON) is not None
    assert shipped.lookup("1006.30.10", AS_ON).hsn_prefix == "1006"
    assert shipped.lookup("", AS_ON) is None
    assert shipped.lookup(None, AS_ON) is None
    assert shipped.lookup("abcd", AS_ON) is None


def test_slabs_and_is_legal_rate_depend_on_the_date(shipped: GstSchedule):
    old = shipped.slabs(date(2020, 1, 1))
    new = shipped.slabs(date(2026, 1, 1))
    assert Decimal(12) in old and Decimal(12) not in new
    assert Decimal(40) in new and Decimal(40) not in old
    assert shipped.is_legal_rate(Decimal("12"), date(2020, 1, 1))
    assert not shipped.is_legal_rate(Decimal("12"), date(2026, 1, 1))
    assert not shipped.is_legal_rate("not a number", date(2026, 1, 1))


# -------------------------------------------------------------- rule: known


def test_rate_known_skips_when_no_hsn_is_printed(shipped: GstSchedule):
    checks = run_gst_checks(one_line_bill(None, Decimal(5)), shipped)
    known = rule(checks, "GST.RATE_KNOWN")
    assert known.status is CheckStatus.SKIPPED
    assert "not an error on the supplier" in known.message


def test_rate_known_passes_for_a_scheduled_hsn(shipped: GstSchedule):
    entry = shipped.lookup("1905", AS_ON)
    checks = run_gst_checks(one_line_bill("19059090", entry.rate_percent), shipped)
    assert rule(checks, "GST.RATE_KNOWN").status is CheckStatus.PASS


def test_rate_known_reports_an_hsn_the_schedule_does_not_cover(shipped: GstSchedule):
    checks = run_gst_checks(one_line_bill("84713010", Decimal(18)), shipped)
    known = rule(checks, "GST.RATE_KNOWN")
    assert known.status is CheckStatus.FAIL
    assert "gap in our reference file" in known.message


# -------------------------------------------------------------- rule: match


def test_rate_match_passes_on_the_scheduled_rate(shipped: GstSchedule):
    entry = shipped.lookup("3401", AS_ON)
    checks = run_gst_checks(one_line_bill("34011190", entry.rate_percent), shipped)
    assert rule(checks, "GST.RATE_MATCH").status is CheckStatus.PASS


def test_rate_match_fails_and_names_the_schedule_and_the_rupees(shipped: GstSchedule):
    entry = shipped.lookup("3401", AS_ON)
    wrong = Decimal(18) if entry.rate_percent != Decimal(18) else Decimal(5)
    bill = one_line_bill("34011190", wrong, unit_paise=10_000, qty=3)
    check = rule(run_gst_checks(bill, shipped), "GST.RATE_MATCH")
    assert check.status is CheckStatus.FAIL
    assert shipped.schedule_version in check.message
    assert "reference data" in check.message
    # 300 rupees taxable: 18 percent is 54.00, 5 percent is 15.00.
    taxable = Money.from_rupees("300.00")
    expected_delta = taxable.percent(wrong) - taxable.percent(entry.rate_percent)
    assert check.delta == expected_delta
    assert check.evidence["schedule_version"] == shipped.schedule_version


def test_rate_match_skips_when_the_bill_prints_no_rate(shipped: GstSchedule):
    checks = run_gst_checks(one_line_bill("34011190", None), shipped)
    assert rule(checks, "GST.RATE_MATCH").status is CheckStatus.SKIPPED


# --------------------------------------------------------------- rule: slab


def test_slab_legal_fails_on_a_rate_that_never_existed(shipped: GstSchedule):
    bill = one_line_bill("34011190", Decimal("14"))
    check = rule(run_gst_checks(bill, shipped), "GST.SLAB_LEGAL")
    assert check.status is CheckStatus.FAIL
    assert "not a legal slab" in check.message


def test_slab_legal_is_date_aware(shipped: GstSchedule):
    old = one_line_bill("2106", Decimal("12"), invoice_date=date(2024, 5, 1))
    assert rule(run_gst_checks(old, shipped), "GST.SLAB_LEGAL").status is CheckStatus.PASS
    new = one_line_bill("2106", Decimal("12"), invoice_date=date(2026, 5, 1))
    assert rule(run_gst_checks(new, shipped), "GST.SLAB_LEGAL").status is CheckStatus.FAIL


# ---------------------------------------------------------- rule: recompute


def test_tax_recompute_passes_on_a_consistent_bill(shipped: GstSchedule):
    bill = one_line_bill("34011190", shipped.lookup("3401", AS_ON).rate_percent)
    assert rule(run_gst_checks(bill, shipped), "GST.TAX_RECOMPUTE").status is CheckStatus.PASS


def test_tax_recompute_tolerates_rounding_but_not_a_real_error(shipped: GstSchedule):
    rate = shipped.lookup("3401", AS_ON).rate_percent
    good = one_line_bill("34011190", rate)
    inside = Money.from_paise(good.tax_total.paise + 99)
    outside = Money.from_paise(good.tax_total.paise + 500)

    near = one_line_bill("34011190", rate, tax_override=inside)
    assert rule(run_gst_checks(near, shipped), "GST.TAX_RECOMPUTE").status is CheckStatus.PASS

    far = one_line_bill("34011190", rate, tax_override=outside)
    check = rule(run_gst_checks(far, shipped), "GST.TAX_RECOMPUTE")
    assert check.status is CheckStatus.FAIL
    assert check.delta == Money.from_paise(500)
    assert check.evidence["tolerance_paise"] == 100


def test_tax_recompute_window_grows_with_line_count(shipped: GstSchedule):
    lines = []
    for i in range(10):
        unit = Money.from_paise(1_999)
        lines.append(
            BillLine(
                index=i,
                description=f"soap {i}",
                quantity=Decimal(2),
                unit_price=unit,
                line_total=unit * Decimal(2),
                hsn="34011190",
                gst_rate=shipped.lookup("3401", AS_ON).rate_percent,
            )
        )
    subtotal = money_total(l.line_total for l in lines)
    exact = money_total(l.line_total.percent(l.gst_rate) for l in lines)
    tax = Money.from_paise(exact.paise + 400)          # inside 10 x 50 paise
    bill = Bill(
        bill_id="t10",
        invoice_date=AS_ON,
        lines=lines,
        subtotal=subtotal,
        tax_total=tax,
        grand_total=subtotal + tax,
    )
    bill.cgst, bill.sgst = tax.split_half()
    check = rule(run_gst_checks(bill, shipped), "GST.TAX_RECOMPUTE")
    assert check.evidence["tolerance_paise"] == 500
    assert check.status is CheckStatus.PASS


def test_tax_recompute_skips_without_a_printed_tax(shipped: GstSchedule):
    bill = one_line_bill("34011190", Decimal(5))
    bill.tax_total = None
    bill.cgst = bill.sgst = bill.igst = None
    assert rule(run_gst_checks(bill, shipped), "GST.TAX_RECOMPUTE").status is CheckStatus.SKIPPED


# -------------------------------------------------------------- rule: split


def test_odd_paisa_split_adds_back_exactly(shipped: GstSchedule):
    # A tax of 15.01 rupees cannot be halved evenly; the odd paisa goes to CGST.
    tax = Money.from_rupees("15.01")
    cgst, sgst = tax.split_half()
    assert cgst.paise == 751 and sgst.paise == 750
    assert cgst + sgst == tax

    bill = one_line_bill("34011190", Decimal(5), tax_override=tax, split=False)
    bill.cgst, bill.sgst = cgst, sgst
    check = rule(run_gst_checks(bill, shipped), "GST.SPLIT_CONSISTENT")
    assert check.status is CheckStatus.PASS
    assert check.evidence["expected_cgst_paise"] == 751


def test_split_fails_when_the_halves_do_not_add_to_the_tax(shipped: GstSchedule):
    tax = Money.from_rupees("15.01")
    bill = one_line_bill("34011190", Decimal(5), tax_override=tax, split=False)
    bill.cgst = Money.from_paise(750)
    bill.sgst = Money.from_paise(750)
    check = rule(run_gst_checks(bill, shipped), "GST.SPLIT_CONSISTENT")
    assert check.status is CheckStatus.FAIL
    assert check.delta == Money.from_paise(-1)


def test_split_fails_when_the_halves_are_uneven(shipped: GstSchedule):
    tax = Money.from_rupees("20.00")
    bill = one_line_bill("34011190", Decimal(5), tax_override=tax, split=False)
    bill.cgst = Money.from_rupees("12.00")
    bill.sgst = Money.from_rupees("8.00")
    check = rule(run_gst_checks(bill, shipped), "GST.SPLIT_CONSISTENT")
    assert check.status is CheckStatus.FAIL
    assert "should split as" in check.message


def test_igst_must_not_coexist_with_cgst_and_sgst(shipped: GstSchedule):
    bill = one_line_bill("34011190", Decimal(5), igst=Money.from_rupees("15.00"))
    check = rule(run_gst_checks(bill, shipped), "GST.SPLIT_CONSISTENT")
    assert check.status is CheckStatus.FAIL
    assert "either inside the state" in check.message


def test_split_skips_on_an_igst_only_bill(shipped: GstSchedule):
    bill = one_line_bill("34011190", Decimal(5), split=False)
    bill.igst = bill.tax_total
    assert rule(run_gst_checks(bill, shipped), "GST.SPLIT_CONSISTENT").status is CheckStatus.SKIPPED


# -------------------------------------------------------------- integration


def test_clean_bill_raises_no_gst_failure_through_the_suite(shipped: GstSchedule):
    checker = make_gst_checker(shipped)
    entry = shipped.lookup("1905", AS_ON)
    bill = one_line_bill("19059090", entry.rate_percent)
    report = verify_bill(bill, gst_checker=checker)
    assert report.failures_in(("gst",)) == []
    assert all(c.rule_id.startswith("GST.") for c in report.by_tier("gst"))
    assert checker.schedule_version == shipped.schedule_version


def test_every_gst_check_records_the_schedule_version(shipped: GstSchedule):
    bill = one_line_bill("19059090", Decimal("14"))
    for check in run_gst_checks(bill, shipped):
        assert check.evidence.get("schedule_version") == shipped.schedule_version


def test_missing_invoice_date_is_flagged_as_assumed(shipped: GstSchedule):
    bill = one_line_bill("19059090", Decimal(5))
    bill.invoice_date = None
    checks = run_gst_checks(bill, shipped)
    assert all(c.evidence["invoice_date_assumed"] for c in checks)
