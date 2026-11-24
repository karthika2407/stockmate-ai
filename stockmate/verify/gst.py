"""GST checks on a supplier bill.

These rules compare what the bill charged against a dated rate schedule that is
loaded as reference data.  That distinction runs through every message below: a
rate mismatch is evidence that the bill and our schedule disagree, and the
schedule is the part of that pair we cannot prove.  So every failure says which
schedule version judged it and that the schedule may be out of date, because the
shopkeeper is going to read the message out to a supplier who will argue back.

The arithmetic rules in ``arithmetic.py`` are the style reference, and the
tolerance policy is the same one: suppliers round tax at different points in the
calculation, so the recompute window is explicit, is stated in the evidence, and
grows with the number of taxed lines rather than staying at a flat rupee.

What these rules never do is fail a bill for something that is not the supplier's
doing.  A bill that prints no HSN is normal at the small end of the trade, and a
code our schedule has never heard of is a gap in our file.  Both are reported,
neither is charged to the supplier.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ..models import Bill, BillLine, Check, CheckStatus, Severity
from ..money import Money, total as money_total
from ..reference.gst_rates import GstSchedule, RateEntry

__all__ = [
    "RECOMPUTE_FLOOR",
    "RECOMPUTE_PER_LINE",
    "check_rate_known",
    "check_rate_match",
    "check_slab_legal",
    "check_tax_recompute",
    "check_split_consistent",
    "run_gst_checks",
    "make_gst_checker",
]

#: Smallest recompute window, in paise.  One rupee absorbs the rounding a single
#: taxed line can introduce between the supplier's software and ours.
RECOMPUTE_FLOOR = 100

#: Added per taxed line, in paise.  Each line rounds its own tax to the paisa, so
#: the error the total can legitimately carry grows with the line count.
RECOMPUTE_PER_LINE = 50


def _within(a: Money, b: Money, tolerance: Money) -> bool:
    return abs((a - b).paise) <= tolerance.paise


def _taxable(line: BillLine) -> Money | None:
    """The value the tax should have been charged on for this line.

    The printed line total is used when there is one, because that is the figure
    the supplier themselves taxed.  Falling back to quantity times price keeps the
    rule alive on bills that print a rate and a quantity but no line total.
    """
    if line.line_total is not None:
        return line.line_total
    expected = line.expected_total()
    if expected is None:
        return None
    return expected - line.discount if line.discount else expected


def _as_on(bill: Bill) -> tuple[date, bool]:
    """Which day's schedule applies, and whether that day had to be assumed.

    A bill with no readable date is judged against today's schedule, because
    refusing to check it at all is worse than checking it against the rates now
    in force.  The assumption is recorded in the evidence of every check that
    used it.
    """
    if bill.invoice_date is not None:
        return bill.invoice_date, False
    return date.today(), True


def _rate_of(line: BillLine) -> Decimal | None:
    if line.gst_rate is None:
        return None
    return line.gst_rate if isinstance(line.gst_rate, Decimal) else Decimal(str(line.gst_rate))


def _pct(rate: Decimal) -> str:
    """Print a rate the way a bill prints it: 5, 18, 18.5, not 5.000."""
    normalised = rate.normalize()
    text = format(normalised, "f")
    return text


def _base_evidence(schedule: GstSchedule, on_date: date, assumed: bool) -> dict:
    return {
        "schedule_version": schedule.schedule_version,
        "schedule_source_note": schedule.source_note,
        "judged_as_on": on_date.isoformat(),
        "invoice_date_assumed": assumed,
    }


# ------------------------------------------------------------------ rule: known


def check_rate_known(bill: Bill, schedule: GstSchedule) -> list[Check]:
    """Is the line's HSN covered by the schedule at all?

    A missing HSN is skipped, never failed.  Small suppliers below the HSN
    reporting threshold are not required to print one, and penalising the shop's
    bill for that would train the owner to ignore the auditor.
    """
    on_date, assumed = _as_on(bill)
    base = _base_evidence(schedule, on_date, assumed)
    checks: list[Check] = []
    for line in bill.lines:
        if not line.hsn:
            checks.append(
                Check(
                    rule_id="GST.RATE_KNOWN",
                    status=CheckStatus.SKIPPED,
                    message=(
                        f"Line {line.index + 1} ({line.description}): the bill prints "
                        "no HSN code, so the scheduled rate cannot be looked up. "
                        "This is not an error on the supplier's part."
                    ),
                    line_index=line.index,
                    severity=Severity.INFO,
                    evidence=dict(base),
                )
            )
            continue

        entry = schedule.lookup(line.hsn, on_date)
        if entry is not None:
            checks.append(
                Check(
                    rule_id="GST.RATE_KNOWN",
                    status=CheckStatus.PASS,
                    message=(
                        f"Line {line.index + 1} ({line.description}): HSN {line.hsn} "
                        f"matches schedule entry {entry.hsn_prefix} "
                        f"({entry.description}) at {_pct(entry.rate_percent)} percent."
                    ),
                    expected=entry.rate_percent,
                    observed=line.hsn,
                    line_index=line.index,
                    severity=Severity.INFO,
                    evidence={
                        **base,
                        "matched_prefix": entry.hsn_prefix,
                        "entry_confidence": entry.confidence,
                        "entry_source": entry.source,
                    },
                )
            )
            continue

        checks.append(
            Check(
                rule_id="GST.RATE_KNOWN",
                status=CheckStatus.FAIL,
                message=(
                    f"Line {line.index + 1} ({line.description}): HSN {line.hsn} is "
                    "not in our rate schedule "
                    f"({schedule.schedule_version}), so the charged rate could not "
                    "be checked. This is a gap in our reference file, not a claim "
                    "against the supplier. Add the code to "
                    "data/reference/gst_rates.csv from the current CBIC schedule."
                ),
                expected="an HSN prefix present in the schedule",
                observed=line.hsn,
                line_index=line.index,
                severity=Severity.LOW,
                evidence={**base, "prefixes_tried": _prefixes_tried(line.hsn)},
            )
        )
    return checks


def _prefixes_tried(hsn: str) -> list[str]:
    digits = "".join(ch for ch in str(hsn) if ch.isdigit())
    return [digits[:n] for n in (8, 6, 4, 2) if len(digits) >= n]


# ------------------------------------------------------------------ rule: match


def check_rate_match(bill: Bill, schedule: GstSchedule) -> list[Check]:
    """Does the charged rate equal the scheduled rate for that HSN?"""
    on_date, assumed = _as_on(bill)
    base = _base_evidence(schedule, on_date, assumed)
    checks: list[Check] = []
    for line in bill.lines:
        charged = _rate_of(line)
        entry = schedule.lookup(line.hsn, on_date) if line.hsn else None
        if charged is None or entry is None:
            reason = (
                "the bill prints no GST rate for this line"
                if charged is None
                else f"HSN {line.hsn} is not in the schedule"
            )
            checks.append(
                Check(
                    rule_id="GST.RATE_MATCH",
                    status=CheckStatus.SKIPPED,
                    message=(
                        f"Line {line.index + 1} ({line.description}): cannot compare "
                        f"rates because {reason}."
                    ),
                    line_index=line.index,
                    severity=Severity.INFO,
                    evidence=dict(base),
                )
            )
            continue

        ok = charged == entry.rate_percent
        taxable = _taxable(line)
        delta = None
        money_phrase = ""
        if taxable is not None:
            charged_tax = taxable.percent(charged)
            scheduled_tax = taxable.percent(entry.rate_percent)
            delta = charged_tax - scheduled_tax
            if not ok:
                direction = "more" if delta.paise > 0 else "less"
                money_phrase = (
                    f" On a taxable value of {taxable.format()} that is "
                    f"{abs(delta).format()} {direction} tax than the schedule gives: "
                    f"{charged_tax.format()} charged against {scheduled_tax.format()} "
                    "scheduled."
                )

        evidence = {
            **base,
            "matched_prefix": entry.hsn_prefix,
            "entry_row": entry.row_number,
            "entry_confidence": entry.confidence,
            "entry_source": entry.source,
            "taxable_paise": taxable.paise if taxable is not None else None,
        }

        if ok:
            checks.append(
                Check(
                    rule_id="GST.RATE_MATCH",
                    status=CheckStatus.PASS,
                    message=(
                        f"Line {line.index + 1} ({line.description}): charged "
                        f"{_pct(charged)} percent, which is what schedule "
                        f"{schedule.schedule_version} gives for HSN {line.hsn}."
                    ),
                    expected=entry.rate_percent,
                    observed=charged,
                    delta=delta,
                    line_index=line.index,
                    severity=Severity.INFO,
                    evidence=evidence,
                )
            )
            continue

        checks.append(
            Check(
                rule_id="GST.RATE_MATCH",
                status=CheckStatus.FAIL,
                message=(
                    f"Line {line.index + 1} ({line.description}): the bill charges "
                    f"{_pct(charged)} percent GST, but schedule "
                    f"{schedule.schedule_version} gives {_pct(entry.rate_percent)} "
                    f"percent for HSN {line.hsn} ({entry.description}) as on "
                    f"{on_date.isoformat()}." + money_phrase
                    + " That schedule is reference data, not a legal citation: row "
                    f"{entry.row_number} is marked {entry.confidence} and the rate "
                    "may have changed since. Ask the supplier which notification "
                    "they billed under before treating this as an overcharge."
                ),
                expected=entry.rate_percent,
                observed=charged,
                delta=delta,
                line_index=line.index,
                severity=Severity.MEDIUM,
                evidence=evidence,
            )
        )
    return checks


# ------------------------------------------------------------------- rule: slab


def check_slab_legal(bill: Bill, schedule: GstSchedule) -> list[Check]:
    """Is the charged rate one of the slabs the tax allowed on that date?

    This rule needs no item knowledge at all, which makes it the strongest GST
    rule in the suite: 14 percent was never a GST slab, whoever the supplier is
    and whatever the goods were.
    """
    on_date, assumed = _as_on(bill)
    base = _base_evidence(schedule, on_date, assumed)
    slabs = schedule.slabs(on_date)
    slab_text = ", ".join(_pct(s) for s in sorted(slabs))
    checks: list[Check] = []
    for line in bill.lines:
        charged = _rate_of(line)
        if charged is None:
            checks.append(
                Check(
                    rule_id="GST.SLAB_LEGAL",
                    status=CheckStatus.SKIPPED,
                    message=(
                        f"Line {line.index + 1} ({line.description}): the bill prints "
                        "no GST rate for this line."
                    ),
                    line_index=line.index,
                    severity=Severity.INFO,
                    evidence=dict(base),
                )
            )
            continue

        ok = schedule.is_legal_rate(charged, on_date)
        taxable = _taxable(line)
        checks.append(
            Check(
                rule_id="GST.SLAB_LEGAL",
                status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                message=(
                    f"Line {line.index + 1} ({line.description}): {_pct(charged)} "
                    "percent is one of the slabs in force on "
                    f"{on_date.isoformat()} ({slab_text})."
                    if ok
                    else (
                        f"Line {line.index + 1} ({line.description}): the bill "
                        f"charges {_pct(charged)} percent GST, which was not a legal "
                        f"slab on {on_date.isoformat()}. The slabs then were "
                        f"{slab_text}."
                        + (
                            f" On a taxable value of {taxable.format()} the line "
                            f"carries {taxable.percent(charged).format()} of tax at a "
                            "rate that does not exist."
                            if taxable is not None
                            else ""
                        )
                        + " The slab list is reference data held alongside "
                        f"{schedule.schedule_version} and may be out of date if the "
                        "Council has since changed the structure."
                    )
                ),
                expected=slab_text,
                observed=charged,
                line_index=line.index,
                severity=Severity.HIGH if not ok else Severity.INFO,
                evidence={**base, "legal_slabs": [str(s) for s in sorted(slabs)]},
            )
        )
    return checks


# -------------------------------------------------------------- rule: recompute


def check_tax_recompute(
    bill: Bill, schedule: GstSchedule, tolerance: Money | None = None
) -> Check:
    """Rebuild the tax from taxable value times rate and compare with the print.

    The rate used is the one the bill itself charged, not the scheduled one, so
    this rule is independent of whether our schedule is current.  It catches the
    common case of a bill whose rates and values are all plausible but whose tax
    line does not follow from them.
    """
    on_date, assumed = _as_on(bill)
    base = _base_evidence(schedule, on_date, assumed)

    taxed: list[tuple[BillLine, Decimal, Money]] = []
    for line in bill.lines:
        rate = _rate_of(line)
        taxable = _taxable(line)
        if rate is None or taxable is None:
            continue
        taxed.append((line, rate, taxable))

    printed = bill.tax_total
    if printed is None:
        parts = [t for t in (bill.cgst, bill.sgst, bill.igst) if t is not None]
        printed = money_total(parts) if parts else None

    if not taxed or printed is None:
        reason = (
            "no line carries both a taxable value and a rate"
            if not taxed
            else "the bill prints no tax figure"
        )
        return Check(
            rule_id="GST.TAX_RECOMPUTE",
            status=CheckStatus.SKIPPED,
            message=f"Tax cannot be recomputed: {reason}.",
            severity=Severity.INFO,
            evidence={**base, "taxed_lines": len(taxed)},
        )

    computed = money_total(taxable.percent(rate) for _, rate, taxable in taxed)
    tol = tolerance or Money.from_paise(
        max(RECOMPUTE_FLOOR, RECOMPUTE_PER_LINE * len(taxed))
    )
    delta = printed - computed
    ok = _within(printed, computed, tol)

    evidence = {
        **base,
        "taxed_lines": len(taxed),
        "tolerance_paise": tol.paise,
        "tolerance_rule": (
            f"max({RECOMPUTE_FLOOR}, {RECOMPUTE_PER_LINE} x taxed lines) paise"
        ),
        "per_line": [
            {
                "line_index": line.index,
                "rate_percent": str(rate),
                "taxable_paise": taxable.paise,
                "tax_paise": taxable.percent(rate).paise,
            }
            for line, rate, taxable in taxed
        ],
    }

    return Check(
        rule_id="GST.TAX_RECOMPUTE",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        message=(
            (
                f"{len(taxed)} taxed lines at their own printed rates come to "
                f"{computed.format()} of GST; the bill says {printed.format()}, "
                f"within the {tol.format()} rounding window."
            )
            if ok
            else (
                f"{len(taxed)} taxed lines at their own printed rates come to "
                f"{computed.format()} of GST, but the bill charges "
                f"{printed.format()}, a difference of {delta.format()}. The "
                f"rounding window allowed here is {tol.format()} "
                f"({RECOMPUTE_PER_LINE} paise a line, at least "
                f"{RECOMPUTE_FLOOR} paise). This arithmetic uses the rates the "
                "bill itself printed, so it does not depend on our rate schedule "
                "being current."
            )
        ),
        expected=computed,
        observed=printed,
        delta=delta,
        severity=Severity.HIGH if not ok else Severity.INFO,
        evidence=evidence,
    )


# ------------------------------------------------------------------ rule: split


def check_split_consistent(bill: Bill, schedule: GstSchedule) -> Check:
    """CGST plus SGST should be the whole tax, in halves, and never with IGST.

    The half-paisa case is real: tax of 15.01 rupees splits as 7.51 and 7.50, and
    a bill that prints 7.50 twice is a paisa short.  ``Money.split_half`` produces
    exactly the split billing software does, so the check compares against that
    rather than against a tolerance.
    """
    on_date, assumed = _as_on(bill)
    base = _base_evidence(schedule, on_date, assumed)

    has_intra = bill.cgst is not None or bill.sgst is not None
    has_inter = bill.igst is not None and not bill.igst.is_zero()

    if has_intra and has_inter:
        return Check(
            rule_id="GST.SPLIT_CONSISTENT",
            status=CheckStatus.FAIL,
            message=(
                f"The bill charges IGST {bill.igst.format()} alongside CGST and "
                "SGST. A supply is either inside the state, taxed as CGST plus "
                "SGST, or across state lines, taxed as IGST. Ask the supplier to "
                "reissue with one or the other."
            ),
            expected="CGST and SGST, or IGST, not both",
            observed="CGST, SGST and IGST together",
            severity=Severity.HIGH,
            evidence={
                **base,
                "cgst_paise": bill.cgst.paise if bill.cgst else None,
                "sgst_paise": bill.sgst.paise if bill.sgst else None,
                "igst_paise": bill.igst.paise,
            },
        )

    # An IGST-only bill lands here as well: there is nothing to split.
    if bill.cgst is None or bill.sgst is None:
        return Check(
            rule_id="GST.SPLIT_CONSISTENT",
            status=CheckStatus.SKIPPED,
            message="The bill does not print CGST and SGST separately.",
            severity=Severity.INFO,
            evidence=dict(base),
        )

    pair_total = bill.cgst + bill.sgst
    stated = bill.tax_total if bill.tax_total is not None else pair_total
    expected_cgst, expected_sgst = stated.split_half()

    sums_right = pair_total == stated
    halves_right = bill.cgst == expected_cgst and bill.sgst == expected_sgst
    ok = sums_right and halves_right

    if ok:
        return Check(
            rule_id="GST.SPLIT_CONSISTENT",
            status=CheckStatus.PASS,
            message=(
                f"CGST {bill.cgst.format()} plus SGST {bill.sgst.format()} is the "
                f"whole tax of {stated.format()}, split in halves."
            ),
            expected=stated,
            observed=pair_total,
            delta=pair_total - stated,
            severity=Severity.INFO,
            evidence={
                **base,
                "expected_cgst_paise": expected_cgst.paise,
                "expected_sgst_paise": expected_sgst.paise,
            },
        )

    delta = pair_total - stated
    if not sums_right:
        detail = (
            f"CGST {bill.cgst.format()} plus SGST {bill.sgst.format()} comes to "
            f"{pair_total.format()}, but the bill's tax total is "
            f"{stated.format()}, a difference of {delta.format()}."
        )
    else:
        detail = (
            f"Tax of {stated.format()} should split as CGST "
            f"{expected_cgst.format()} and SGST {expected_sgst.format()}, but the "
            f"bill prints CGST {bill.cgst.format()} and SGST {bill.sgst.format()}."
        )

    return Check(
        rule_id="GST.SPLIT_CONSISTENT",
        status=CheckStatus.FAIL,
        message=(
            detail + " CGST and SGST are each exactly half of the GST on an "
            "in-state supply, with any odd paisa going to the first of the two. "
            "This is arithmetic on the bill's own figures and does not depend on "
            "our rate schedule."
        ),
        expected=stated,
        observed=pair_total,
        delta=delta,
        severity=Severity.MEDIUM,
        evidence={
            **base,
            "cgst_paise": bill.cgst.paise,
            "sgst_paise": bill.sgst.paise,
            "expected_cgst_paise": expected_cgst.paise,
            "expected_sgst_paise": expected_sgst.paise,
        },
    )


# ------------------------------------------------------------------- the runner


def run_gst_checks(bill: Bill, schedule: GstSchedule) -> list[Check]:
    """Every GST rule, in the order an owner would read them."""
    checks: list[Check] = []
    checks.extend(check_rate_known(bill, schedule))
    checks.extend(check_rate_match(bill, schedule))
    checks.extend(check_slab_legal(bill, schedule))
    checks.append(check_tax_recompute(bill, schedule))
    checks.append(check_split_consistent(bill, schedule))
    return checks


def make_gst_checker(schedule: GstSchedule):
    """Bind a schedule to a one-argument checker for ``verify_bill``.

    The suite injects the checker rather than importing it so that a deployment
    with no usable schedule still runs the arithmetic and duplicate rules instead
    of failing to start.  Binding here keeps the schedule, and therefore the
    version recorded in every check, fixed for the life of the checker.
    """

    def gst_checker(bill: Bill) -> list[Check]:
        return run_gst_checks(bill, schedule)

    gst_checker.schedule_version = schedule.schedule_version
    gst_checker.source_note = schedule.source_note
    return gst_checker
