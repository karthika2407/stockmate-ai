"""Benchmark: does the GST rule engine behave as specified?

Protocol, and what this measurement is not
------------------------------------------
There is no public corpus of annotated Indian GST invoices with ground-truth
rates, and the official rate sources could not be fetched, so nothing here
measures whether our rate schedule is factually right.  What is measured is the
**rule engine**: given a schedule, does lookup resolve as specified, does a
perturbed rate get caught, how small a tax error is reliably caught, and does a
correctly built bill stay silent.  Every synthetic bill below is generated from
the schedule the engine is then judged against, which makes the rate-mismatch
number a self-consistency measurement of the engine and not field accuracy
against real supplier bills.

Four parts:
1. Lookup, exhaustively.  Every prefix in the schedule resolves on a date inside
   its own range; every prefix padded out to 8 digits resolves to the longest
   prefix that an independent brute-force scan over the rows says it should; and
   every 4 digit code the schedule does not cover returns None.
2. Rate-mismatch detection.  For each active schedule entry, build a correct
   one-line bill and move the charged rate to each other legal slab in turn.
   Scored as rule-engine detection, not as field accuracy.
3. Tax-recompute sensitivity.  Inject a known paise error into the printed tax
   of correctly built bills and sweep its size, reporting the smallest error
   caught on every bill of a given line count.  The engine's own tolerance is
   max(100, 50 x taxed lines) paise, so an error below that is expected to be
   missed by design.
4. False alarms.  Correctly built bills must produce zero GST failures.

Run:
    python benchmarks/bench_gst_rules.py
Writes results/gst_rules.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockmate.models import Bill, BillLine  # noqa: E402
from stockmate.money import Money, total as money_total  # noqa: E402
from stockmate.reference.gst_rates import (  # noqa: E402
    DEFAULT_CSV,
    PREFIX_LENGTHS,
    GstSchedule,
)
from stockmate.verify.gst import RECOMPUTE_FLOOR, RECOMPUTE_PER_LINE  # noqa: E402
from stockmate.verify.gst import make_gst_checker  # noqa: E402
from stockmate.verify.suite import verify_bill  # noqa: E402

#: Error sizes swept through the printed tax, in paise.
ERROR_SWEEP_PAISE = (
    1, 2, 5, 10, 25, 50, 75, 100, 101, 125, 150, 151, 200, 250, 251, 301,
    500, 501, 1000, 5000,
)

#: Line counts the recompute sweep is run at, because the tolerance grows with
#: the number of taxed lines and a single number would hide that.
LINE_COUNTS = (1, 2, 3, 5, 10)

#: Unit prices used to build lines, in paise.  Deliberately odd so that tax lands
#: on half paise and the rounding path is exercised rather than avoided.
UNIT_PRICES_PAISE = (1_999, 4_750, 12_333, 875, 23_457)


# ----------------------------------------------------------------- bill builder


def build_bill(
    bill_id: str,
    rows: list[tuple[str, Decimal, int, int]],
    invoice_date: date,
) -> Bill:
    """A bill that is correct by construction.

    ``rows`` is a list of (hsn, rate_percent, unit_price_paise, quantity).  Every
    total is computed the way billing software computes it, including the half-up
    rounding at each line, so a check that fires on one of these bills has found
    a fault in the engine and not in the fixture.
    """
    lines: list[BillLine] = []
    for index, (hsn, rate, price_paise, qty) in enumerate(rows):
        unit_price = Money.from_paise(price_paise)
        quantity = Decimal(qty)
        lines.append(
            BillLine(
                index=index,
                description=f"item {index + 1} hsn {hsn}",
                quantity=quantity,
                unit_price=unit_price,
                line_total=unit_price * quantity,
                hsn=hsn,
                gst_rate=rate,
            )
        )

    subtotal = money_total(l.line_total for l in lines)
    tax = money_total(l.line_total.percent(l.gst_rate) for l in lines)
    cgst, sgst = tax.split_half()
    return Bill(
        bill_id=bill_id,
        supplier_name="Synthetic Traders",
        invoice_number=bill_id,
        invoice_date=invoice_date,
        lines=lines,
        subtotal=subtotal,
        tax_total=tax,
        cgst=cgst,
        sgst=sgst,
        grand_total=subtotal + tax,
    )


def gst_failures(bill: Bill, checker) -> list:
    return verify_bill(bill, gst_checker=checker).failures_in(("gst",))


def failed_rules(bill: Bill, checker) -> set[str]:
    return {c.rule_id for c in gst_failures(bill, checker)}


# ------------------------------------------------------------------ part 1: lookup


def _brute_force_lookup(schedule: GstSchedule, digits: str, on_date: date) -> str | None:
    """Independent longest-prefix scan, used only to judge ``lookup``."""
    best: str | None = None
    for entry in schedule.entries:
        if not entry.covers(on_date):
            continue
        p = entry.hsn_prefix
        if digits.startswith(p) and len(p) in PREFIX_LENGTHS:
            if best is None or len(p) > len(best):
                best = p
    return best


def part_lookup(schedule: GstSchedule, on_date: date) -> dict:
    direct_total = direct_ok = 0
    for entry in schedule.entries:
        probe = entry.effective_from
        found = schedule.lookup(entry.hsn_prefix, probe)
        direct_total += 1
        direct_ok += int(found is not None and found.covers(probe))

    # Longest-prefix fallback: pad every prefix out to 8 digits with a filler that
    # is unlikely to collide, and with the digits of a sibling prefix, then check
    # the result against the brute-force scan.
    fallback_total = fallback_ok = 0
    fallback_examples: list[dict] = []
    for entry in schedule.entries:
        for filler in ("0", "1", "9"):
            padded = (entry.hsn_prefix + filler * 8)[:8]
            expected = _brute_force_lookup(schedule, padded, on_date)
            got = schedule.lookup(padded, on_date)
            got_prefix = got.hsn_prefix if got else None
            fallback_total += 1
            fallback_ok += int(got_prefix == expected)
            if len(fallback_examples) < 8 and expected is not None:
                fallback_examples.append(
                    {"hsn": padded, "resolved_prefix": got_prefix, "expected_prefix": expected}
                )

    # Unknown codes, exhaustively over the whole 4 digit space.
    covered = set(schedule.prefixes)
    unknown_total = unknown_none = 0
    for value in range(10_000):
        code = f"{value:04d}"
        if code in covered or code[:2] in covered:
            continue
        unknown_total += 1
        unknown_none += int(schedule.lookup(code, on_date) is None)

    return {
        "direct_prefix_rows": direct_total,
        "direct_prefix_resolved": direct_ok,
        "direct_resolution_rate": round(direct_ok / direct_total, 4) if direct_total else None,
        "padded_to_8_digit_probes": fallback_total,
        "padded_matching_brute_force": fallback_ok,
        "fallback_agreement_rate": round(fallback_ok / fallback_total, 4) if fallback_total else None,
        "fallback_examples": fallback_examples,
        "unknown_4_digit_codes_tried": unknown_total,
        "unknown_returned_none": unknown_none,
        "unknown_none_rate": round(unknown_none / unknown_total, 4) if unknown_total else None,
    }


# ----------------------------------------------------- part 2: rate perturbation


def part_rate_mismatch(schedule: GstSchedule, checker, on_date: date) -> dict:
    slabs = sorted(schedule.slabs(on_date))
    active = [e for e in schedule.entries if e.covers(on_date)]

    cases = detected = 0
    by_pair: dict[str, dict] = {}
    for n, entry in enumerate(active):
        for wrong in slabs:
            if wrong == entry.rate_percent:
                continue
            price = UNIT_PRICES_PAISE[n % len(UNIT_PRICES_PAISE)]
            bill = build_bill(
                f"rate-{entry.hsn_prefix}-{wrong}",
                [(entry.hsn_prefix, entry.rate_percent, price, 3)],
                on_date,
            )
            # Charge the wrong rate but keep the tax consistent with it, so only
            # the rate rule can fire and the recompute rule cannot do its work
            # for it.
            bill.lines[0].gst_rate = wrong
            tax = bill.lines[0].line_total.percent(wrong)
            bill.tax_total = tax
            bill.cgst, bill.sgst = tax.split_half()
            bill.grand_total = bill.subtotal + tax

            rules = failed_rules(bill, checker)
            hit = "GST.RATE_MATCH" in rules
            cases += 1
            detected += int(hit)
            key = f"{entry.rate_percent} -> {wrong}"
            cell = by_pair.setdefault(key, {"cases": 0, "detected": 0})
            cell["cases"] += 1
            cell["detected"] += int(hit)

    return {
        "label": "rule-engine detection of a perturbed rate, not field accuracy",
        "active_entries": len(active),
        "legal_slabs": [str(s) for s in slabs],
        "cases": cases,
        "detected": detected,
        "detection_rate": round(detected / cases, 4) if cases else None,
        "by_rate_pair": {
            k: {**v, "rate": round(v["detected"] / v["cases"], 4)} for k, v in sorted(by_pair.items())
        },
    }


# -------------------------------------------------------- part 3: recompute sweep


def _rows_for(schedule: GstSchedule, on_date: date, count: int) -> list[tuple[str, Decimal, int, int]]:
    active = [e for e in schedule.entries if e.covers(on_date) and e.rate_percent > 0]
    rows: list[tuple[str, Decimal, int, int]] = []
    for i in range(count):
        entry = active[i % len(active)]
        rows.append(
            (
                entry.hsn_prefix,
                entry.rate_percent,
                UNIT_PRICES_PAISE[i % len(UNIT_PRICES_PAISE)],
                (i % 4) + 1,
            )
        )
    return rows


def part_recompute(schedule: GstSchedule, checker, on_date: date, repeats: int) -> dict:
    by_lines: dict[str, dict] = {}
    for count in LINE_COUNTS:
        tolerance = max(RECOMPUTE_FLOOR, RECOMPUTE_PER_LINE * count)
        per_error: dict[str, dict] = {}
        for error in ERROR_SWEEP_PAISE:
            cases = detected = 0
            for rep in range(repeats):
                rows = _rows_for(schedule, on_date, count)
                rows = rows[rep:] + rows[:rep]          # rotate so repeats differ
                bill = build_bill(f"tax-{count}-{error}-{rep}", rows, on_date)
                sign = 1 if rep % 2 == 0 else -1
                bill.tax_total = bill.tax_total + Money.from_paise(sign * error)
                bill.cgst, bill.sgst = bill.tax_total.split_half()
                bill.grand_total = bill.subtotal + bill.tax_total
                cases += 1
                detected += int("GST.TAX_RECOMPUTE" in failed_rules(bill, checker))
            per_error[str(error)] = {
                "cases": cases,
                "detected": detected,
                "rate": round(detected / cases, 4) if cases else None,
            }
        caught_always = [
            int(e) for e in per_error if per_error[e]["detected"] == per_error[e]["cases"]
        ]
        by_lines[str(count)] = {
            "tolerance_paise": tolerance,
            "smallest_error_always_caught_paise": min(caught_always) if caught_always else None,
            "by_error_paise": per_error,
        }
    return {
        "tolerance_rule": f"max({RECOMPUTE_FLOOR}, {RECOMPUTE_PER_LINE} x taxed lines) paise",
        "error_sweep_paise": list(ERROR_SWEEP_PAISE),
        "by_line_count": by_lines,
    }


# -------------------------------------------------------- part 4: false alarms


def part_false_alarms(schedule: GstSchedule, checker, on_date: date) -> dict:
    bills = []
    active = [e for e in schedule.entries if e.covers(on_date)]
    for n, entry in enumerate(active):
        price = UNIT_PRICES_PAISE[n % len(UNIT_PRICES_PAISE)]
        bills.append(
            build_bill(
                f"clean-single-{entry.hsn_prefix}",
                [(entry.hsn_prefix, entry.rate_percent, price, (n % 7) + 1)],
                on_date,
            )
        )
    for count in LINE_COUNTS:
        for rep in range(5):
            rows = _rows_for(schedule, on_date, count)
            rows = rows[rep:] + rows[:rep]
            bills.append(build_bill(f"clean-{count}-{rep}", rows, on_date))

    offenders: list[dict] = []
    failures = 0
    for bill in bills:
        fails = gst_failures(bill, checker)
        failures += len(fails)
        if fails:
            offenders.append(
                {"bill_id": bill.bill_id, "rules": sorted({c.rule_id for c in fails})}
            )

    return {
        "clean_bills": len(bills),
        "gst_failures": failures,
        "offending_bills": offenders[:10],
    }


# ------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", default=str(DEFAULT_CSV))
    ap.add_argument(
        "--as-of",
        default="2026-04-01",
        help="date whose schedule is benchmarked, ISO format",
    )
    ap.add_argument("--repeats", type=int, default=5, help="bills per error size")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    on_date = date.fromisoformat(args.as_of)
    schedule = GstSchedule.load(args.schedule)
    checker = make_gst_checker(schedule)

    print(f"schedule {schedule.schedule_version}")
    print(f"  {schedule.unverified_rows} of {len(schedule.entries)} rows unverified")
    print(f"  judged as on {on_date.isoformat()}")

    t0 = time.time()
    lookup = part_lookup(schedule, on_date)
    rate = part_rate_mismatch(schedule, checker, on_date)
    recompute = part_recompute(schedule, checker, on_date, args.repeats)
    clean = part_false_alarms(schedule, checker, on_date)
    bench_s = time.time() - t0

    payload = {
        "protocol": (
            "Rule-engine benchmark, not a field-accuracy benchmark. There is no "
            "public corpus of annotated Indian GST invoices and the official rate "
            "sources (pib.gov.in, taxinformation.cbic.gov.in) could not be "
            "fetched, so no number here says whether our rate schedule is "
            "factually correct. Synthetic bills are generated from the schedule "
            "the engine is then judged against: part 1 tests longest-prefix "
            "lookup against an independent brute-force scan, part 2 perturbs a "
            "correct charged rate to each other legal slab and scores detection, "
            "part 3 sweeps a known paise error through the printed tax and "
            "reports the smallest error caught on every bill, part 4 confirms "
            "correctly built bills raise no GST failure. Rate correctness is out "
            "of scope and is declared unverified in the reference data."
        ),
        "schedule": schedule.describe(),
        "as_of": on_date.isoformat(),
        "lookup": lookup,
        "rate_mismatch": rate,
        "tax_recompute": recompute,
        "false_alarms": clean,
        "runtime_seconds": round(bench_s, 2),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "gst_rules.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\npart 1  longest-prefix lookup")
    print(f"  {'schedule rows resolving on their own dates':<46}"
          f"{lookup['direct_prefix_resolved']}/{lookup['direct_prefix_rows']}")
    print(f"  {'8-digit probes agreeing with brute force':<46}"
          f"{lookup['padded_matching_brute_force']}/{lookup['padded_to_8_digit_probes']}")
    print(f"  {'uncovered 4-digit codes returning None':<46}"
          f"{lookup['unknown_returned_none']}/{lookup['unknown_4_digit_codes_tried']}")

    print("\npart 2  rate perturbed to each other legal slab (engine test)")
    print(f"  {'cases':<46}{rate['cases']}")
    print(f"  {'detected by GST.RATE_MATCH':<46}{rate['detected']} "
          f"({rate['detection_rate']:.1%})")

    print("\npart 3  smallest injected tax error always caught")
    print(f"  {'lines':>6} {'tolerance p':>12} {'smallest caught p':>18}")
    for count in LINE_COUNTS:
        cell = recompute["by_line_count"][str(count)]
        smallest = cell["smallest_error_always_caught_paise"]
        print(f"  {count:>6} {cell['tolerance_paise']:>12} "
              f"{(smallest if smallest is not None else 'none in sweep'):>18}")

    print("\npart 4  false alarms")
    print(f"  {'clean bills':<46}{clean['clean_bills']}")
    print(f"  {'GST failures raised':<46}{clean['gst_failures']}")

    print(f"\nruntime {bench_s:.2f}s")
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
