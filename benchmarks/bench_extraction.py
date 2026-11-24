"""Benchmark: how much of a receipt does the extractor actually read correctly?

Protocol
--------
1. Stream annotated CORD-v2 receipts with their photographs attached.
2. For each one, build the reference bill from the annotation alone, then run the
   real pipeline on the image: preparation, OCR, parsing.  The reference and the
   candidate therefore differ only by extraction error.
3. Score four things separately, because they fail for different reasons:
   the number of lines found, the grand total, the subtotal, and the content of
   the lines that were found.
4. Score line content only over pairs that could be matched, and report the
   match rate alongside, so a parser that finds two lines and gets both right
   cannot look better than one that finds ten and gets nine.

Matching is by description similarity using difflib, greedily, best pair first.
That is the only sound way to compare here: the extractor's line order follows
the printed page while the annotation's follows its own, and the descriptions
themselves carry recognition errors, so neither index nor exact text can be used
as the key.

What the numbers do not say
---------------------------
CORD is Indonesian retail and restaurant paper, not Indian GST invoices, so
these figures measure reading a photographed till roll and nothing more.  The
annotation is also not a transcript: it records the fields of interest, so a
receipt whose annotation lists one item may well have a second item printed on
it, and the line-count figures carry that disagreement.  This is stated because
the exact-match rate below is depressed by it and would otherwise look like a
parser fault.

Run:
    python benchmarks/bench_extraction.py --limit 100
    python benchmarks/bench_extraction.py --limit 100 --no-prep
Writes results/extraction_accuracy.json and results/extraction_per_receipt.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockmate.datasets.receipts import cord_to_bill, iter_cord_rows  # noqa: E402
from stockmate.extract.ocr import run_ocr, warmup  # noqa: E402
from stockmate.extract.parse import parse_bill  # noqa: E402
from stockmate.extract.prep import prepare_for_ocr  # noqa: E402

#: Below this similarity two descriptions are not the same item, and pairing them
#: would credit or blame the parser for a line it never read.
MATCH_FLOOR = 0.45


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.upper().strip(), b.upper().strip()).ratio()


def match_lines(got, ref) -> list[tuple[int, int, float]]:
    """Pair extracted lines with annotated lines, best description match first.

    Greedy rather than optimal.  A full assignment solution would change almost
    nothing on receipts of this size, where the competing candidates are usually
    distinct enough that the best pair is unambiguous.
    """
    scored = sorted(
        (
            (_similar(g.description, r.description), gi, ri)
            for gi, g in enumerate(got)
            for ri, r in enumerate(ref)
        ),
        reverse=True,
    )
    used_got: set[int] = set()
    used_ref: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, gi, ri in scored:
        if score < MATCH_FLOOR or gi in used_got or ri in used_ref:
            continue
        used_got.add(gi)
        used_ref.add(ri)
        pairs.append((gi, ri, score))
    return pairs


def _paise(value) -> int | None:
    return value.paise if value is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT.parent / "_datasets" / "cord_v2"))
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument(
        "--no-prep",
        action="store_true",
        help="skip the image preparation stage, to measure what it is worth",
    )
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()

    warm_ms = warmup()
    print(f"engine warm in {warm_ms:.0f} ms; preparation {'off' if args.no_prep else 'on'}")

    rows: list[dict] = []
    ocr_ms: list[float] = []
    end_to_end_ms: list[float] = []
    prep_ms: list[float] = []

    t0 = time.time()
    for receipt in iter_cord_rows(
        args.data, splits=tuple(args.splits), limit=args.limit, with_images=True
    ):
        image = receipt.image()
        if image is None:
            continue
        array = np.array(image.convert("RGB"))[:, :, ::-1].copy()

        start = time.perf_counter()
        if args.no_prep:
            prepared = array
            prep_elapsed = 0.0
        else:
            prepared = prepare_for_ocr(array)
            prep_elapsed = (time.perf_counter() - start) * 1000.0
        result = run_ocr(prepared)
        bill = parse_bill(result, receipt.receipt_id, source_image=receipt.source_file)
        total_elapsed = (time.perf_counter() - start) * 1000.0

        ref = cord_to_bill(receipt)
        pairs = match_lines(bill.lines, ref.lines)

        qty_exact = sum(
            1 for gi, ri, _ in pairs if bill.lines[gi].quantity == ref.lines[ri].quantity
        )
        total_exact = sum(
            1
            for gi, ri, _ in pairs
            if _paise(bill.lines[gi].line_total) is not None
            and _paise(bill.lines[gi].line_total) == _paise(ref.lines[ri].line_total)
        )

        ocr_ms.append(result.elapsed_ms)
        end_to_end_ms.append(total_elapsed)
        prep_ms.append(prep_elapsed)

        rows.append({
            "bill_id": receipt.bill_id if hasattr(receipt, "bill_id") else receipt.receipt_id,
            "ocr_rows": len(result.lines),
            "mean_ocr_confidence": round(result.mean_confidence, 4),
            "lines_ref": len(ref.lines),
            "lines_got": len(bill.lines),
            "lines_matched": len(pairs),
            "qty_exact": qty_exact,
            "line_total_exact": total_exact,
            "grand_total_ref_paise": _paise(ref.grand_total),
            "grand_total_got_paise": _paise(bill.grand_total),
            "grand_total_exact": int(
                ref.grand_total is not None
                and _paise(bill.grand_total) == _paise(ref.grand_total)
            ),
            "subtotal_ref_paise": _paise(ref.subtotal),
            "subtotal_got_paise": _paise(bill.subtotal),
            "subtotal_exact": int(
                ref.subtotal is not None and _paise(bill.subtotal) == _paise(ref.subtotal)
            ),
            "low_confidence_fields": len(bill.low_confidence_fields()),
            "prep_ms": round(prep_elapsed, 1),
            "ocr_ms": round(result.elapsed_ms, 1),
            "end_to_end_ms": round(total_elapsed, 1),
        })
    bench_s = time.time() - t0

    n = len(rows)
    if n == 0:
        print("no receipts with images were read; nothing to report")
        return 1

    line_errors = [abs(r["lines_got"] - r["lines_ref"]) for r in rows]
    with_total = [r for r in rows if r["grand_total_ref_paise"] is not None]
    with_subtotal = [r for r in rows if r["subtotal_ref_paise"] is not None]
    total_pairs = sum(r["lines_matched"] for r in rows)
    ref_lines = sum(r["lines_ref"] for r in rows)
    got_lines = sum(r["lines_got"] for r in rows)

    def rate(hits: int, base: int) -> float | None:
        return round(hits / base, 4) if base else None

    payload = {
        "protocol": (
            "real pipeline (prepare, OCR, parse) run on annotated receipt "
            "photographs; every figure scored against the annotation, with line "
            "content scored only over description-matched pairs"
        ),
        "dataset": {
            "source": "CORD-v2 (naver-clova-ix/cord-v2), CC BY 4.0",
            "splits": args.splits,
            "receipts": n,
            "annotated_lines": ref_lines,
            "extracted_lines": got_lines,
            "caveat": (
                "Indonesian retail and restaurant receipts, not Indian GST "
                "invoices; the annotation records fields of interest rather than "
                "a full transcript of the paper"
            ),
        },
        "pipeline": {
            "preparation": not args.no_prep,
            "ocr_engine": "rapidocr_onnxruntime (ONNX, CPU, offline)",
            "warmup_ms": round(warm_ms, 1),
        },
        "line_count": {
            "exact_match_rate": rate(sum(1 for e in line_errors if e == 0), n),
            "mean_absolute_error": round(statistics.fmean(line_errors), 3),
            "median_absolute_error": statistics.median(line_errors),
            "over_extracted_receipts": sum(1 for r in rows if r["lines_got"] > r["lines_ref"]),
            "under_extracted_receipts": sum(1 for r in rows if r["lines_got"] < r["lines_ref"]),
        },
        "grand_total": {
            "receipts_with_annotation": len(with_total),
            "found": sum(1 for r in with_total if r["grand_total_got_paise"] is not None),
            "exact_match_rate": rate(
                sum(r["grand_total_exact"] for r in with_total), len(with_total)
            ),
            "exact_match_rate_when_found": rate(
                sum(r["grand_total_exact"] for r in with_total),
                sum(1 for r in with_total if r["grand_total_got_paise"] is not None),
            ),
        },
        "subtotal": {
            "receipts_with_annotation": len(with_subtotal),
            "found": sum(1 for r in with_subtotal if r["subtotal_got_paise"] is not None),
            "exact_match_rate": rate(
                sum(r["subtotal_exact"] for r in with_subtotal), len(with_subtotal)
            ),
        },
        "per_line": {
            "annotated_lines": ref_lines,
            "matched_pairs": total_pairs,
            "match_rate_over_annotated_lines": rate(total_pairs, ref_lines),
            "quantity_exact_rate": rate(sum(r["qty_exact"] for r in rows), total_pairs),
            "line_total_exact_rate": rate(sum(r["line_total_exact"] for r in rows), total_pairs),
            "line_total_exact_over_annotated": rate(
                sum(r["line_total_exact"] for r in rows), ref_lines
            ),
        },
        "latency_ms": {
            "mean_prepare": round(statistics.fmean(prep_ms), 1),
            "mean_ocr": round(statistics.fmean(ocr_ms), 1),
            "median_ocr": round(statistics.median(ocr_ms), 1),
            "mean_end_to_end": round(statistics.fmean(end_to_end_ms), 1),
            "p90_end_to_end": round(sorted(end_to_end_ms)[int(0.9 * (n - 1))], 1),
        },
        "runtime_seconds": round(bench_s, 2),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    json_path = out / f"extraction_accuracy{tag}.json"
    csv_path = out / f"extraction_per_receipt{tag}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{n} receipts, {ref_lines} annotated lines, {got_lines} extracted lines")
    print(f"\n{'measure':<34} {'base':>7} {'rate':>9}")
    print("-" * 52)
    lc = payload["line_count"]
    pl = payload["per_line"]
    print(f"{'line count exact':<34} {n:>7} {lc['exact_match_rate']:>9.1%}")
    print(f"{'line count mean abs error':<34} {n:>7} {lc['mean_absolute_error']:>9.2f}")
    print(f"{'grand total exact':<34} {len(with_total):>7} "
          f"{payload['grand_total']['exact_match_rate']:>9.1%}")
    print(f"{'subtotal exact':<34} {len(with_subtotal):>7} "
          f"{payload['subtotal']['exact_match_rate']:>9.1%}")
    print(f"{'lines matched to annotation':<34} {ref_lines:>7} "
          f"{pl['match_rate_over_annotated_lines']:>9.1%}")
    print(f"{'quantity exact (matched pairs)':<34} {total_pairs:>7} "
          f"{pl['quantity_exact_rate']:>9.1%}")
    print(f"{'line total exact (matched pairs)':<34} {total_pairs:>7} "
          f"{pl['line_total_exact_rate']:>9.1%}")
    print("-" * 52)
    lat = payload["latency_ms"]
    print(f"\nlatency: prepare {lat['mean_prepare']:.0f} ms, OCR {lat['mean_ocr']:.0f} ms, "
          f"end to end {lat['mean_end_to_end']:.0f} ms (p90 {lat['p90_end_to_end']:.0f} ms)")
    print(f"written: {json_path}")
    print(f"         {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
