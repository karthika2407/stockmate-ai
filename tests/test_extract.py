"""Tests for the OCR extraction pipeline.

Each test pins down a behaviour that an extraction bug would otherwise hide
until it showed up as a wrong rupee figure in front of a shopkeeper: that
receipt-style amounts become the exact paise they print, that words on the same
printed line are grouped as one row even when their boxes differ in height, that
an e-invoice QR payload is read without the signature ever being called verified,
that a blank photograph yields an empty bill rather than an exception, and that
the whole pipeline still reads a real annotated receipt end to end.

The last test needs the CORD shards and the ONNX models, so it is skipped rather
than failed where they are absent.  The rest run with no data at all.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stockmate.extract.ocr import OcrLine, OcrResult, OcrWord, group_into_rows
from stockmate.extract.parse import classify_rows, parse_bill
from stockmate.extract.prep import (
    adaptive_threshold,
    deskew,
    estimate_skew_angle,
    normalise_contrast,
    prepare_for_ocr,
    upscale_if_small,
)
from stockmate.extract.qr import decode_qr, parse_einvoice_payload
from stockmate.money import Money, parse_money

DATA_DIR = ROOT.parent / "_datasets" / "cord_v2"


# ------------------------------------------------------------------- money text


@pytest.mark.parametrize(
    "text, paise",
    [
        ("Rs 1,234.50", 123450),
        ("₹900/-", 90000),
        ("18,000", 1800000),
        ("60.000", 6000000),          # Indonesian thousands separator
        ("1.234,50", 123450),         # European ordering
        ("17500", 1750000),
        ("-60.000", -6000000),        # a discount line
        ("45.50", 4550),
        ("INR 0.01", 1),
    ],
)
def test_parse_money_reads_receipt_amounts(text, paise):
    parsed = parse_money(text)
    assert parsed is not None, text
    assert parsed.paise == paise
    assert isinstance(parsed.paise, int)


def test_parse_money_returns_none_rather_than_zero():
    # A missing amount must never become Money(0), because zero is a claim and
    # None is the absence of one.
    assert parse_money("") is None
    assert parse_money("TOTAL") is None


# --------------------------------------------------------------- row grouping


def _word(text, x0, y0, x1, y1, confidence=0.99):
    return OcrWord(text=text, confidence=confidence, bbox=(x0, y0, x1, y1))


def test_rows_group_by_vertical_overlap_not_by_centre():
    # The amount on a till roll is often set larger than the item name beside
    # it, so the two boxes share a line while their centres sit apart.
    words = [
        _word("NASI GORENG", 20, 100, 180, 118),
        _word("2", 200, 96, 214, 122),
        _word("30.000", 240, 94, 320, 124),
        _word("TOTAL", 20, 160, 90, 178),
        _word("60.000", 240, 158, 320, 180),
    ]
    rows = group_into_rows(words)
    assert len(rows) == 2
    assert rows[0].text == "NASI GORENG 2 30.000"
    assert rows[1].text == "TOTAL 60.000"
    # Rows run down the page and words run left to right within a row.
    assert rows[0].centre_y < rows[1].centre_y
    assert [w.bbox[0] for w in rows[0].words] == sorted(w.bbox[0] for w in rows[0].words)


def test_rows_split_when_boxes_do_not_overlap():
    words = [_word("A", 0, 0, 10, 20), _word("B", 0, 40, 10, 60)]
    assert len(group_into_rows(words)) == 2


def test_group_into_rows_handles_no_words():
    assert group_into_rows([]) == []


# ------------------------------------------------------------------- parsing


def _result(rows: list[list[OcrWord]]) -> OcrResult:
    words = [w for row in rows for w in row]
    return OcrResult(lines=group_into_rows(words), elapsed_ms=0.0)


def test_parses_qty_rate_amount_layout():
    result = _result([
        [_word("NASI GORENG", 20, 100, 180, 118), _word("2", 200, 100, 214, 118),
         _word("30.000", 240, 100, 320, 118), _word("60.000", 360, 100, 440, 118)],
        [_word("TOTAL", 20, 160, 90, 178), _word("60.000", 360, 160, 440, 178)],
    ])
    bill = parse_bill(result, "unit-1")
    assert len(bill.lines) == 1
    line = bill.lines[0]
    assert line.description == "NASI GORENG"
    assert line.quantity == Decimal(2)
    assert line.unit_price == Money.from_rupees("30000")
    assert line.line_total == Money.from_rupees("60000")
    assert bill.grand_total == Money.from_rupees("60000")
    assert "grand_total" in bill.fields


def test_parses_name_on_one_row_amounts_on_the_next():
    result = _result([
        [_word("AYAM BAKAR SPECIAL", 20, 100, 260, 118)],
        [_word("1", 200, 130, 212, 148), _word("45.000", 360, 130, 440, 148)],
        [_word("CASH", 20, 200, 80, 218), _word("45.000", 360, 200, 440, 218)],
    ])
    bill = parse_bill(result, "unit-2")
    assert [l.description for l in bill.lines] == ["AYAM BAKAR SPECIAL"]
    assert bill.lines[0].line_total == Money.from_rupees("45000")
    # No row said "total", so the tendered amount stands in for it and is marked
    # as derived rather than read.
    assert bill.grand_total == Money.from_rupees("45000")
    assert bill.fields["grand_total"].source == "derived"


def test_keyword_rows_are_not_read_as_items():
    result = _result([
        [_word("TEH MANIS", 20, 100, 140, 118), _word("12.000", 360, 100, 440, 118)],
        [_word("Subtotal", 20, 140, 110, 158), _word("12.000", 360, 140, 440, 158)],
        [_word("TOTALDISCS", 20, 170, 150, 188), _word("-2.000", 360, 170, 440, 188)],
        [_word("PPN", 20, 200, 60, 218), _word("1.000", 360, 200, 440, 218)],
        [_word("TOTAL", 20, 230, 90, 248), _word("11.000", 360, 230, 440, 248)],
    ])
    bill = parse_bill(result, "unit-3")
    assert [l.description for l in bill.lines] == ["TEH MANIS"]
    assert bill.subtotal == Money.from_rupees("12000")
    assert bill.discount_total == Money.from_rupees("2000")   # held unsigned
    assert bill.tax_total == Money.from_rupees("1000")
    assert bill.grand_total == Money.from_rupees("11000")


def test_low_confidence_fields_are_reported():
    result = _result([
        [_word("KOPI SUSU", 20, 100, 130, 118, confidence=0.41),
         _word("18.000", 360, 100, 440, 118, confidence=0.41)],
        [_word("TOTAL", 20, 140, 90, 158, confidence=0.98),
         _word("18.000", 360, 140, 440, 158, confidence=0.98)],
    ])
    bill = parse_bill(result, "unit-4")
    assert "line_1" in bill.low_confidence_fields()
    assert "grand_total" not in bill.low_confidence_fields()


def test_quantity_token_is_not_mistaken_for_money():
    rows = classify_rows(_result([
        [_word("ES TEH", 20, 100, 90, 118), _word("2.00", 200, 100, 250, 118),
         _word("8.000", 360, 100, 440, 118)],
    ]))
    assert rows[0].quantity == Decimal(2)
    assert [str(m) for _, m in rows[0].amounts] == ["8,000.00"]


# ------------------------------------------------------------------------- QR


def _jws(payload: dict) -> str:
    """Build a compact JWS with a signature that is deliberately meaningless."""
    def seg(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    header = seg(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    body = seg(json.dumps(payload).encode())
    return f"{header}.{body}.{seg(b'not-a-real-signature')}"


SAMPLE_EINVOICE = {
    "SellerGstin": "33AABCU9603R1ZM",
    "BuyerGstin": "33AAACR5055K1Z7",
    "DocNo": "INV/2026/0417",
    "DocTyp": "INV",
    "DocDt": "17/04/2026",
    "TotInvVal": "12480.75",
    "ItemCnt": "4",
    "MainHsnCode": "1006",
    "Irn": "a" * 64,
}


def test_jws_payload_is_decoded_without_claiming_verification():
    payload = parse_einvoice_payload(_jws(SAMPLE_EINVOICE))
    assert payload.seller_gstin == "33AABCU9603R1ZM"
    assert payload.buyer_gstin == "33AAACR5055K1Z7"
    assert payload.doc_number == "INV/2026/0417"
    assert payload.doc_date == date(2026, 4, 17)
    assert payload.total_value == Money.from_rupees("12480.75")
    assert payload.total_value.paise == 1248075
    assert payload.irn == "a" * 64
    # The signature is never checked, so this flag is never true.
    assert payload.signature_verified is False


def test_plain_json_qr_is_read_as_a_payload():
    payload = parse_einvoice_payload(json.dumps(SAMPLE_EINVOICE))
    assert payload.doc_number == "INV/2026/0417"
    assert payload.signature_verified is False


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "https://example-supplier.invalid/bill/4417",
        "not base64 at all",
        "aaa.!!!not-base64!!!.bbb",
        "a.b",
        json.dumps([1, 2, 3]),
    ],
)
def test_non_jws_qr_content_does_not_raise(raw):
    payload = parse_einvoice_payload(raw)
    assert payload.raw == raw
    assert payload.signature_verified is False
    assert payload.doc_number is None
    assert payload.total_value is None


# -------------------------------------------------------------- blank images


def _blank(width=600, height=800):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_preparation_survives_a_blank_image():
    blank = _blank()
    assert estimate_skew_angle(blank) == 0.0
    assert deskew(blank).shape == blank.shape
    assert normalise_contrast(blank).shape == blank.shape[:2]
    assert adaptive_threshold(blank).shape == blank.shape[:2]
    prepared = prepare_for_ocr(blank)
    assert prepared.ndim == 3 and prepared.shape[2] == 3


def test_upscale_only_touches_small_images():
    small = np.full((120, 90, 3), 255, dtype=np.uint8)
    assert min(upscale_if_small(small).shape[:2]) >= 360
    big = _blank(1200, 1600)
    assert upscale_if_small(big).shape == big.shape


def test_blank_image_yields_an_empty_bill_not_an_error():
    assert decode_qr(_blank()) is None
    bill = parse_bill(OcrResult(lines=[], elapsed_ms=0.0), "blank")
    assert bill.lines == []
    assert bill.grand_total is None
    assert bill.low_confidence_fields() == []


def test_a_row_with_no_words_is_still_parsed():
    # Callers may build an OcrLine by hand, for example when replaying stored
    # text, and the parser must not depend on the word list being present.
    line = OcrLine(text="TOTAL 60.000", confidence=0.9, bbox=(0, 0, 100, 20), centre_y=10.0)
    bill = parse_bill(OcrResult(lines=[line], elapsed_ms=0.0), "replay")
    assert bill.grand_total == Money.from_rupees("60000")


# ----------------------------------------------------------------- end to end


@pytest.mark.skipif(
    not list(DATA_DIR.glob("*test*.parquet")),
    reason="CORD-v2 shards are not present",
)
def test_end_to_end_on_a_real_cord_receipt():
    from stockmate.datasets.receipts import cord_to_bill, iter_cord_rows
    from stockmate.extract.parse import extract_bill

    receipt = next(iter_cord_rows(DATA_DIR, splits=("test",), limit=1))
    reference = cord_to_bill(receipt)
    bill, result = extract_bill(receipt.image(), receipt.receipt_id)

    assert result.lines, "the engine read nothing from a real receipt"
    assert result.elapsed_ms > 0
    assert bill.ocr_ms == result.elapsed_ms
    # This particular receipt is a single discounted ticket; what is asserted is
    # that the printed total is read exactly, which is the figure every later
    # check depends on.
    assert reference.grand_total is not None
    assert bill.grand_total == reference.grand_total
    assert all(isinstance(l.line_total.paise, int) for l in bill.lines if l.line_total)
