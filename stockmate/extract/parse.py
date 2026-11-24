"""Turn recognised rows of text into a Bill.

What makes this hard is not the reading, it is that there is no such thing as a
receipt layout.  Across a few hundred real till rolls the same information
appears as ``NAME QTY RATE AMOUNT``, as ``NAME 2 x 30.000 60.000``, as a name on
one row with its figures on the next, and as a name with a single amount and no
count printed anywhere.  A parser written against any one of those shapes scores
well on the shop it was written for and badly everywhere else.

The approach taken here is to commit to as little layout as possible.  Three
things are assumed, and they are the three that hold across the corpus:

* money is right aligned, so the last number on a row is that row's amount;
* a document level figure announces itself in words next to the number, as
  subtotal, total, tax, cash or discount;
* a count, when it is printed at all, sits to the left of the money.

Everything else is inferred per row.  Where a figure is not printed it is left
as None rather than guessed, with one stated exception: a line with no printed
count is read as one unit, which is the same convention the dataset's own
annotation uses and which keeps the quantity comparison in the benchmark
meaningful rather than vacuous.

Every amount goes through ``stockmate.money.parse_money``, so no rupee value in
this module is ever a float.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from ..models import Bill, BillLine, Field
from ..money import Money
from .ocr import OcrLine, OcrResult, OcrWord, run_ocr

__all__ = ["parse_bill", "extract_bill", "ParsedRow", "classify_rows"]


# A money token.  Three shapes are accepted: grouped thousands such as 60.000 or
# 1,234.50, a plain run of four or more digits such as 17500, and a small amount
# carrying two decimal places such as 45.50.  Anything with letters in it is not
# money, which is what keeps a garbled detection out of the totals.
_AMOUNT_RE = re.compile(
    r"""^[(\[]?\s*[-+]?\s*(?:rs\.?|rp\.?|inr|idr|₹)?\s*(
          \d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?
        | \d{4,}(?:[.,]\d{1,2})?
        | \d{1,3}[.,]\d{2}
      )\s*(?:,-|[-/)\]*])*$""",
    re.VERBOSE | re.IGNORECASE,
)

# A bare two or three digit number.  On its own this is as likely to be a count,
# a table number or a line index as it is an amount, so it is only read as money
# when it is the only candidate on the row and sits at the right hand edge.
_LOOSE_AMOUNT_RE = re.compile(
    r"^[(\[]?\s*[-+]?\s*(?:rs\.?|rp\.?|inr|idr|₹)?\s*(\d{2,3})\s*(?:,-|[-/)\]*])*$", re.I
)

# A printed count.  Two digits is the ceiling for a bare number, because a bare
# three digit number on a receipt row is more often a small amount than a count
# of a hundred-odd units.  A larger count is accepted only when the layout marks
# it as one with a multiplication sign, or when it carries the decimal tail of
# zeros that tills print as 2.00.
_QTY_RE = re.compile(
    r"""^(?: [xX*]\s*(\d{1,3})
           | (\d{1,3})\s*[xX*]
           | (\d{1,3})[.,]0{1,2}
           | (\d{1,2})
        )$""",
    re.VERBOSE,
)

# A count folded into the description, as in "2 x Nasi Goreng" or "Teh 3x".
_QTY_IN_TEXT_RE = re.compile(r"(?:^|\s)(\d{1,3})\s*[xX*](?:\s|$)")

#: Keywords that mark a document level figure, in the order they are tested.
#: Order carries meaning.  "Subtotal" contains "total" and "total discount"
#: contains both, so the more specific label has to be tried first or every
#: discount line would be read as the grand total.
_DOC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("change", ("change", "kembali", "kembalian", "refund")),
    ("discount_total", ("disc", "diskon", "potongan", "voucher")),
    ("subtotal", ("subtotal", "subttl", "subtot")),
    ("tax_total", ("tax", "ppn", "pb1", "vat", "gst", "cgst", "sgst", "service")),
    ("round_off", ("rounding", "roundoff", "pembulatan")),
    ("grand_total", ("grandtotal", "nettotal", "total", "nett", "amountdue", "jumlah", "netto")),
    ("payment", ("cash", "tunai", "credit", "debit", "card", "edc", "bayar", "visa", "master")),
)

#: Header and footer wording that carries numbers but never an item.  Without
#: this list a telephone number or an invoice serial becomes a priced line.
_NOT_AN_ITEM = (
    "tel", "telp", "phone", "fax", "npwp", "gstin", "gst no", "invoice", "struk",
    "kasir", "cashier", "pelanggan", "customer", "date", "time", "table", "meja",
    "guest", "pax", "no.", "bill no", "receipt", "www", ".com", "terima kasih",
    "thank", "qty", "item", "price", "amount", "www.", "jl.", "jalan", "street",
)


def _normalise(text: str) -> str:
    """Lower case, letters only.

    The recogniser sometimes returns a row with its spaces dropped, so
    "TOTAL DISC" arrives as "TOTALDISCS".  Comparing on a letters-only string
    makes keyword matching indifferent to that.
    """
    return re.sub(r"[^a-z]", "", text.lower())


def _qty_of(text: str) -> Decimal | None:
    """Read a token as a printed count, or return None."""
    match = _QTY_RE.match(text.strip())
    if not match:
        return None
    digits = next((g for g in match.groups() if g), None)
    return Decimal(digits) if digits else None


def _amount_of(word: OcrWord) -> Money | None:
    if not _AMOUNT_RE.match(word.text):
        return None
    return parse_amount(word.text)


def parse_amount(text: str) -> Money | None:
    """Read one token as money, or return None.

    Thin wrapper over the shared parser so that this module has a single place
    where a string becomes a Money and no code path can accidentally build one
    from a float.
    """
    from ..money import parse_money

    return parse_money(text)


@dataclass
class ParsedRow:
    """One OCR row after its parts have been told apart."""

    line: OcrLine
    description: str
    quantity: Decimal | None
    amounts: list[tuple[OcrWord, Money]]
    doc_field: str | None            # which document figure this row announces
    is_rightmost_amount: bool

    @property
    def confidence(self) -> float:
        return self.line.confidence

    @property
    def has_letters(self) -> bool:
        return len(re.sub(r"[^A-Za-z]", "", self.description)) >= 2


#: Labels worth recognising even when the recogniser mangles them, with the
#: field each one fills.  These three are the words that decide whether a row is
#: a document figure or an item, and a single wrong character in "SUBTOTAL"
#: otherwise promotes the subtotal to a priced line and loses the real total.
_FUZZY_LABELS: tuple[tuple[str, str], ...] = (
    ("subtotal", "subtotal"),
    ("grandtotal", "grand_total"),
    ("total", "grand_total"),
)

#: Similarity needed before a mangled word is accepted as a label.  Set from the
#: corpus: at 0.8 the misreadings seen in practice are caught while ordinary item
#: names stay below the bar.
_FUZZY_THRESHOLD = 0.8


def _fuzzy_keyword(flat: str) -> str | None:
    """Match a label through one or two recognition errors.

    Only applied to a row whose whole text is about the length of a label, since
    a long item description can accidentally reach a high similarity score
    against a short word and there is no benefit in letting it.
    """
    from difflib import SequenceMatcher

    if not 4 <= len(flat) <= 12:
        return None
    for label, field_name in _FUZZY_LABELS:
        if SequenceMatcher(None, flat, label).ratio() >= _FUZZY_THRESHOLD:
            return field_name
    return None


#: The longest token that a short keyword is allowed to match inside.  Keywords
#: are tested against each word of the row rather than against the row as a
#: whole, because joining the row's words produces junctions that contain the
#: keyword by accident: "STEAMED CHEESEC" concatenates to a string holding "edc",
#: and an item would then be read as a card payment line.  Containment inside a
#: single long word is suppressed for the same reason, while short words such as
#: "TOTALDISCS" are still matched, which is what the space-dropping recogniser
#: needs.
_MAX_CONTAINING_TOKEN = 12


def _row_keyword(text: str) -> str | None:
    tokens = [t for t in (_normalise(part) for part in text.split()) if t]
    if not tokens:
        return None
    # First pass: the long labels, matched against the row's words joined
    # together, so that "Sub Total" printed as two words is still the subtotal
    # and is not claimed by the shorter "total" in the pass below.  Only labels
    # of six characters or more take part, since those are long enough that a
    # junction between two words will not produce one by chance.
    joined = "".join(tokens)
    for field_name, keys in _DOC_KEYWORDS:
        for key in keys:
            flat_key = key.replace(" ", "")
            if len(flat_key) >= 6 and flat_key in joined:
                return field_name

    for field_name, keys in _DOC_KEYWORDS:
        for key in keys:
            flat_key = key.replace(" ", "")
            for token in tokens:
                if token.startswith(flat_key):
                    return field_name
                if len(token) <= _MAX_CONTAINING_TOKEN and flat_key in token:
                    return field_name
    return _fuzzy_keyword(joined)


def _synthetic_words(line: OcrLine) -> list[OcrWord]:
    """Split a row's text into tokens when the word boxes are not available.

    Rows replayed from stored text, or built by hand in a test, carry no per-word
    geometry.  The tokens are laid out in reading order across the row's own box
    so that the left-to-right rules still hold; the positions are approximate and
    are used only for ordering, never for measurement.
    """
    tokens = line.text.split()
    if not tokens:
        return []
    x0, y0, x1, y1 = line.bbox
    step = max((x1 - x0) // len(tokens), 1)
    return [
        OcrWord(
            text=token,
            confidence=line.confidence,
            bbox=(x0 + index * step, y0, x0 + (index + 1) * step, y1),
        )
        for index, token in enumerate(tokens)
    ]


def classify_rows(result: OcrResult) -> list[ParsedRow]:
    """Split every row into description, count and money.

    The count is consumed before the amounts so that a token such as ``2.00``,
    which is shaped like both, is read as the count it almost always is when
    money follows it on the same row.
    """
    rows: list[ParsedRow] = []
    for line in result.lines:
        words = list(line.words) or _synthetic_words(line)

        # The count is identified before the amounts, because a token such as
        # 2.00 is shaped like both and is a count whenever real money follows it
        # on the same row.  Taking it out first is what stops a quantity from
        # being read as the unit price.
        quantity: Decimal | None = None
        qty_word: OcrWord | None = None
        for position, word in enumerate(words):
            candidate = _qty_of(word.text)
            if candidate is None:
                continue
            if any(
                _AMOUNT_RE.match(later.text) and _qty_of(later.text) is None
                for later in words[position + 1:]
            ):
                quantity, qty_word = candidate, word
                break

        payload = [w for w in words if w is not qty_word]
        strict = [(w, m) for w in payload if (m := _amount_of(w)) is not None]
        amounts = strict
        if not strict and payload:
            # Nothing looked like a formatted amount.  Allow a bare two or
            # three digit number, but only as the final token on the row.
            last = payload[-1]
            if _LOOSE_AMOUNT_RE.match(last.text) and (m := parse_amount(last.text)) is not None:
                amounts = [(last, m)]

        amount_ids = {id(w) for w, _ in amounts}
        rest = [w for w in payload if id(w) not in amount_ids]

        description = " ".join(
            w.text for w in rest if w is not qty_word
        ).strip(" :.-•|")

        if quantity is None:
            match = _QTY_IN_TEXT_RE.search(description)
            if match:
                quantity = Decimal(match.group(1))
                description = (description[: match.start()] + " " + description[match.end():]).strip()

        rows.append(
            ParsedRow(
                line=line,
                description=description,
                quantity=quantity,
                amounts=amounts,
                doc_field=_row_keyword(description) if description else None,
                is_rightmost_amount=bool(amounts) and amounts[-1][0] is words[-1],
            )
        )
    return rows


def _looks_like_noise(description: str) -> bool:
    """Decide whether a row's words describe something other than an item.

    Matching is per word rather than on the row as a whole, because the short
    entries in the list are also fragments of ordinary product names: a plain
    substring test rejects a hotel's coffee for containing "tel" and a numbered
    burger for containing "no.".  Multi-word and punctuated entries are matched
    as substrings, since those cannot collide in the same way.
    """
    low = description.lower()
    words = re.split(r"[\s,;:|]+", low)
    for entry in _NOT_AN_ITEM:
        if "." in entry or " " in entry:
            if entry in low:
                return True
            continue
        for word in words:
            if word == entry or (len(entry) >= 5 and word.startswith(entry)):
                return True
    return False


def _derive_unit_price(line_total: Money, quantity: Decimal) -> Money | None:
    """Unit price implied by a printed total and count.

    Receipts frequently print the count and the line total but not the rate.
    The rate is reconstructed here so that the arithmetic checks have something
    to verify, using exact integer paise and the same half-up convention as the
    rest of the system.
    """
    if not quantity or quantity == 0:
        return None
    from ..money import quantize_half_up

    return Money(int(quantize_half_up(Decimal(line_total.paise) / quantity)))


def _build_lines(rows: list[ParsedRow]) -> list[BillLine]:
    """Pull item lines out of the classified rows.

    Two layouts are handled together.  The common one has everything on a single
    row.  The other splits a long item name onto its own row with the figures
    beneath it, which is dealt with by letting an amount-only row adopt the
    description of the row immediately above when that row had words and no
    money of its own.
    """
    lines: list[BillLine] = []
    pending_name: ParsedRow | None = None

    for row in rows:
        if row.doc_field is not None:
            # A row that names a document figure is never an item, even if it
            # also carries a count, because the count there is the menu quantity
            # summary that some tills print beside the total.
            pending_name = None
            continue

        if not row.amounts:
            # Candidate for the name of a split line.  Only a row with real
            # words qualifies; a stray mark would otherwise capture the next
            # amount and invent an item.
            pending_name = row if row.has_letters and not _looks_like_noise(row.description) else None
            continue

        description = row.description
        confidence = row.confidence
        quantity = row.quantity
        used_words = [w for w, _ in row.amounts]

        if not row.has_letters and pending_name is not None:
            description = pending_name.description
            confidence = min(confidence, pending_name.confidence)
            if quantity is None:
                quantity = pending_name.quantity
        pending_name = None

        if len(re.sub(r"[^A-Za-z]", "", description)) < 2 or _looks_like_noise(description):
            continue
        if not row.is_rightmost_amount:
            # Money that is not at the right hand edge of its row is usually a
            # serial number or a date fragment that happened to parse.  The row
            # is still allowed to supply a name to the figures below it, because
            # an item code printed to the right of the name is exactly the layout
            # that puts the amounts on their own row.
            pending_name = row
            continue

        values = [m for _, m in row.amounts]
        line_total = values[-1]
        unit_price = values[-2] if len(values) >= 2 else None

        if quantity is None:
            # No printed count.  One unit is the reading the annotation itself
            # uses, and it is recorded in the raw text so the assumption is
            # visible to anyone reviewing the extraction.
            quantity = Decimal(1)
            qty_note = "qty assumed 1"
        else:
            qty_note = f"qty read as {quantity}"

        if unit_price is None:
            unit_price = _derive_unit_price(line_total, quantity)
        elif unit_price.paise == line_total.paise and quantity != 1:
            # Both columns show the same figure, which on these receipts means
            # the rate column was not printed and the total was repeated.
            unit_price = _derive_unit_price(line_total, quantity)

        lines.append(
            BillLine(
                index=len(lines),
                description=description.strip(),
                quantity=quantity,
                unit_price=unit_price,
                line_total=line_total,
                confidence=min([confidence] + [w.confidence for w in used_words]),
                raw_text=f"{row.line.text} [{qty_note}]",
            )
        )
    return lines


def _collect_doc_fields(rows: list[ParsedRow]) -> dict[str, tuple[Money, float, ParsedRow]]:
    """Find the document level figures by their printed label.

    Where a label appears more than once, the last occurrence wins.  Tills print
    a running total and then a final one, and it is the final one that is the
    document's own figure.  A label with no number beside it is allowed to take
    the number from the row below, which is how wide thermal rolls wrap.
    """
    found: dict[str, tuple[Money, float, ParsedRow]] = {}
    for index, row in enumerate(rows):
        if row.doc_field is None:
            continue
        amounts = row.amounts
        source = row
        if not amounts and index + 1 < len(rows):
            nxt = rows[index + 1]
            if nxt.amounts and not nxt.has_letters:
                amounts, source = nxt.amounts, nxt
        if not amounts:
            continue
        word, value = amounts[-1]
        found[row.doc_field] = (value, min(row.confidence, word.confidence), source)
    return found


def parse_bill(result: OcrResult, bill_id: str, *, source_image: str | None = None) -> Bill:
    """Assemble a Bill from one OCR result.

    The grand total is taken from a row labelled as a total.  When no such row
    carries a number, the amount tendered is used instead, because a payment line
    on a cash receipt states the same figure and a bill with no total at all is
    useless to every check downstream.  That substitution is recorded in the
    field's source so it is never mistaken for a printed total.
    """
    rows = classify_rows(result)
    lines = _build_lines(rows)
    doc = _collect_doc_fields(rows)

    bill = Bill(
        bill_id=bill_id,
        source_image=source_image,
        lines=lines,
        ocr_ms=result.elapsed_ms,
    )
    fields: dict[str, Field] = {}

    def take(name: str, attr: str, *, absolute: bool = False, source: str = "ocr") -> None:
        if name not in doc:
            return
        value, confidence, row = doc[name]
        if absolute:
            value = Money.from_paise(abs(value.paise))
        setattr(bill, attr, value)
        fields[attr] = Field(
            value=value,
            confidence=confidence,
            source=source,
            raw_text=row.line.text,
            bbox=row.line.bbox,
        )

    take("subtotal", "subtotal")
    take("tax_total", "tax_total")
    take("discount_total", "discount_total", absolute=True)
    take("round_off", "round_off")
    take("grand_total", "grand_total")

    if bill.grand_total is None and "payment" in doc:
        value, confidence, row = doc["payment"]
        bill.grand_total = value
        fields["grand_total"] = Field(
            value=value,
            # A tendered amount is weaker evidence than a printed total, so the
            # confidence is deliberately reduced rather than inherited.
            confidence=min(confidence, 0.7),
            source="derived",
            raw_text=row.line.text,
            bbox=row.line.bbox,
        )

    for line in lines:
        fields[f"line_{line.index + 1}"] = Field(
            value=line.line_total,
            confidence=line.confidence,
            source="ocr",
            raw_text=line.raw_text,
        )

    bill.fields = fields
    return bill


def extract_bill(image, bill_id: str, *, prepare: bool = True) -> tuple[Bill, OcrResult]:
    """Prepare, recognise and parse one bill image in one call.

    Returns the OCR result alongside the bill because the caller usually wants
    both: the bill to verify and the raw rows to show the owner when a field
    comes back needing review.
    """
    if prepare:
        import numpy as np

        from .prep import prepare_for_ocr

        if not isinstance(image, np.ndarray):
            array = np.array(image.convert("RGB"))[:, :, ::-1].copy()
        else:
            array = image
        image = prepare_for_ocr(array)

    result = run_ocr(image)
    return parse_bill(result, bill_id), result
