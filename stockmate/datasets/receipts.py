"""Read CORD-v2 receipts into the Bill structures the verifier consumes.

Source
------
CORD: A Consolidated Receipt Dataset for Post-OCR Parsing (Park et al., 2019),
distributed as ``naver-clova-ix/cord-v2`` under CC BY 4.0.  One thousand
photographed receipts with line-item level annotation: item name, count, unit
price, line total, subtotal, discount, tax and grand total.

Why receipts and not Indian GST invoices
----------------------------------------
There is no openly licensed corpus of photographed Indian GST supplier
invoices.  CORD is the closest public stand-in that carries the structure the
arithmetic checks need, namely a quantity, a price and a stated total per line
plus a document-level total.  It is Indonesian retail and restaurant paper, so
it is used here to measure extraction and arithmetic verification only.  The
GST-specific rules are not evaluated on it, and no claim about Indian invoices
is drawn from these results.

A practical consequence worth knowing before reading any number produced from
this data: real receipts are not internally consistent.  Lines are discounted
after printing, service charges appear without a line, and totals are rounded in
ways the printed figures do not explain.  The loader therefore reports, for every
receipt, whether its own arithmetic closes, and the benchmarks separate receipts
that balance from those that do not instead of assuming the ground truth is
clean.
"""

from __future__ import annotations

import glob
import io
import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator

from ..models import Bill, BillLine
from ..money import Money

__all__ = ["CordReceipt", "load_cord", "cord_to_bill", "iter_cord_rows"]


def _dec(value: object) -> Decimal | None:
    """CORD prints numbers as strings such as '60.000' and '1.234.500'.

    Indonesian receipts use a dot as the thousands separator, so '60.000' is
    sixty thousand rupiah and not sixty.  Three trailing digits after the last
    dot are therefore a thousands group, not a fraction, unless there are
    exactly two of them.
    """
    if value is None:
        return None
    s = str(value).strip().replace(" ", "")
    if not s:
        return None
    negative = s.startswith("-")
    s = s.lstrip("-+")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".,")
    if not s:
        return None

    if "," in s and "." in s:
        dec_sep = "," if s.rfind(",") > s.rfind(".") else "."
    elif "," in s:
        dec_sep = "," if len(s.split(",")[-1]) == 2 else ""
    elif "." in s:
        tail = s.split(".")[-1]
        dec_sep = "." if len(tail) == 2 else ""
    else:
        dec_sep = ""

    if dec_sep:
        whole, _, frac = s.rpartition(dec_sep)
        digits = "".join(ch for ch in whole if ch.isdigit()) or "0"
        frac_digits = "".join(ch for ch in frac if ch.isdigit())[:2]
        s = f"{digits}.{frac_digits}" if frac_digits else digits
    else:
        s = "".join(ch for ch in s if ch.isdigit())

    try:
        out = Decimal(s or "0")
    except InvalidOperation:
        return None
    return -out if negative else out


def _as_list(node: object) -> list[dict]:
    if node is None:
        return []
    if isinstance(node, dict):
        return [node]
    if isinstance(node, list):
        return [n for n in node if isinstance(n, dict)]
    return []


@dataclass
class CordReceipt:
    """One annotated receipt, before it becomes a Bill."""

    receipt_id: str
    split: str
    image_bytes: bytes | None
    width: int
    height: int
    gt_parse: dict
    source_file: str = ""

    @property
    def menu_items(self) -> list[dict]:
        return _as_list(self.gt_parse.get("menu"))

    def image(self):
        """Decode lazily; the shards hold around 2 GB of JPEG in total."""
        if not self.image_bytes:
            return None
        from PIL import Image

        return Image.open(io.BytesIO(self.image_bytes))


def iter_cord_rows(
    data_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("test",),
    limit: int | None = None,
    with_images: bool = True,
) -> Iterator[CordReceipt]:
    """Stream receipts out of the downloaded parquet shards."""
    import pyarrow.parquet as pq

    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "*.parquet")))
    files = [f for f in files if any(f"_{s}_" in Path(f).name for s in splits)]
    if not files:
        raise FileNotFoundError(f"no parquet shards for splits {splits} in {data_dir}")

    seen = 0
    for path in files:
        split = next((s for s in splits if f"_{s}_" in Path(path).name), "unknown")
        pf = pq.ParquetFile(path)
        columns = ["image", "ground_truth"] if with_images else ["ground_truth"]
        for batch in pf.iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                gt = json.loads(row["ground_truth"])
                meta = gt.get("meta", {})
                size = meta.get("image_size", {})
                img = row.get("image") or {}
                yield CordReceipt(
                    receipt_id=f"{split}-{meta.get('image_id', seen)}",
                    split=split,
                    image_bytes=img.get("bytes") if with_images else None,
                    width=int(size.get("width", 0)),
                    height=int(size.get("height", 0)),
                    gt_parse=gt.get("gt_parse", {}),
                    source_file=Path(path).name,
                )
                seen += 1
                if limit is not None and seen >= limit:
                    return


def cord_to_bill(receipt: CordReceipt) -> Bill:
    """Build a Bill from the annotation, with no OCR involved.

    This is the reference path: it isolates the verification rules from
    extraction error, so a failed arithmetic check is a property of the receipt
    rather than of the reader.
    """
    lines: list[BillLine] = []
    for i, item in enumerate(receipt.menu_items):
        qty = _dec(item.get("cnt")) or Decimal(1)
        unit = _dec(item.get("unitprice"))
        line_total = _dec(item.get("price")) or _dec(item.get("itemsubtotal"))
        if unit is None and line_total is not None and qty and qty != 0:
            unit = (line_total / qty)
        lines.append(
            BillLine(
                index=i,
                description=str(item.get("nm", "")).strip(),
                quantity=qty,
                unit_price=Money.from_rupees(unit) if unit is not None else None,
                line_total=Money.from_rupees(line_total) if line_total is not None else None,
                discount=(
                    Money.from_rupees(abs(d))
                    if (d := _dec(item.get("discountprice"))) is not None
                    else None
                ),
                raw_text=json.dumps(item, ensure_ascii=False),
            )
        )

    sub = _as_list(receipt.gt_parse.get("sub_total"))
    sub = sub[0] if sub else {}
    tot = _as_list(receipt.gt_parse.get("total"))
    tot = tot[0] if tot else {}

    def money(node: dict, key: str) -> Money | None:
        v = _dec(node.get(key))
        return Money.from_rupees(v) if v is not None else None

    discount = money(sub, "discount_price")
    bill = Bill(
        bill_id=receipt.receipt_id,
        source_image=receipt.source_file,
        lines=lines,
        subtotal=money(sub, "subtotal_price"),
        discount_total=Money.from_paise(abs(discount.paise)) if discount else None,
        tax_total=money(sub, "tax_price"),
        grand_total=money(tot, "total_price"),
    )
    return bill


@dataclass
class CordStats:
    receipts: int = 0
    with_lines: int = 0
    with_total: int = 0
    with_subtotal: int = 0
    line_count: int = 0
    notes: dict = field(default_factory=dict)


def load_cord(
    data_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("test",),
    limit: int | None = None,
    with_images: bool = True,
) -> tuple[list[CordReceipt], list[Bill], CordStats]:
    """Load receipts and their reference Bills, with a summary of what is there."""
    receipts = list(iter_cord_rows(data_dir, splits=splits, limit=limit, with_images=with_images))
    bills = [cord_to_bill(r) for r in receipts]
    stats = CordStats(
        receipts=len(receipts),
        with_lines=sum(1 for b in bills if b.lines),
        with_total=sum(1 for b in bills if b.grand_total is not None),
        with_subtotal=sum(1 for b in bills if b.subtotal is not None),
        line_count=sum(len(b.lines) for b in bills),
        notes={
            "source": "CORD-v2 (naver-clova-ix/cord-v2), CC BY 4.0",
            "splits": list(splits),
        },
    )
    return receipts, bills, stats
