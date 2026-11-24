"""Request bodies for the JSON API.

Money crosses this boundary as a decimal string of rupees, not a float.  JSON
numbers are IEEE doubles, and a bill auditor that accepts 1234.35 as a double
has already lost the argument about whether its arithmetic is exact.  Strings
are parsed into integer paise on the way in and never become floats at all.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field, field_validator

from ..models import Bill, BillLine
from ..money import Money

__all__ = ["BillLineIn", "BillIn", "PlanQuery", "bill_from_payload"]


def _money(value: str | int | float | None) -> Money | None:
    if value is None or value == "":
        return None
    if isinstance(value, float):
        raise ValueError(
            "money must be sent as a decimal string, not a JSON number, "
            "so that no amount is rounded in transit"
        )
    try:
        return Money.from_rupees(Decimal(str(value)))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a money amount: {value!r}") from exc


class BillLineIn(BaseModel):
    description: str
    quantity: str | int | None = None
    unit_price: str | int | None = None
    line_total: str | int | None = None
    hsn: str | None = None
    gst_rate: str | int | None = None
    discount: str | int | None = None
    sku_id: str | None = None
    confidence: float = 1.0

    @field_validator("unit_price", "line_total", "discount", mode="before")
    @classmethod
    def _no_floats(cls, v):
        if isinstance(v, float):
            raise ValueError("send money as a decimal string, not a JSON number")
        return v


class BillIn(BaseModel):
    bill_id: str | None = None
    supplier_name: str | None = None
    supplier_gstin: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    lines: list[BillLineIn] = Field(default_factory=list)
    subtotal: str | int | None = None
    discount_total: str | int | None = None
    tax_total: str | int | None = None
    cgst: str | int | None = None
    sgst: str | int | None = None
    igst: str | int | None = None
    round_off: str | int | None = None
    grand_total: str | int | None = None


class PlanQuery(BaseModel):
    budget: str = "15000"
    service_level: float = 0.95
    review_days: int = 7


def bill_from_payload(payload: BillIn) -> Bill:
    """Build the domain Bill, failing loudly on anything ambiguous."""
    lines = []
    for i, item in enumerate(payload.lines):
        qty = Decimal(str(item.quantity)) if item.quantity not in (None, "") else None
        lines.append(
            BillLine(
                index=i,
                description=item.description,
                quantity=qty,
                unit_price=_money(item.unit_price),
                line_total=_money(item.line_total),
                hsn=item.hsn,
                gst_rate=Decimal(str(item.gst_rate)) if item.gst_rate not in (None, "") else None,
                discount=_money(item.discount),
                sku_id=item.sku_id,
                confidence=item.confidence,
            )
        )

    bill_id = payload.bill_id or f"bill-{payload.invoice_number or 'adhoc'}"
    return Bill(
        bill_id=bill_id,
        supplier_name=payload.supplier_name,
        supplier_gstin=payload.supplier_gstin,
        invoice_number=payload.invoice_number,
        invoice_date=payload.invoice_date,
        lines=lines,
        subtotal=_money(payload.subtotal),
        discount_total=_money(payload.discount_total),
        tax_total=_money(payload.tax_total),
        cgst=_money(payload.cgst),
        sgst=_money(payload.sgst),
        igst=_money(payload.igst),
        round_off=_money(payload.round_off),
        grand_total=_money(payload.grand_total),
    )
