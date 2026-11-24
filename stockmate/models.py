"""Domain records shared by the extract, verify, audit and planning stages.

These are plain dataclasses rather than ORM rows.  The database stores events;
these objects are what the engines reason about, and keeping them free of any
persistence machinery is what lets the benchmarks run the same code paths as the
live service without a database at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from .money import Money

__all__ = [
    "Confidence",
    "Field",
    "BillLine",
    "Bill",
    "QRPayload",
    "Severity",
    "CheckStatus",
    "Check",
    "Finding",
    "StockItem",
    "SalesPoint",
    "OrderSuggestion",
    "PlanDecision",
]


# --------------------------------------------------------------------- fields


class Confidence(float, Enum):
    """Named cut-offs for the confidence gate.

    Anything the extractor returns below REVIEW is not allowed to reach an
    engine without the owner confirming it first.
    """

    CERTAIN = 0.95
    TRUSTED = 0.80
    REVIEW = 0.60


@dataclass
class Field:
    """One extracted value together with how it was obtained."""

    value: Any
    confidence: float = 1.0
    source: str = "ocr"           # ocr | qr | owner | derived | catalogue
    raw_text: str | None = None
    bbox: tuple[int, int, int, int] | None = None

    @property
    def needs_review(self) -> bool:
        return self.confidence < Confidence.REVIEW

    @property
    def is_trusted(self) -> bool:
        return self.confidence >= Confidence.TRUSTED

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(self.value, Money):
            d["value"] = self.value.paise
            d["value_unit"] = "paise"
        elif isinstance(self.value, (date, datetime)):
            d["value"] = self.value.isoformat()
        elif isinstance(self.value, Decimal):
            d["value"] = str(self.value)
        return d


# ---------------------------------------------------------------------- bills


@dataclass
class BillLine:
    """A single charged line on a supplier bill."""

    index: int
    description: str
    quantity: Decimal | None = None
    unit_price: Money | None = None
    line_total: Money | None = None
    hsn: str | None = None
    gst_rate: Decimal | None = None
    discount: Money | None = None
    free_qty: Decimal | None = None
    sku_id: str | None = None          # filled by the SKU resolver
    confidence: float = 1.0
    raw_text: str = ""

    def expected_total(self) -> Money | None:
        """qty x unit price, before any discount, or None if either is missing."""
        if self.quantity is None or self.unit_price is None:
            return None
        return self.unit_price * self.quantity

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "unit_price_paise": self.unit_price.paise if self.unit_price else None,
            "line_total_paise": self.line_total.paise if self.line_total else None,
            "hsn": self.hsn,
            "gst_rate": str(self.gst_rate) if self.gst_rate is not None else None,
            "discount_paise": self.discount.paise if self.discount else None,
            "free_qty": str(self.free_qty) if self.free_qty is not None else None,
            "sku_id": self.sku_id,
            "confidence": self.confidence,
        }


@dataclass
class QRPayload:
    """Decoded contents of an e-invoice QR code.

    The signature on a real NIC e-invoice QR is a JWS.  This system decodes and
    cross-checks the payload but does not verify that signature, because doing so
    needs NIC's published key material that is not available to the prototype.
    Everything derived from here is therefore treated as a claim to be matched
    against the printed bill, never as proof on its own.
    """

    raw: str
    seller_gstin: str | None = None
    buyer_gstin: str | None = None
    doc_number: str | None = None
    doc_date: date | None = None
    total_value: Money | None = None
    irn: str | None = None
    signature_verified: bool = False

    def to_dict(self) -> dict:
        return {
            "seller_gstin": self.seller_gstin,
            "buyer_gstin": self.buyer_gstin,
            "doc_number": self.doc_number,
            "doc_date": self.doc_date.isoformat() if self.doc_date else None,
            "total_value_paise": self.total_value.paise if self.total_value else None,
            "irn": self.irn,
            "signature_verified": self.signature_verified,
        }


@dataclass
class Bill:
    """A supplier bill after extraction, before verification."""

    bill_id: str
    source_image: str | None = None
    image_sha256: str | None = None
    supplier_name: str | None = None
    supplier_gstin: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    lines: list[BillLine] = field(default_factory=list)
    subtotal: Money | None = None
    discount_total: Money | None = None
    tax_total: Money | None = None
    cgst: Money | None = None
    sgst: Money | None = None
    igst: Money | None = None
    round_off: Money | None = None
    grand_total: Money | None = None
    qr: QRPayload | None = None
    fields: dict[str, Field] = field(default_factory=dict)
    ocr_ms: float | None = None

    def content_hash(self) -> str:
        """Stable identity for duplicate detection, independent of the image."""
        parts = [
            (self.supplier_gstin or self.supplier_name or "").strip().upper(),
            (self.invoice_number or "").strip().upper(),
            self.invoice_date.isoformat() if self.invoice_date else "",
            str(self.grand_total.paise) if self.grand_total else "",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def low_confidence_fields(self) -> list[str]:
        return [name for name, f in self.fields.items() if f.needs_review]

    def to_dict(self) -> dict:
        return {
            "bill_id": self.bill_id,
            "image_sha256": self.image_sha256,
            "supplier_name": self.supplier_name,
            "supplier_gstin": self.supplier_gstin,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date.isoformat() if self.invoice_date else None,
            "lines": [l.to_dict() for l in self.lines],
            "subtotal_paise": self.subtotal.paise if self.subtotal else None,
            "discount_total_paise": self.discount_total.paise if self.discount_total else None,
            "tax_total_paise": self.tax_total.paise if self.tax_total else None,
            "cgst_paise": self.cgst.paise if self.cgst else None,
            "sgst_paise": self.sgst.paise if self.sgst else None,
            "igst_paise": self.igst.paise if self.igst else None,
            "round_off_paise": self.round_off.paise if self.round_off else None,
            "grand_total_paise": self.grand_total.paise if self.grand_total else None,
            "qr": self.qr.to_dict() if self.qr else None,
            "content_hash": self.content_hash(),
            "ocr_ms": self.ocr_ms,
        }


# ------------------------------------------------------------- checks/findings


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"       # the inputs this check needs were not on the bill


@dataclass
class Check:
    """The result of one deterministic verification rule."""

    rule_id: str
    status: CheckStatus
    message: str
    expected: Money | Decimal | str | None = None
    observed: Money | Decimal | str | None = None
    delta: Money | None = None
    line_index: int | None = None
    severity: Severity = Severity.MEDIUM
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAIL

    def to_dict(self) -> dict:
        def val(v):
            if isinstance(v, Money):
                return v.paise
            if isinstance(v, Decimal):
                return str(v)
            return v

        return {
            "rule_id": self.rule_id,
            "status": self.status.value,
            "message": self.message,
            "expected": val(self.expected),
            "observed": val(self.observed),
            "delta_paise": self.delta.paise if self.delta else None,
            "line_index": self.line_index,
            "severity": self.severity.value,
            "evidence": self.evidence,
        }


@dataclass
class Finding:
    """Something the owner is being told, with the money it is worth.

    ``computation_id`` points at the stored computation that produced
    ``amount``.  The rupee guardrail refuses to render any finding that carries
    an amount without one, so no number can reach the owner that the system
    cannot show its working for.
    """

    finding_id: str
    kind: str                     # overcharge | gst_error | scheme_missed | ...
    title: str
    detail: str
    amount: Money | None = None
    bucket: str | None = None     # found | released | protected
    severity: Severity = Severity.MEDIUM
    bill_id: str | None = None
    line_index: int | None = None
    sku_id: str | None = None
    computation_id: str | None = None
    checks: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "amount_paise": self.amount.paise if self.amount else None,
            "bucket": self.bucket,
            "severity": self.severity.value,
            "bill_id": self.bill_id,
            "line_index": self.line_index,
            "sku_id": self.sku_id,
            "computation_id": self.computation_id,
            "checks": list(self.checks),
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


# -------------------------------------------------------------- stock/planning


@dataclass
class StockItem:
    """Current position for one SKU in the shop."""

    sku_id: str
    name: str
    on_hand: Decimal
    unit_cost: Money
    sale_price: Money | None = None
    hsn: str | None = None
    expiry: date | None = None
    lead_time_days: int = 3
    pack_size: Decimal = Decimal(1)
    last_sold: date | None = None

    def value_locked(self) -> Money:
        return self.unit_cost * self.on_hand


@dataclass
class SalesPoint:
    sku_id: str
    day: date
    units: Decimal


@dataclass
class OrderSuggestion:
    sku_id: str
    name: str
    decision: "PlanDecision"
    suggest_qty: Decimal
    reorder_point: Decimal
    cover_days: float | None
    forecast_daily: float
    cash_required: Money
    reason: str
    computation_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "sku_id": self.sku_id,
            "name": self.name,
            "decision": self.decision.value,
            "suggest_qty": str(self.suggest_qty),
            "reorder_point": str(self.reorder_point),
            "cover_days": self.cover_days,
            "forecast_daily": self.forecast_daily,
            "cash_required_paise": self.cash_required.paise,
            "reason": self.reason,
            "computation_id": self.computation_id,
        }


class PlanDecision(str, Enum):
    BUY = "buy"
    BUY_LESS = "buy_less"
    DONT_BUY = "dont_buy"
