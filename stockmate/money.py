"""Exact money arithmetic for StockMate AI.

A bill auditor that uses binary floating point for rupees will eventually
disagree with the supplier's own arithmetic by a paisa and raise a finding that
is the auditor's fault, not the supplier's.  Every monetary value in this system
is therefore an integer number of paise, and every rounding step is explicit.

Rounding follows the convention used on Indian tax invoices: half-up away from
zero at the paisa, which is what spreadsheet and billing software produces.
Python's ``round`` uses banker's rounding and is deliberately not used here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

__all__ = ["Money", "ZERO", "parse_money", "quantize_half_up"]

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def _group_indian(value: Decimal) -> str:
    """Group digits the way an Indian bill prints them: 12,34,567.89.

    Western grouping would read 1,234,567.89, which a shopkeeper has to stop
    and translate. The last three digits stay together and everything above
    them is grouped in pairs, so the number reads directly as lakhs and crores.
    """
    whole, _, frac = f"{value:.2f}".partition(".")
    if len(whole) <= 3:
        return f"{whole}.{frac}"
    head, tail = whole[:-3], whole[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return f"{','.join(parts)},{tail}.{frac}"


def quantize_half_up(value: Decimal, places: int = 0) -> Decimal:
    """Round ``value`` half-up away from zero to ``places`` decimal places."""
    exp = Decimal(1).scaleb(-places)
    return value.quantize(exp, rounding=ROUND_HALF_UP)


@dataclass(frozen=True, order=True)
class Money:
    """An exact rupee amount held as an integer count of paise."""

    paise: int

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_rupees(cls, rupees: Decimal | int | str | float) -> "Money":
        if isinstance(rupees, float):
            # A float literal has already lost precision; go through its shortest
            # repr so that 18.07 means 18.07 and not 18.069999999999999.
            rupees = Decimal(repr(rupees))
        elif not isinstance(rupees, Decimal):
            rupees = Decimal(str(rupees))
        return cls(int(quantize_half_up(rupees * 100)))

    @classmethod
    def from_paise(cls, paise: int) -> "Money":
        return cls(int(paise))

    @classmethod
    def zero(cls) -> "Money":
        return cls(0)

    # ------------------------------------------------------------- conversions
    @property
    def rupees(self) -> Decimal:
        return (Decimal(self.paise) / 100).quantize(Decimal("0.01"))

    def __float__(self) -> float:  # for charts and metrics only, never for audit
        return float(self.rupees)

    def __str__(self) -> str:
        return _group_indian(abs(self.rupees)) if self.paise >= 0 else (
            "-" + _group_indian(abs(self.rupees))
        )

    def __repr__(self) -> str:
        return f"Money({self.paise}p = Rs {self})"

    def format(self, symbol: str = "Rs ") -> str:
        sign = "-" if self.paise < 0 else ""
        return f"{sign}{symbol}{_group_indian(abs(self.rupees))}"

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.paise + other.paise)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.paise - other.paise)

    def __neg__(self) -> "Money":
        return Money(-self.paise)

    def __abs__(self) -> "Money":
        return Money(abs(self.paise))

    def __mul__(self, factor: Decimal | int | str) -> "Money":
        """Multiply by a dimensionless quantity, rounding half-up to the paisa."""
        if isinstance(factor, float):
            factor = Decimal(repr(factor))
        elif not isinstance(factor, Decimal):
            factor = Decimal(str(factor))
        return Money(int(quantize_half_up(Decimal(self.paise) * factor)))

    __rmul__ = __mul__

    def percent(self, rate_percent: Decimal | int | str) -> "Money":
        """The ``rate_percent`` share of this amount, rounded half-up."""
        if not isinstance(rate_percent, Decimal):
            rate_percent = Decimal(str(rate_percent))
        return Money(int(quantize_half_up(Decimal(self.paise) * rate_percent / 100)))

    def split_half(self) -> tuple["Money", "Money"]:
        """Split into two halves that still sum exactly to the original.

        CGST and SGST are each half of the total GST.  On an odd number of paise
        the convention is to give the extra paisa to the first component so the
        two still add back to the stated tax.
        """
        first = Money(-(-self.paise // 2)) if self.paise >= 0 else Money(self.paise // 2)
        return first, Money(self.paise - first.paise)

    def is_zero(self) -> bool:
        return self.paise == 0

    def _check(self, other: object) -> None:
        if not isinstance(other, Money):
            raise TypeError(f"cannot combine Money with {type(other).__name__}")


ZERO = Money(0)


def total(amounts: Iterable[Money]) -> Money:
    """Sum money exactly; an empty iterable sums to zero, not to None."""
    out = 0
    for a in amounts:
        out += a.paise
    return Money(out)


def parse_money(text: str) -> Money | None:
    """Read a rupee amount out of a line of OCR text.

    Receipt OCR produces things like ``Rs 1,234.50``, ``1.234,50``, ``18,000``
    and ``₹900/-``.  Returns None when the text holds no usable number, which the
    caller must treat as a missing field rather than as zero.
    """
    if not text:
        return None
    cleaned = text.replace("₹", " ").replace("Rs", " ").replace("rs", " ")
    cleaned = cleaned.replace("INR", " ").replace("/-", " ")
    match = _NUM_RE.search(cleaned.replace(" ", ""))
    if not match:
        return None
    raw = match.group(0)

    # Decide which separator is the decimal point.  The last separator that is
    # followed by exactly two digits is the decimal point; everything else is a
    # thousands separator.
    if "," in raw and "." in raw:
        decimal_sep = "," if raw.rfind(",") > raw.rfind(".") else "."
    elif raw.count(",") == 1 and len(raw.split(",")[-1]) == 2:
        decimal_sep = ","
    elif raw.count(".") == 1 and len(raw.split(".")[-1]) in (1, 2):
        decimal_sep = "."
    else:
        decimal_sep = ""

    if decimal_sep:
        whole, _, frac = raw.rpartition(decimal_sep)
        digits = re.sub(r"\D", "", whole) or "0"
        frac_digits = re.sub(r"\D", "", frac)[:2].ljust(2, "0")
        negative = raw.lstrip().startswith("-")
        value = Decimal(f"{digits}.{frac_digits}")
    else:
        digits = re.sub(r"[^\d]", "", raw) or "0"
        negative = raw.lstrip().startswith("-")
        value = Decimal(digits)

    return Money.from_rupees(-value if negative else value)
