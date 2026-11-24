"""The GST rate schedule, loaded from versioned reference data.

Why this is a CSV and not a table of constants
----------------------------------------------
Indian GST rates move by Council decision.  The 56th Council meeting changed the
slab structure with effect from 22 September 2025, and a 57th meeting was
announced in October 2026.  A schedule compiled into a Python module can only be
corrected by whoever can edit and redeploy the package, which is never the person
holding the wrong bill.  So the schedule lives in ``data/reference/gst_rates.csv``,
it carries its own effective dates and provenance per row, and the code's job is
to load it, refuse to load it when it is malformed, and record which version of
it judged any given bill.

What is validated here, and what is not
---------------------------------------
This module validates the *shape* of the schedule: that every rate is one of the
legal slabs for its own effective date, that no prefix has two rows claiming the
same day, and that dates are ordered.  It cannot validate that a rate is the one
the law actually prescribes, because that needs a primary document.  Rows that
were not traced to a primary document are marked ``unverified`` in the data, and
``source_note`` carries that caveat into every check the verifier emits.

Lookup is longest-prefix.  A bill printing an 8 digit HSN is matched against the
8 digit rows first, then 6, then 4, then 2, which is how the published schedule
is organised: the broad chapter heading applies unless a narrower entry overrides
it.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

__all__ = [
    "GstScheduleError",
    "RateEntry",
    "GstSchedule",
    "SLAB_ERAS",
    "PREFIX_LENGTHS",
    "DEFAULT_CSV",
    "default_schedule",
]

#: Where the shipped schedule lives, relative to the repository root.
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "reference" / "gst_rates.csv"

#: Prefix lengths tried on lookup, longest first.  The published schedule is
#: organised in 2, 4, 6 and 8 digit entries and nothing in between.
PREFIX_LENGTHS = (8, 6, 4, 2)

#: The legal slab sets, by the date each set took effect.
#:
#: These are structural facts about the tax, not item-level rates, and they are
#: what makes "18.5 percent GST" rejectable without knowing anything about the
#: item.  They are still secondary knowledge: the 0/5/18/40 set reflects reported
#: coverage of the 56th Council meeting and was not read off a notification, so a
#: later Council decision that adds or drops a slab has to be entered here by
#: hand alongside the CSV refresh.
SLAB_ERAS: tuple[tuple[date, frozenset[Decimal]], ...] = (
    (
        date(2017, 7, 1),
        frozenset(Decimal(x) for x in ("0", "5", "12", "18", "28")),
    ),
    (
        date(2025, 9, 22),
        frozenset(Decimal(x) for x in ("0", "5", "18", "40")),
    ),
)

_HEADER = (
    "hsn_prefix",
    "description",
    "rate_percent",
    "effective_from",
    "effective_to",
    "source",
    "confidence",
)

_CONFIDENCE_VALUES = ("verified", "unverified")


class GstScheduleError(ValueError):
    """The schedule file is malformed and must not be used to judge a bill.

    Loading raises rather than skipping the bad row.  A silently dropped row
    means a bill is checked against a schedule with a hole in it, and the hole
    shows up as a finding against a supplier who did nothing wrong.
    """


@dataclass(frozen=True)
class RateEntry:
    """One dated rate row, carrying its own provenance."""

    hsn_prefix: str
    description: str
    rate_percent: Decimal
    effective_from: date
    effective_to: date | None
    source: str
    confidence: str
    row_number: int

    @property
    def is_verified(self) -> bool:
        return self.confidence == "verified"

    def covers(self, on_date: date) -> bool:
        if on_date < self.effective_from:
            return False
        return self.effective_to is None or on_date <= self.effective_to

    def to_dict(self) -> dict:
        return {
            "hsn_prefix": self.hsn_prefix,
            "description": self.description,
            "rate_percent": str(self.rate_percent),
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "source": self.source,
            "confidence": self.confidence,
            "row_number": self.row_number,
        }


def _parse_date(text: str, row_number: int, column: str) -> date:
    try:
        return date.fromisoformat(text.strip())
    except ValueError as exc:
        raise GstScheduleError(
            f"row {row_number}: {column} is not an ISO date: {text!r}"
        ) from exc


def _slabs_on(on_date: date) -> frozenset[Decimal]:
    """The legal slab set in force on ``on_date``.

    Dates before the first era are judged against the first era rather than
    against nothing, because an invoice dated before GST started is a data error
    on the bill, not a reason to stop checking rates.
    """
    chosen = SLAB_ERAS[0][1]
    for start, slabs in SLAB_ERAS:
        if on_date >= start:
            chosen = slabs
    return chosen


def _normalise_hsn(hsn: str) -> str:
    """Keep only the digits of an HSN as printed.

    Bills print HSN as ``1006 30 10``, ``10063010`` or ``1006.30.10`` depending on
    the billing software, and OCR adds its own spacing.  All of those are the same
    code.
    """
    return "".join(ch for ch in str(hsn) if ch.isdigit())


class GstSchedule:
    """A loaded, validated GST rate schedule.

    The object is immutable once built.  ``schedule_version`` is derived from the
    file's own bytes, so two runs that report the same version really did judge
    against the same schedule, and a refreshed CSV produces a visibly different
    version in every stored check.
    """

    def __init__(
        self,
        entries: list[RateEntry],
        *,
        schedule_version: str,
        source_path: str,
        verified_rows: int,
        unverified_rows: int,
    ) -> None:
        self._entries = list(entries)
        self._by_prefix: dict[str, list[RateEntry]] = {}
        for entry in self._entries:
            self._by_prefix.setdefault(entry.hsn_prefix, []).append(entry)
        self.schedule_version = schedule_version
        self.source_path = source_path
        self.verified_rows = verified_rows
        self.unverified_rows = unverified_rows

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path = DEFAULT_CSV) -> "GstSchedule":
        """Read and validate the schedule CSV, or raise ``GstScheduleError``."""
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise GstScheduleError(f"cannot read schedule at {path}: {exc}") from exc

        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        if tuple(reader.fieldnames or ()) != _HEADER:
            raise GstScheduleError(
                f"{path}: header must be {','.join(_HEADER)}, found "
                f"{','.join(reader.fieldnames or ())}"
            )

        entries: list[RateEntry] = []
        for offset, row in enumerate(reader):
            # Row 1 is the header, so the first data row is row 2.  Error
            # messages quote the number a text editor shows.
            row_number = offset + 2
            entries.append(cls._parse_row(row, row_number))

        cls._check_overlaps(entries)

        digest = hashlib.sha256(raw).hexdigest()[:12]
        verified = sum(1 for e in entries if e.is_verified)
        return cls(
            entries,
            schedule_version=f"{path.name}@{digest} ({len(entries)} rows)",
            source_path=str(path),
            verified_rows=verified,
            unverified_rows=len(entries) - verified,
        )

    @staticmethod
    def _parse_row(row: dict[str, str], row_number: int) -> RateEntry:
        prefix = _normalise_hsn(row.get("hsn_prefix") or "")
        if not prefix:
            raise GstScheduleError(f"row {row_number}: hsn_prefix is empty or has no digits")
        if len(prefix) not in PREFIX_LENGTHS:
            raise GstScheduleError(
                f"row {row_number}: hsn_prefix {prefix!r} has {len(prefix)} digits; "
                f"the schedule is organised in {PREFIX_LENGTHS} digit prefixes"
            )

        raw_rate = (row.get("rate_percent") or "").strip()
        try:
            rate = Decimal(raw_rate)
        except (InvalidOperation, ValueError) as exc:
            raise GstScheduleError(
                f"row {row_number}: rate_percent is not a number: {raw_rate!r}"
            ) from exc

        eff_from = _parse_date(row.get("effective_from") or "", row_number, "effective_from")
        raw_to = (row.get("effective_to") or "").strip()
        eff_to = _parse_date(raw_to, row_number, "effective_to") if raw_to else None
        if eff_to is not None and eff_to < eff_from:
            raise GstScheduleError(
                f"row {row_number}: effective_to {eff_to.isoformat()} is before "
                f"effective_from {eff_from.isoformat()}"
            )

        legal = _slabs_on(eff_from)
        if rate not in legal:
            allowed = ", ".join(str(s) for s in sorted(legal))
            raise GstScheduleError(
                f"row {row_number}: rate {rate} percent is not a legal slab on "
                f"{eff_from.isoformat()}; legal slabs then were {allowed}"
            )

        confidence = (row.get("confidence") or "").strip().lower()
        if confidence not in _CONFIDENCE_VALUES:
            raise GstScheduleError(
                f"row {row_number}: confidence must be one of "
                f"{', '.join(_CONFIDENCE_VALUES)}, found {confidence!r}"
            )

        source = (row.get("source") or "").strip()
        if not source:
            raise GstScheduleError(
                f"row {row_number}: source is empty; a rate with no stated origin "
                "cannot be put in front of a supplier"
            )

        return RateEntry(
            hsn_prefix=prefix,
            description=(row.get("description") or "").strip(),
            rate_percent=rate,
            effective_from=eff_from,
            effective_to=eff_to,
            source=source,
            confidence=confidence,
            row_number=row_number,
        )

    @staticmethod
    def _check_overlaps(entries: list[RateEntry]) -> None:
        """No prefix may have two rows covering the same day.

        Two overlapping rows mean the lookup result depends on file order, and a
        bill would be judged against whichever row happened to be read first.
        """
        by_prefix: dict[str, list[RateEntry]] = {}
        for entry in entries:
            by_prefix.setdefault(entry.hsn_prefix, []).append(entry)
        for prefix, rows in by_prefix.items():
            ordered = sorted(rows, key=lambda e: (e.effective_from, e.row_number))
            for earlier, later in zip(ordered, ordered[1:]):
                end = earlier.effective_to
                if end is None or end >= later.effective_from:
                    raise GstScheduleError(
                        f"row {later.row_number}: prefix {prefix} overlaps row "
                        f"{earlier.row_number}; row {earlier.row_number} runs from "
                        f"{earlier.effective_from.isoformat()} to "
                        f"{'open ended' if end is None else end.isoformat()} and row "
                        f"{later.row_number} starts on "
                        f"{later.effective_from.isoformat()}. Close the earlier row "
                        "the day before the change."
                    )

    # ------------------------------------------------------------------ queries
    @property
    def entries(self) -> list[RateEntry]:
        return list(self._entries)

    @property
    def prefixes(self) -> list[str]:
        return sorted(self._by_prefix)

    @property
    def source_note(self) -> str:
        """One sentence a check can quote about how much to trust the schedule."""
        if self.unverified_rows == 0:
            return (
                f"Schedule {self.schedule_version}: all {self.verified_rows} rows "
                "traced to a primary document."
            )
        return (
            f"Schedule {self.schedule_version}: reference data, "
            f"{self.unverified_rows} of {len(self._entries)} rows are marked "
            "unverified because they were not traced to a primary CBIC document. "
            "Refresh data/reference/gst_rates.csv against the current CBIC "
            "schedule before relying on a rate finding."
        )

    def lookup(self, hsn: str | None, on_date: date) -> RateEntry | None:
        """The scheduled rate row for ``hsn`` on ``on_date``, longest prefix first.

        Returns None when nothing matches, which the caller must treat as "this
        file does not cover that code" and never as a rate of zero.
        """
        if not hsn:
            return None
        digits = _normalise_hsn(hsn)
        if not digits:
            return None
        for length in PREFIX_LENGTHS:
            if len(digits) < length:
                continue
            for entry in self._by_prefix.get(digits[:length], ()):
                if entry.covers(on_date):
                    return entry
        return None

    def slabs(self, on_date: date) -> set[Decimal]:
        """The legal slab set in force on ``on_date``."""
        return set(_slabs_on(on_date))

    def is_legal_rate(self, rate: Decimal | str | int, on_date: date) -> bool:
        """Whether ``rate`` is one of the slabs the tax allowed on ``on_date``."""
        if not isinstance(rate, Decimal):
            try:
                rate = Decimal(str(rate))
            except (InvalidOperation, ValueError):
                return False
        return rate in _slabs_on(on_date)

    def describe(self) -> dict:
        return {
            "schedule_version": self.schedule_version,
            "source_path": self.source_path,
            "rows": len(self._entries),
            "prefixes": len(self._by_prefix),
            "verified_rows": self.verified_rows,
            "unverified_rows": self.unverified_rows,
            "source_note": self.source_note,
        }


def default_schedule() -> GstSchedule:
    """Load the schedule shipped with the repository."""
    return GstSchedule.load(DEFAULT_CSV)
