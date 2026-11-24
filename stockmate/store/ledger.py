"""The rupee guardrail.

Every rupee figure StockMate shows an owner has to be reproducible.  A bill
auditor that says "you were overcharged Rs 1,240" and cannot show the
subtraction behind it is worse than one that says nothing, because the owner
will take it to the supplier and be wrong.

The rule enforced here is narrow and mechanical: an amount reaches the owner
only if a :class:`Computation` with the same value was recorded first, and that
record keeps the formula and every input that produced it.  Code that tries to
attach an amount without one raises :class:`UnbackedAmountError` instead of
rendering.  The failure is loud by design -- a missing audit trail is a bug in
the engine, not something to paper over at display time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable, Protocol

from ..models import Finding
from ..money import Money

__all__ = [
    "Computation",
    "Ledger",
    "InMemoryLedger",
    "UnbackedAmountError",
    "LedgerMismatchError",
]


class UnbackedAmountError(RuntimeError):
    """A finding carried an amount with no recorded computation behind it."""


class LedgerMismatchError(RuntimeError):
    """A finding's amount disagreed with the computation it points at."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Money):
        return {"paise": value.paise, "rupees": str(value.rupees)}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class Computation:
    """One recorded calculation: what went in, the rule applied, what came out."""

    computation_id: str
    kind: str
    formula: str
    inputs: dict[str, Any]
    output_paise: int
    bill_id: str | None = None
    sku_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def output(self) -> Money:
        return Money(self.output_paise)

    def explain(self) -> str:
        """A one-line working, suitable for showing under a finding."""
        parts = []
        for key, value in self.inputs.items():
            if isinstance(value, Money):
                parts.append(f"{key}={value.format()}")
            else:
                parts.append(f"{key}={value}")
        return f"{self.formula}  [{', '.join(parts)}]  = {self.output.format()}"

    def to_dict(self) -> dict:
        return {
            "computation_id": self.computation_id,
            "kind": self.kind,
            "formula": self.formula,
            "inputs": _jsonable(self.inputs),
            "output_paise": self.output_paise,
            "bill_id": self.bill_id,
            "sku_id": self.sku_id,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


class Ledger(Protocol):
    """Append-only record of every money calculation the engines perform."""

    def record(
        self,
        kind: str,
        formula: str,
        inputs: dict[str, Any],
        output: Money,
        *,
        bill_id: str | None = None,
        sku_id: str | None = None,
    ) -> str: ...

    def get(self, computation_id: str) -> Computation | None: ...

    def all(self) -> Iterable[Computation]: ...


class InMemoryLedger:
    """The default ledger.

    Deterministic ids: the same calculation with the same inputs produces the
    same id, which is what lets a replayed bill produce a byte-identical audit
    trail and makes the reproducibility test meaningful.
    """

    def __init__(self) -> None:
        self._rows: dict[str, Computation] = {}

    def record(
        self,
        kind: str,
        formula: str,
        inputs: dict[str, Any],
        output: Money,
        *,
        bill_id: str | None = None,
        sku_id: str | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "kind": kind,
                "formula": formula,
                "inputs": _jsonable(inputs),
                "output": output.paise,
                "bill_id": bill_id,
                "sku_id": sku_id,
            },
            sort_keys=True,
            default=str,
        )
        cid = "c_" + hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
        if cid not in self._rows:
            self._rows[cid] = Computation(
                computation_id=cid,
                kind=kind,
                formula=formula,
                inputs=inputs,
                output_paise=output.paise,
                bill_id=bill_id,
                sku_id=sku_id,
            )
        return cid

    def get(self, computation_id: str) -> Computation | None:
        return self._rows.get(computation_id)

    def all(self) -> Iterable[Computation]:
        return list(self._rows.values())

    def __len__(self) -> int:
        return len(self._rows)


def verify_finding(finding: Finding, ledger: Ledger) -> Computation | None:
    """Check one finding against the ledger before it is shown.

    Returns the backing computation, or None when the finding legitimately
    carries no amount.  Raises when an amount is present but unsupported.
    """
    if finding.amount is None:
        return None
    if not finding.computation_id:
        raise UnbackedAmountError(
            f"finding {finding.finding_id} ({finding.kind}) states "
            f"{finding.amount.format()} with no computation recorded"
        )
    comp = ledger.get(finding.computation_id)
    if comp is None:
        raise UnbackedAmountError(
            f"finding {finding.finding_id} points at computation "
            f"{finding.computation_id}, which is not in the ledger"
        )
    if comp.output_paise != finding.amount.paise:
        raise LedgerMismatchError(
            f"finding {finding.finding_id} states {finding.amount.format()} but "
            f"computation {comp.computation_id} produced {comp.output.format()}"
        )
    return comp


def guard(findings: Iterable[Finding], ledger: Ledger) -> list[Finding]:
    """Gate a batch of findings.  Raises on the first unbacked amount."""
    out = []
    for f in findings:
        verify_finding(f, ledger)
        out.append(f)
    return out


def audit_trail(findings: Iterable[Finding], ledger: Ledger) -> list[dict]:
    """Findings paired with their working, for the dashboard and for evidence."""
    trail = []
    for f in findings:
        comp = verify_finding(f, ledger)
        trail.append(
            {
                "finding": f.to_dict(),
                "computation": comp.to_dict() if comp else None,
                "working": comp.explain() if comp else None,
            }
        )
    return trail
