"""The verification suite: every deterministic rule, in one place.

Grouping the rules into named tiers is not cosmetic.  It is what lets the
benchmark report what each tier is worth on its own, and it is what lets the
service run the cheap rules on every bill and the expensive ones only where
there is history to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Bill, Check, CheckStatus
from .arithmetic import run_arithmetic_checks
from .duplicate import check_duplicate_bill, check_duplicate_lines

__all__ = ["VerificationReport", "verify_bill", "TIERS"]

#: Which rule prefixes belong to which tier, for the ablation.
TIERS = {
    "arithmetic": ("ARITH.",),
    "duplicate": ("DUP.",),
    "gst": ("GST.",),
    "qr": ("QR.",),
}


@dataclass
class VerificationReport:
    bill_id: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def passed(self) -> bool:
        return not self.failures

    def by_tier(self, tier: str) -> list[Check]:
        prefixes = TIERS.get(tier, ())
        return [c for c in self.checks if c.rule_id.startswith(prefixes)]

    def failures_in(self, tiers: tuple[str, ...]) -> list[Check]:
        prefixes = tuple(p for t in tiers for p in TIERS.get(t, ()))
        return [c for c in self.failures if c.rule_id.startswith(prefixes)]

    def to_dict(self) -> dict:
        return {
            "bill_id": self.bill_id,
            "passed": self.passed,
            "failure_count": len(self.failures),
            "checks": [c.to_dict() for c in self.checks],
        }


def verify_bill(
    bill: Bill,
    *,
    seen_hashes: dict[str, str] | None = None,
    gst_checker=None,
    qr_checker=None,
) -> VerificationReport:
    """Run every applicable rule against one bill.

    ``gst_checker`` and ``qr_checker`` are injected rather than imported so that
    a deployment without a rate schedule, or a bill with no QR code, runs the
    rest of the suite instead of failing.
    """
    checks: list[Check] = []
    checks.extend(run_arithmetic_checks(bill))
    checks.extend(check_duplicate_lines(bill))
    if seen_hashes is not None:
        checks.append(check_duplicate_bill(bill, seen_hashes))
    if gst_checker is not None:
        checks.extend(gst_checker(bill))
    if qr_checker is not None and bill.qr is not None:
        checks.extend(qr_checker(bill))
    return VerificationReport(bill_id=bill.bill_id, checks=checks)
