"""Fail the build if a rupee amount can reach a user without its working.

This lives in a file rather than inline in the CI workflow on purpose. The
first version was a shell heredoc inside a YAML block, which is three quoting
systems stacked on top of each other, and the job failed with bash's syntax
error code while the project itself was fine on Linux. A script has one
quoting system and can be run locally in exactly the form CI runs it:

    python scripts/check_guardrail.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stockmate.models import Finding  # noqa: E402
from stockmate.money import Money  # noqa: E402
from stockmate.store.ledger import (  # noqa: E402
    InMemoryLedger,
    LedgerMismatchError,
    UnbackedAmountError,
    verify_finding,
)


def expect_refused(finding: Finding, ledger: InMemoryLedger, label: str) -> None:
    try:
        verify_finding(finding, ledger)
    except (UnbackedAmountError, LedgerMismatchError) as exc:
        print(f"  refused as it should be: {label}")
        print(f"    {exc}")
        return
    raise SystemExit(
        f"GUARDRAIL FAILED: {label} was allowed through. An amount could reach "
        "a shopkeeper with nothing behind it."
    )


def main() -> int:
    print("checking the rupee guardrail")

    ledger = InMemoryLedger()

    expect_refused(
        Finding(finding_id="a", kind="k", title="t", detail="d",
                amount=Money.from_rupees("100.00")),
        ledger,
        "an amount with no computation recorded at all",
    )

    expect_refused(
        Finding(finding_id="b", kind="k", title="t", detail="d",
                amount=Money.from_rupees("100.00"), computation_id="c_missing"),
        ledger,
        "an amount pointing at a computation that is not in the ledger",
    )

    cid = ledger.record(
        kind="k", formula="f", inputs={}, output=Money.from_rupees("250.00")
    )
    expect_refused(
        Finding(finding_id="c", kind="k", title="t", detail="d",
                amount=Money.from_rupees("100.00"), computation_id=cid),
        ledger,
        "an amount that disagrees with the computation it points at",
    )

    backed = Finding(finding_id="d", kind="k", title="t", detail="d",
                     amount=Money.from_rupees("250.00"), computation_id=cid)
    comp = verify_finding(backed, ledger)
    if comp is None or comp.output != Money.from_rupees("250.00"):
        raise SystemExit("GUARDRAIL FAILED: a properly backed amount was not allowed")
    print("  allowed, correctly: an amount the ledger reproduces")

    print("guardrail active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
