"""The rupee guardrail is a safety property, so it is tested as one.

The question each test asks is not "does it work" but "can a number reach the
owner that the system cannot show its working for".  The answer has to be no in
every case, including the ones where the mistake is the engine's own.
"""

import pytest

from stockmate.models import Finding
from stockmate.money import Money
from stockmate.store.ledger import (
    InMemoryLedger,
    LedgerMismatchError,
    UnbackedAmountError,
    audit_trail,
    guard,
    verify_finding,
)


def make_finding(**kw) -> Finding:
    base = dict(
        finding_id="f1", kind="price_creep", title="t", detail="d",
        amount=Money.from_rupees("380.00"),
    )
    base.update(kw)
    return Finding(**base)


def test_backed_finding_passes():
    ledger = InMemoryLedger()
    cid = ledger.record(
        kind="price_creep", formula="(now - before) x qty",
        inputs={"now": Money.from_rupees("348.00")},
        output=Money.from_rupees("380.00"),
    )
    comp = verify_finding(make_finding(computation_id=cid), ledger)
    assert comp is not None
    assert comp.output == Money.from_rupees("380.00")


def test_amount_without_any_computation_is_refused():
    with pytest.raises(UnbackedAmountError):
        verify_finding(make_finding(computation_id=None), InMemoryLedger())


def test_amount_pointing_at_a_missing_computation_is_refused():
    with pytest.raises(UnbackedAmountError):
        verify_finding(make_finding(computation_id="c_nope"), InMemoryLedger())


def test_amount_that_disagrees_with_its_computation_is_refused():
    ledger = InMemoryLedger()
    cid = ledger.record(
        kind="x", formula="f", inputs={}, output=Money.from_rupees("100.00")
    )
    with pytest.raises(LedgerMismatchError):
        verify_finding(make_finding(computation_id=cid), ledger)


def test_finding_with_no_amount_needs_no_computation():
    assert verify_finding(make_finding(amount=None, computation_id=None), InMemoryLedger()) is None


def test_guard_fails_the_batch_on_one_bad_finding():
    ledger = InMemoryLedger()
    cid = ledger.record(kind="x", formula="f", inputs={}, output=Money.from_rupees("380.00"))
    good = make_finding(computation_id=cid)
    bad = make_finding(finding_id="f2", computation_id=None)
    with pytest.raises(UnbackedAmountError):
        guard([good, bad], ledger)


def test_recording_is_deterministic():
    # The same calculation must produce the same id in two separate processes,
    # otherwise a replayed bill cannot be compared with its original.
    a, b = InMemoryLedger(), InMemoryLedger()
    args = dict(kind="k", formula="f",
                inputs={"q": "10", "p": Money.from_rupees("31.00")},
                output=Money.from_rupees("310.00"))
    assert a.record(**args) == b.record(**args)


def test_recording_twice_does_not_duplicate():
    ledger = InMemoryLedger()
    args = dict(kind="k", formula="f", inputs={}, output=Money.from_rupees("1.00"))
    ledger.record(**args)
    ledger.record(**args)
    assert len(ledger) == 1


def test_audit_trail_carries_the_working():
    ledger = InMemoryLedger()
    cid = ledger.record(
        kind="price_creep",
        formula="(unit price now - median before) x quantity",
        inputs={"now": Money.from_rupees("348.00"),
                "before": Money.from_rupees("310.00"), "quantity": "10"},
        output=Money.from_rupees("380.00"),
    )
    trail = audit_trail([make_finding(computation_id=cid)], ledger)
    assert len(trail) == 1
    working = trail[0]["working"]
    assert "348.00" in working and "310.00" in working and "380.00" in working


def test_empty_ledger_is_falsy_but_still_usable():
    # This is the bug the guardrail caught during development: `ledger or
    # InMemoryLedger()` silently swapped in a fresh ledger whenever the real one
    # happened to be empty, because __len__ returned 0.
    ledger = InMemoryLedger()
    assert not ledger
    assert len(ledger) == 0
    cid = ledger.record(kind="k", formula="f", inputs={}, output=Money.from_rupees("5.00"))
    assert ledger.get(cid) is not None
