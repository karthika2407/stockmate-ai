"""The StockMate service.

Two front doors onto the same pipeline.  One is a webhook shaped like the
WhatsApp Cloud API, so that moving from the bundled demo channel to a real
WhatsApp number is a deployment change and not a rewrite.  The other is a small
JSON API that the bundled web client uses.

On the WhatsApp question, plainly: this project has no WhatsApp Business
account, so no message has ever travelled over Meta's infrastructure here.  What
is implemented is the message envelope their Cloud API posts, and a local client
that posts the same shape.  Nothing in this code or in the report should be read
as a working WhatsApp integration, and the demo channel is labelled as a
simulation everywhere it appears.
"""

from __future__ import annotations

import time
from datetime import date
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..engines.bill_auditor import PriceHistory, audit_bill
from ..engines.cash_trap import scan_cash_traps
from ..engines.planner import build_plan
from ..explain.llm import NullExplainer, describe
from ..models import Bill, Finding
from ..money import Money
from ..store.ledger import InMemoryLedger, UnbackedAmountError, audit_trail, guard
from ..verify.suite import verify_bill
from .state import ShopState, demo_state
from .schemas import BillIn, PlanQuery, bill_from_payload

UI_DIR = Path(__file__).resolve().parents[2] / "ui"

app = FastAPI(
    title="StockMate AI",
    version=__version__,
    description=(
        "A profit guardian for small retail shops. Extract, verify, audit and "
        "plan, on the bills a shop already receives."
    ),
)

STATE: ShopState = demo_state()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "skus": len(STATE.items),
        "bills_seen": len(STATE.bills),
        "ledger_entries": len(STATE.ledger),
        "channel": "simulated; no WhatsApp Business account is connected",
    }


def _process(bill: Bill) -> dict:
    """Run one bill through the whole pipeline and return everything about it."""
    started = time.perf_counter()

    report = verify_bill(bill, seen_hashes=STATE.seen_hashes)
    findings, _ = audit_bill(bill, STATE.price_history, ledger=STATE.ledger)

    # Failed deterministic checks become findings too, so that the owner sees
    # one ranked list rather than two parallel ones.
    for check in report.failures:
        amount = abs(check.delta) if check.delta else None
        cid = None
        if amount is not None:
            cid = STATE.ledger.record(
                kind=check.rule_id,
                formula=f"difference found by rule {check.rule_id}",
                inputs={"expected": check.expected, "observed": check.observed,
                        "rule": check.rule_id},
                output=amount,
                bill_id=bill.bill_id,
            )
        findings.append(Finding(
            finding_id=f"{bill.bill_id}:{check.rule_id}:{check.line_index}",
            kind=check.rule_id.lower().replace(".", "_"),
            title=check.rule_id.replace(".", " ").title(),
            detail=check.message,
            amount=amount,
            bucket="found" if amount else None,
            severity=check.severity,
            bill_id=bill.bill_id,
            line_index=check.line_index,
            computation_id=cid,
            checks=[check.rule_id],
        ))

    # Nothing reaches the owner before the rupee guardrail has seen it.
    try:
        findings = guard(findings, STATE.ledger)
    except UnbackedAmountError as exc:
        raise HTTPException(status_code=500, detail=f"rupee guardrail: {exc}") from exc

    STATE.remember(bill, findings)

    found = Money(sum(f.amount.paise for f in findings
                      if f.bucket == "found" and f.amount))
    return {
        "bill": bill.to_dict(),
        "checks": [c.to_dict() for c in report.checks],
        "failures": len(report.failures),
        "findings": describe(findings, NullExplainer()),
        "audit_trail": audit_trail(findings, STATE.ledger),
        "money_found": found.format(),
        "money_found_paise": found.paise,
        "needs_confirmation": bill.low_confidence_fields(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/api/bills")
def post_bill(payload: BillIn) -> dict:
    """Accept a already-extracted bill as JSON and audit it."""
    try:
        bill = bill_from_payload(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _process(bill)


@app.get("/api/dashboard")
def dashboard() -> dict:
    forecasts = STATE.forecast_rates()
    traps, _ = scan_cash_traps(
        STATE.items, forecasts, today=STATE.today, ledger=STATE.ledger
    )
    try:
        trap_findings = guard(traps.findings, STATE.ledger)
    except UnbackedAmountError as exc:
        raise HTTPException(status_code=500, detail=f"rupee guardrail: {exc}") from exc

    found = Money(sum(f.amount.paise for f in STATE.findings
                      if f.bucket == "found" and f.amount))
    released = traps.slow_value + traps.dead_value
    protected = traps.expiry_value

    # The dashboard carries the working for everything it shows.  Being able to
    # expand any figure and see the arithmetic behind it is the whole promise,
    # so the trail travels with the findings rather than sitting behind a second
    # request the client might not make.
    shown = STATE.findings[-25:]
    trail = audit_trail(shown, STATE.ledger) + audit_trail(trap_findings[:25], STATE.ledger)

    return {
        "audit_trail": trail,
        "headline": {
            "found": found.format(),
            "released": released.format(),
            "protected": protected.format(),
            "found_paise": found.paise,
            "released_paise": released.paise,
            "protected_paise": protected.paise,
        },
        "bill_findings": describe(STATE.findings[-25:], NullExplainer()),
        "cash_traps": describe(trap_findings[:25], NullExplainer()),
        "trap_totals": traps.to_dict() | {"findings": None},
        "bills_seen": len(STATE.bills),
        "skus": len(STATE.items),
        "as_of": STATE.today.isoformat(),
    }


@app.get("/api/plan")
def plan(budget: float = 15000.0, service_level: float = 0.95, review_days: int = 7) -> dict:
    forecasts = STATE.forecast_pairs()
    result, _ = build_plan(
        STATE.items,
        forecasts,
        Money.from_rupees(str(budget)),
        review_days=review_days,
        service_level=service_level,
        ledger=STATE.ledger,
    )
    data = result.to_dict()
    data["explanation"] = (
        f"Wanted orders came to more than {Money.from_rupees(str(budget)).format()}, "
        f"so {result.cut_count} lines were reduced."
        if result.cut_count else
        "Everything needed this cycle fits inside the budget."
    )
    return data


@app.post("/api/webhook")
async def webhook(request: Request) -> JSONResponse:
    """Accept a WhatsApp Cloud API shaped message envelope.

    Only the text path is handled here.  A media message carries an id that must
    be exchanged for the file through Meta's media endpoint, which needs an
    account this project does not have, so that branch returns a clear
    explanation rather than pretending to download anything.
    """
    body = await request.json()
    try:
        value = body["entry"][0]["changes"][0]["value"]
        message = value["messages"][0]
        sender = message.get("from", "unknown")
        kind = message.get("type", "text")
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=422, detail="not a WhatsApp message envelope")

    if kind != "text":
        return JSONResponse({
            "reply": (
                "Images arrive as a media id that has to be fetched from the "
                "WhatsApp media endpoint. This deployment has no WhatsApp "
                "Business account, so please upload the bill in the web client."
            ),
            "handled": False,
            "from": sender,
        })

    text = message.get("text", {}).get("body", "").strip()
    return JSONResponse({"reply": STATE.answer(text), "handled": True, "from": sender})


@app.post("/api/ask")
def ask(payload: dict) -> dict:
    """The demo chat client's endpoint, same answering logic as the webhook."""
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="empty message")
    return {"reply": STATE.answer(text)}


#: A supplier bill for the demonstration, in the same shape the extractor
#: produces. It is a fixture, not a measurement: the rice line is priced above
#: what the fixture's earlier bills charged so that the auditor has something
#: real to find, and the printed total is a few rupees above the lines so that
#: the arithmetic check has something to say as well.
DEMO_BILL = {
    "bill_id": "demo-today",
    "supplier_name": "Anand Distributors",
    "invoice_number": "AD/2026/1184",
    "lines": [
        {"description": "Sona Masoori Rice 5kg", "quantity": "10",
         "unit_price": "348.00", "line_total": "3480.00", "hsn": "1006"},
        {"description": "Sunflower Oil 1L", "quantity": "12",
         "unit_price": "139.00", "line_total": "1668.00", "hsn": "1512"},
        {"description": "Sugar 1kg", "quantity": "20",
         "unit_price": "46.00", "line_total": "920.00", "hsn": "1701"},
        {"description": "Tea Dust 250g", "quantity": "24",
         "unit_price": "118.00", "line_total": "2832.00", "hsn": "0902"},
        {"description": "Sugar 1kg", "quantity": "20",
         "unit_price": "46.00", "line_total": "920.00", "hsn": "1701"},
    ],
    "subtotal": "9820.00",
    "grand_total": "9852.00",
}


@app.post("/api/demo/bill")
def demo_bill() -> dict:
    """Audit a prepared supplier bill, for demonstrations.

    This exists so the interface can be shown without a camera pointed at a
    paper bill. The payload is a fixture and is labelled as one in the response,
    so nothing downstream can mistake it for a measurement.
    """
    from datetime import date as _date

    payload = dict(DEMO_BILL)
    payload["invoice_date"] = _date.today().isoformat()
    payload["bill_id"] = f"demo-{_date.today().isoformat()}-{len(STATE.bills)}"
    result = _process(bill_from_payload(BillIn(**payload)))
    result["fixture"] = True
    result["note"] = "Demonstration bill, not measured data."
    return result


@app.post("/api/reset")
def reset() -> dict:
    global STATE
    STATE = demo_state()
    return {"status": "reset", "skus": len(STATE.items)}


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(UI_DIR / "index.html"))
