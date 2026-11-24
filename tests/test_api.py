"""Service-level tests, driven through the HTTP layer the client actually uses."""

from fastapi.testclient import TestClient

from stockmate.api.app import app

client = TestClient(app)


def post_bill(**overrides) -> dict:
    payload = {
        "bill_id": "test-bill",
        "supplier_name": "Anand Distributors",
        "invoice_number": "AD/TEST/1",
        "invoice_date": "2026-11-24",
        "lines": [
            {"description": "Sona Masoori Rice 5kg", "quantity": "10",
             "unit_price": "310.00", "line_total": "3100.00", "hsn": "1006"},
            {"description": "Sugar 1kg", "quantity": "20",
             "unit_price": "46.00", "line_total": "920.00", "hsn": "1701"},
        ],
        "subtotal": "4020.00",
        "grand_total": "4020.00",
    }
    payload.update(overrides)
    return client.post("/api/bills", json=payload).json()


def test_health_declares_the_channel_is_simulated():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "simulated" in body["channel"].lower()


def test_a_clean_bill_produces_no_failures():
    body = post_bill()
    assert body["failures"] == 0
    assert body["money_found_paise"] == 0


def test_an_overcharged_line_is_found_with_its_working():
    body = post_bill(bill_id="creep", lines=[
        {"description": "Sona Masoori Rice 5kg", "quantity": "10",
         "unit_price": "348.00", "line_total": "3480.00", "hsn": "1006"},
    ], subtotal="3480.00", grand_total="3480.00")
    creep = [f for f in body["findings"] if f["kind"] == "price_creep"]
    assert creep, "a 12 per cent rise against two prior bills should be reported"
    assert creep[0]["amount_paise"] == 38000

    working = [a["working"] for a in body["audit_trail"] if a["working"]]
    assert any("348.00" in w and "310.00" in w for w in working)


def test_every_reported_amount_has_a_computation():
    body = post_bill(bill_id="guarded", grand_total="4500.00")
    for finding in body["findings"]:
        if finding["amount_paise"]:
            assert finding["computation_id"], finding


def test_money_as_a_json_number_is_rejected():
    r = client.post("/api/bills", json={
        "bill_id": "floaty", "supplier_name": "S",
        "lines": [{"description": "X", "quantity": "1", "unit_price": 310.55,
                   "line_total": "310.55"}],
    })
    assert r.status_code == 422


def test_dashboard_reports_the_three_buckets():
    body = client.get("/api/dashboard").json()
    for key in ("found", "released", "protected"):
        assert key in body["headline"]
    assert isinstance(body["cash_traps"], list)


def test_plan_never_exceeds_the_budget():
    body = client.get("/api/plan", params={"budget": 5000}).json()
    assert body["spend_paise"] <= 500000
    assert body["headroom_paise"] >= 0


def test_a_tighter_budget_cuts_more_lines():
    tight = client.get("/api/plan", params={"budget": 2000}).json()
    loose = client.get("/api/plan", params={"budget": 50000}).json()
    assert tight["spend_paise"] <= loose["spend_paise"]
    assert tight["lines_cut"] >= loose["lines_cut"]


def test_whatsapp_shaped_envelope_is_accepted():
    r = client.post("/api/webhook", json={
        "entry": [{"changes": [{"value": {"messages": [
            {"from": "919000000000", "type": "text",
             "text": {"body": "what should I buy this week"}}
        ]}}]}]
    })
    assert r.status_code == 200
    assert r.json()["handled"] is True


def test_a_media_message_says_plainly_that_it_cannot_be_fetched():
    r = client.post("/api/webhook", json={
        "entry": [{"changes": [{"value": {"messages": [
            {"from": "919000000000", "type": "image", "image": {"id": "123"}}
        ]}}]}]
    })
    body = r.json()
    assert body["handled"] is False
    assert "whatsapp business account" in body["reply"].lower()


def test_a_malformed_envelope_is_rejected():
    assert client.post("/api/webhook", json={"hello": "world"}).status_code == 422


def test_chat_routes_to_the_engines():
    for text, expect in [
        ("what should I buy this week", ("order", "ordering", "till")),
        ("show cash traps", ("tied up", "idle")),
        ("help", ("audit",)),
    ]:
        reply = client.post("/api/ask", json={"text": text}).json()["reply"].lower()
        assert any(e in reply for e in expect), (text, reply)


def test_reset_restores_the_fixture():
    assert client.post("/api/reset").json()["status"] == "reset"
    assert client.get("/api/health").json()["bills_seen"] == 0
