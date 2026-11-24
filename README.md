# StockMate AI

A profit guardian for small retail shops. It reads the supplier bills a shop
already receives, checks them line by line, says how much cash is stuck in stock
that is not moving, and proposes an order that fits the money in the till.

Built for the VConnecTech Engineering Marathon 2026 by team **Hack N Crack**
(Team ID M2026-028), Department of Computer Science and Engineering,
KIT-Kalaignarkarunanidhi Institute of Technology, Coimbatore.

---

## The idea in one line

Existing shop apps record what the owner types. StockMate audits what the
supplier sends.

A kirana works on roughly eight to ten per cent gross margin. At that margin a
quietly raised unit price, a scheme that was agreed and then not applied, or a
crate charged twice is the difference between a profitable month and a flat one,
and none of it is visible without comparing today's bill against what the same
supplier charged last time.

## Two rules the whole system is built around

**Code decides, the language layer only explains.** Every rupee figure is
produced by deterministic arithmetic. The explanation layer is handed a finding
whose amount has already been computed and may only rephrase it. It can never
originate a number. A model that is allowed to compute the refund a shopkeeper
demands from their supplier will eventually compute it wrong, confidently, and
the shopkeeper will be the one standing at the counter being told they are
mistaken.

**No rupee figure reaches the owner without its working.** Every amount is
recorded in an append-only ledger together with the formula and the inputs that
produced it. A finding that carries an amount with no backing computation raises
an error instead of rendering. This is enforced in code, not by convention, and
it caught a real defect in our own engines during development.

## What it does

| Stage | What happens |
|---|---|
| Extract | OCR reads the bill photo; the e-invoice QR payload is decoded where one is present |
| Verify | Line arithmetic, subtotal, grand total, CGST/SGST split, GST rate against a versioned schedule, duplicate lines, duplicate bills |
| Audit | Unit prices compared against the median of what this supplier charged before |
| Plan | Demand forecast, reorder point, then Buy / Buy Less / Don't Buy fitted to a cash budget |

Outcomes are reported in three buckets: money **found** (overcharges on bills),
money **released** (cash freed from stock that is not selling), and money
**protected** (loss avoided before stock expires).

## Quick start

```bash
git clone <this repository>
cd stockmate-ai
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
uvicorn stockmate.api.app:app --reload
```

Open <http://127.0.0.1:8000>. The service starts with a demo shop loaded, so
the dashboard and the chat client have something to show immediately.

Everything runs locally. There are no API keys, no paid services and no network
calls at runtime.

To reproduce the measurements:

```bash
python scripts/fetch_data.py --retail                       # 43 MB
python scripts/fetch_data.py --receipts --splits test validation   # about 330 MB
python scripts/run_benchmarks.py --quick
pytest -q
```

`python scripts/run_benchmarks.py` runs the full set and writes
`results/MANIFEST.json` recording each command, its exit status and its runtime.

## What is measured, and on what

Two public datasets, both CC BY 4.0, neither committed to this repository:

- **Online Retail II** (UCI 502, DOI 10.24432/C5CG6D). 1,067,371 transaction
  lines from a UK retailer, December 2009 to December 2011. Used for demand
  forecasting, the replenishment simulation and real price history.
- **CORD-v2** (naver-clova-ix/cord-v2). One thousand photographed receipts with
  line-item annotation. Used for extraction accuracy and bill-audit detection.

Results land in `results/` as JSON and CSV. Every figure in the project report
is traceable to a file there.

## What this is not

Being clear about the limits is part of the engineering, so these are stated
here rather than buried.

- **There is no WhatsApp integration.** The service implements the message
  envelope the WhatsApp Cloud API posts, and the bundled chat client posts the
  same shape, but no message has ever travelled over Meta's infrastructure here
  and no WhatsApp Business account exists. The demo channel is labelled a
  simulation everywhere it appears.
- **E-invoice QR signatures are not verified.** The QR payload is decoded and
  cross-checked against the printed total. Verifying the signature needs NIC key
  material we do not have, so the payload is treated as a claim to be matched,
  never as proof.
- **The GST rate schedule is unverified reference data.** Rates changed at the
  56th GST Council meeting in September 2025 and again at the 57th in October
  2026. The official sources blocked automated retrieval, so every row in
  `data/reference/gst_rates.csv` is marked `unverified` and carries a secondary
  source. What is validated here is the rule engine, not our knowledge of any
  particular rate. An operator must refresh the file against the current CBIC
  schedule before relying on it. See `data/reference/README.md`.
- **The evaluation data is not Indian retail.** CORD is Indonesian receipts and
  Online Retail II is a UK giftware business. They are used for document
  structure and demand dynamics. No claim about Indian prices or margins is
  drawn from either.
- **The demo shop is a fixture, not data.** The catalogue and stock levels in
  `stockmate/api/state.py` exist so the client has something to display. Nothing
  in it is a measurement and nothing in it supports any claim.
- **No field trial has been run.** Nothing here has been tested in a working
  shop with a real owner.

## Layout

```
stockmate/
  money.py            exact money as integer paise
  models.py           bill, check, finding, stock records
  store/ledger.py     the rupee guardrail
  datasets/           dataset loaders and the fault injector used in evaluation
  extract/            image prep, OCR, parsing, QR decoding
  verify/             arithmetic, duplicate and GST rules
  engines/            bill auditor, cash-trap radar, planner, simulation
  forecast/           baselines, global model, rolling-origin backtest
  explain/            the language layer and the boundary it cannot cross
  api/                FastAPI service and the demo fixture
benchmarks/           one script per measurement, each writing to results/
tests/                131 tests
ui/                   the web client, vanilla HTML, CSS and JavaScript
```

## Licence

MIT. See `LICENSE`.

The two datasets are not ours and are not redistributed here; each is licensed
CC BY 4.0 by its own authors and is fetched from its original source by
`scripts/fetch_data.py`.
