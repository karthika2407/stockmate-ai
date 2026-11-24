"""In-memory shop state, and the demo fixture the bundled client starts from.

A warning that matters more than anything else in this file.  The shop below is
a FIXTURE.  Its catalogue, its stock levels and its demand history are
constructed so that the web client has something to show, and so that a
demonstration can be given without a real shop's books.  Nothing in it is a
measurement and nothing in it supports any claim.

Every number quoted in the project report comes instead from the benchmarks in
``benchmarks/``, which run on two public datasets: Online Retail II for demand
and CORD-v2 for receipts.  If a figure appears on this dashboard and also in the
report, that is a coincidence of the fixture and should be treated as a bug.

The fixture is deterministic.  Demand is drawn from a generator seeded with
blake2b over the SKU code, so the same shop appears on every machine and a
demonstration can be rehearsed.  Python's built-in ``hash`` is deliberately not
used for this: it is salted per process, so a fixture seeded from it would
differ between runs and nobody could reproduce what they saw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from ..datasets.perturb import seeded_random
from ..engines.bill_auditor import PriceHistory
from ..forecast.baselines import croston_sba
from ..models import Bill, BillLine, Finding, StockItem
from ..money import Money
from ..store.ledger import InMemoryLedger

__all__ = ["ShopState", "demo_state", "DEMO_CATALOGUE"]

#: (sku, name, hsn, unit cost in rupees, base daily demand, pack size, shelf life days)
DEMO_CATALOGUE = [
    ("RICE5",  "Sona Masoori Rice 5kg",      "1006", "310.00", 6.0,  10, None),
    ("ATTA5",  "Whole Wheat Atta 5kg",       "1101", "245.00", 5.0,  10, None),
    ("SUGAR1", "Sugar 1kg",                  "1701",  "46.00", 9.0,  20, None),
    ("TEA250", "Tea Dust 250g",              "0902", "118.00", 4.0,  24, None),
    ("OIL1",   "Sunflower Oil 1L",           "1512", "139.00", 7.0,  12, None),
    ("DAL1",   "Toor Dal 1kg",               "0713", "152.00", 4.5,  10, None),
    ("SALT1",  "Iodised Salt 1kg",           "2501",  "22.00", 6.0,  24, None),
    ("MILK500", "UHT Milk 500ml",            "0401",  "31.00", 14.0, 12, 90),
    ("CURD400", "Curd 400g",                 "0403",  "38.00", 9.0,  12, 21),
    ("PANEER",  "Paneer 200g",               "0406",  "89.00", 3.0,   8, 14),
    ("BISC",   "Glucose Biscuits 10x",       "1905",  "92.00", 8.0,  24, 180),
    ("NAMKEEN", "Mixture Namkeen 400g",      "2106",  "85.00", 3.5,  12, 120),
    ("SOAP4",  "Bathing Soap 4x125g",        "3401", "168.00", 3.0,  12, None),
    ("SHAMP",  "Shampoo 180ml",            "330510", "128.00", 2.0,  12, None),
    ("PASTE",  "Toothpaste 200g",            "3306", "112.00", 2.5,  12, None),
    ("DETER",  "Detergent Powder 1kg",       "3402", "118.00", 3.0,  12, None),
    ("WATER",  "Packaged Water 1L x12",      "2201", "144.00", 5.0,  12, 365),
    ("COLA",   "Aerated Drink 750ml x12",    "2202", "468.00", 2.0,  12, 150),
    ("COFFEE", "Instant Coffee 50g",         "0901", "165.00", 1.2,  12, None),
    ("GHEE500", "Ghee 500ml",                "0405", "320.00", 1.5,   6, 270),
]

#: SKUs the fixture deliberately leaves overstocked, so the Cash-Trap Radar has
#: something to find during a demonstration. Chosen, not measured.
DEMO_OVERSTOCK = {"COLA": 42.0, "COFFEE": 55.0, "GHEE500": 30.0, "SHAMP": 48.0}

DEMO_SUPPLIER = "Anand Distributors"


@dataclass
class ShopState:
    """Everything the service knows, held in memory.

    In-memory rather than in a database on purpose: the demonstration must start
    from an identical, known position every time, and /api/reset must return it
    there in milliseconds. A deployment that needs durability swaps this class
    for one backed by the event store without touching the API layer.
    """

    items: list[StockItem] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
    price_history: PriceHistory = field(default_factory=PriceHistory)
    ledger: InMemoryLedger = field(default_factory=InMemoryLedger)
    bills: list[Bill] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    seen_hashes: dict[str, str] = field(default_factory=dict)
    today: date = field(default_factory=date.today)

    def remember(self, bill: Bill, findings: list[Finding]) -> None:
        self.bills.append(bill)
        self.findings.extend(findings)
        self.seen_hashes.setdefault(bill.content_hash(), bill.bill_id)
        self.price_history.add_bill(bill)

    def forecast_rates(self) -> dict[str, float]:
        """Expected daily demand per SKU, from the same forecaster the planner uses."""
        out = {}
        for sku, series in self.history.items():
            import numpy as np

            out[sku] = float(np.mean(croston_sba(np.asarray(series, dtype=float), 10)))
        return out

    def forecast_pairs(self) -> dict[str, tuple[float, float]]:
        """Rate and standard deviation, which is what the planner needs."""
        import numpy as np

        out = {}
        for sku, series in self.history.items():
            arr = np.asarray(series, dtype=float)
            rate = float(np.mean(croston_sba(arr, 10)))
            sd = float(np.std(arr[-56:])) if len(arr) >= 7 else 0.0
            out[sku] = (rate, sd)
        return out

    def item(self, sku: str) -> StockItem | None:
        return next((i for i in self.items if i.sku_id == sku), None)

    # ----------------------------------------------------------------- chat
    def answer(self, text: str) -> str:
        """Route a message to an engine and report what the engine computed.

        Keyword routing rather than a model: every reply below is assembled from
        figures the engines produced, so there is no path by which a number the
        owner reads was invented by a language layer.
        """
        q = text.lower().strip()

        if any(w in q for w in ("help", "what can you", "menu")):
            return (
                "I can do four things with the bills and stock you already have.\n"
                "1. Audit a supplier bill: say 'check my bill'.\n"
                "2. Show cash stuck in stock: say 'cash traps'.\n"
                "3. Suggest this week's order within a budget: say 'what should I buy'.\n"
                "4. Show what the bills have cost you so far: say 'what did I find'."
            )

        if any(w in q for w in ("buy", "order", "reorder", "replenish", "stock up")):
            return self._plan_reply(q)

        if any(w in q for w in ("trap", "stuck", "slow", "dead", "expiry", "expire")):
            return self._trap_reply()

        if any(w in q for w in ("found", "overcharge", "bill", "audit", "supplier")):
            return self._bill_reply()

        return (
            "I did not follow that. Try 'what should I buy this week', "
            "'show cash traps', or 'what did I find on bills'."
        )

    def _plan_reply(self, q: str) -> str:
        from ..engines.planner import build_plan

        budget = Money.from_rupees("15000")
        for token in q.replace(",", "").split():
            if token.isdigit() and len(token) >= 3:
                budget = Money.from_rupees(token)
                break

        result, _ = build_plan(
            self.items, self.forecast_pairs(), budget, ledger=self.ledger
        )
        buys = [s for s in result.suggestions if s.suggest_qty > 0]
        buys.sort(key=lambda s: -s.cash_required.paise)
        if not buys:
            return f"Nothing needs ordering this cycle. Your {budget.format()} stays in the till."

        lines = [
            f"Within {budget.format()} I would order {len(buys)} lines, "
            f"spending {result.spend.format()}."
        ]
        if result.cut_count:
            lines.append(f"{result.cut_count} lines were cut back to fit the budget.")
        lines.append("")
        for s in buys[:6]:
            lines.append(
                f"- {s.name}: {s.decision.value.replace('_', ' ')} {s.suggest_qty} "
                f"({s.cash_required.format()}), about {s.forecast_daily:.1f} a day"
            )
        if len(buys) > 6:
            lines.append(f"...and {len(buys) - 6} more on the dashboard.")
        return "\n".join(lines)

    def _trap_reply(self) -> str:
        from ..engines.cash_trap import scan_cash_traps

        traps, _ = scan_cash_traps(
            self.items, self.forecast_rates(), today=self.today, ledger=self.ledger
        )
        if not traps.findings:
            return "Nothing is sitting idle right now. Every line is turning over."
        lines = [
            f"{traps.total.format()} is tied up in stock that is not moving well:",
            f"- slow movers {traps.slow_value.format()}",
            f"- close to expiry {traps.expiry_value.format()}",
            f"- not selling at all {traps.dead_value.format()}",
            "",
            "The biggest ones:",
        ]
        for f in traps.findings[:4]:
            lines.append(f"- {f.title}: {f.amount.format() if f.amount else ''}")
        return "\n".join(lines)

    def _bill_reply(self) -> str:
        money = Money(sum(f.amount.paise for f in self.findings
                          if f.bucket == "found" and f.amount))
        if not self.findings:
            return "No bills audited yet. Send one and I will check it line by line."
        lines = [
            f"Across {len(self.bills)} bills I have flagged {money.format()} worth of problems.",
            "",
        ]
        for f in sorted(self.findings, key=lambda f: -(f.amount.paise if f.amount else 0))[:5]:
            lines.append(f"- {f.title}: {f.amount.format() if f.amount else 'no amount'}")
        lines.append("")
        lines.append("Open the dashboard to see the working behind each one.")
        return "\n".join(lines)


def _demand_series(sku: str, base: float, days: int = 180) -> list[float]:
    """A deterministic, intermittent daily series for the fixture shop.

    Weekly shape plus Poisson-like noise and genuine zero days, because a demo
    whose demand is smooth would make the forecaster look better than the
    benchmarks show it to be.
    """
    rng = seeded_random("demo-demand", sku)
    series = []
    for d in range(days):
        dow = d % 7
        weekly = 1.0 + (0.35 if dow in (5, 6) else -0.1 if dow == 1 else 0.0)
        lam = max(base * weekly, 0.05)
        # Sum of small uniforms approximates the count shape well enough for a
        # fixture, and keeps the generator dependency-free.
        draw = sum(rng.random() for _ in range(3)) / 3 * 2 * lam
        if rng.random() < 0.18:
            draw = 0.0
        series.append(float(round(draw)))
    return series


def demo_state(today: date | None = None) -> ShopState:
    """Build the fixture shop. See the module docstring: this is not data."""
    today = today or date.today()
    state = ShopState(today=today)

    for sku, name, hsn, cost, base, pack, shelf in DEMO_CATALOGUE:
        series = _demand_series(sku, base)
        state.history[sku] = series

        rate = max(sum(series[-28:]) / 28, 0.01)
        on_hand = DEMO_OVERSTOCK.get(sku, round(rate * 6, 0))

        last_sold = None
        for back, units in enumerate(reversed(series)):
            if units > 0:
                last_sold = today - timedelta(days=back)
                break
        if sku in DEMO_OVERSTOCK:
            last_sold = today - timedelta(days=rng_days(sku))

        state.items.append(StockItem(
            sku_id=sku,
            name=name,
            on_hand=Decimal(str(on_hand)),
            unit_cost=Money.from_rupees(cost),
            hsn=hsn,
            pack_size=Decimal(str(pack)),
            lead_time_days=3,
            expiry=(today + timedelta(days=min(shelf, 25))) if shelf else None,
            last_sold=last_sold,
        ))

    # Two past bills at the established prices, so the auditor has a reference,
    # and nothing is flagged until a bill that actually differs arrives.
    for n in (1, 2):
        bill = Bill(
            bill_id=f"demo-past-{n}",
            supplier_name=DEMO_SUPPLIER,
            invoice_number=f"AD/2026/{1000 + n}",
            invoice_date=today - timedelta(days=30 * (3 - n)),
            lines=[
                BillLine(
                    index=i,
                    description=name,
                    quantity=Decimal(str(pack)),
                    unit_price=Money.from_rupees(cost),
                    line_total=Money.from_rupees(cost) * Decimal(str(pack)),
                    hsn=hsn,
                )
                for i, (sku, name, hsn, cost, base, pack, shelf) in enumerate(DEMO_CATALOGUE)
            ],
        )
        state.price_history.add_bill(bill)
        state.seen_hashes.setdefault(bill.content_hash(), bill.bill_id)

    return state


def rng_days(sku: str) -> int:
    """How long an overstocked fixture item has sat unsold."""
    return 45 + seeded_random("demo-idle", sku).randrange(0, 40)
