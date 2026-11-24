"""Draw every figure in the report directly from results/.

No figure is drawn by hand and none carries a number that is not in a results
file. If a benchmark has not been run, the figure that depends on it is skipped
with a warning rather than drawn from stale or invented values.

Run:
    python scripts/make_figures.py
Writes figures/fig*.png at 300 dpi, sized for a two-column page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# A single restrained palette, used consistently so that the same thing is the
# same colour in every figure.
INK = "#1d2733"
MUTED = "#6b7785"
GRID = "#dfe4ea"
TEAL = "#1f7a8c"
AMBER = "#c98a1b"
CLAY = "#b2503f"
SLATE = "#49657d"
MOSS = "#5a7d4f"
SERIES = [TEAL, AMBER, CLAY, SLATE, MOSS, MUTED]

COL_W = 3.4          # one template column, inches
WIDE_W = 7.0         # across both columns

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.6,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name: str):
    path = RESULTS / name
    if not path.exists():
        print(f"  skip: {name} not found")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name: str):
    FIGS.mkdir(exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def tidy(ax, bottom=True, left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)


# --------------------------------------------------------------- architecture

def fig_architecture():
    fig, ax = plt.subplots(figsize=(WIDE_W, 2.45))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, sub, fc, tc="white"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
            linewidth=0, facecolor=fc))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=6.6, color=tc, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=5.4, color=tc, alpha=0.93, linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0,
                                    shrinkA=1, shrinkB=1))

    row1, row2 = 23.0, 4.0
    h = 12.0
    box(0.5, row1, 17.5, h, "Owner inbox", "photo, PDF\nvoice note", SLATE)
    box(21, row1, 17.5, h, "Extract", "OCR, parse\nQR decode", TEAL)
    box(41.5, row1, 17.5, h, "Verify", "sums, GST\nduplicates", TEAL)
    box(62, row1, 17.5, h, "Confidence gate", "confirm only\non doubt", SLATE)
    box(82, row1, 17.5, h, "Event store", "append only, tied\nto the source image", SLATE)

    box(82, row2, 17.5, h, "Engines", "audit, cash trap\nplanner", TEAL)
    box(62, row2, 17.5, h, "Rupee guardrail", "every figure must match\na stored computation", CLAY)
    box(41.5, row2, 17.5, h, "Language layer", "explains,\nnever computes", AMBER)
    box(0.5, row2, 38, h, "Owner output",
        "findings with their working, one-tap approval", MOSS)

    for a, b in [(18, 21), (38.5, 41.5), (59, 62), (79.5, 82)]:
        arrow(a, row1 + h / 2, b, row1 + h / 2)
    arrow(90.75, row1, 90.75, row2 + h)
    for a, b in [(82, 79.5), (62, 59), (41.5, 38.5)]:
        arrow(a, row2 + h / 2, b, row2 + h / 2)

    save(fig, "fig1_architecture.png")


# ------------------------------------------------------------------- forecast

def fig_forecast():
    d = load("forecast_backtest.json")
    if not d:
        return
    methods = d["methods"]
    order = sorted(methods, key=lambda k: methods[k]["wape"])
    order = [m for m in order if m != "zero"]
    wape = [methods[m]["wape"] for m in order]
    bias = [methods[m]["bias"] for m in order]
    labels = [m.replace("_", " ") for m in order]
    colours = [TEAL if m.startswith("stockmate") else MUTED for m in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDE_W, 2.3))
    y = np.arange(len(order))
    ax1.barh(y, wape, color=colours, height=0.62)
    ax1.set_yticks(y); ax1.set_yticklabels(labels)
    ax1.invert_yaxis(); ax1.set_xlabel("WAPE (lower is better)")
    ax1.set_title("Forecast error", loc="left")
    for i, v in enumerate(wape):
        ax1.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=6.6, color=INK)
    ax1.set_xlim(0, max(wape) * 1.18)
    tidy(ax1)

    ax2.barh(y, bias, color=colours, height=0.62)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.invert_yaxis(); ax2.set_xlabel("bias, units per day")
    ax2.set_title("Systematic over or under forecast", loc="left")
    tidy(ax2)

    fig.suptitle("")
    save(fig, "fig_forecast.png")


# ---------------------------------------------------------------------- audit

def fig_audit_ablation():
    d = load("audit_detection.json")
    if not d:
        return
    abl = d["ablation"]
    kinds = sorted(d["by_fault_kind"])
    names = list(abl)
    fig, ax = plt.subplots(figsize=(WIDE_W, 2.4))
    width = 0.38
    x = np.arange(len(kinds))
    for i, name in enumerate(names):
        vals = [abl[name]["by_fault_kind"].get(k, {}).get("recall") or 0.0 for k in kinds]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=name.replace("_", " "), color=SERIES[i], edgecolor="none")
        for xi, v in zip(x + (i - 0.5) * width, vals):
            if v > 0.02:
                ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=6.0, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=6.8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of injected faults reported")
    ax.set_title("What each tier of rules catches", loc="left", pad=16)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.8)
    tidy(ax)
    save(fig, "fig_audit_ablation.png")


# ---------------------------------------------------------------- price creep

def fig_price_creep():
    d = load("price_creep.json")
    if not d:
        return
    rows = [r for r in d["thresholds"] if r["impact_floor"] == 0.0]
    if not rows:
        rows = d["thresholds"]
    rises = sorted(
        {float(k.split("_")[1].replace("pct", "")) for k in rows[0] if k.startswith("detect_")}
    )
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    for i, r in enumerate(rows):
        ys = [r.get(f"detect_{int(x)}pct") or r.get(f"detect_{x}pct") or 0 for x in rises]
        ax.plot(rises, ys, marker="o", ms=3.2, lw=1.3, color=SERIES[i % len(SERIES)],
                label=f"{r['threshold_pct']:.0f}% rule, {r['alert_burden']:.0%} alerts")
    ax.set_xlabel("size of the price rise, per cent")
    ax.set_ylabel("share reported")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Detection against alert burden", loc="left")
    ax.legend(fontsize=6.2, loc="lower right")
    tidy(ax)
    save(fig, "fig_price_creep.png")


# -------------------------------------------------------------- replenishment

def fig_replenishment():
    d = load("replenishment.json")
    if not d:
        return
    pol = d["policies"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for i, p in enumerate(pol):
        is_ours = p["policy"].startswith("stockmate")
        ax.scatter(p["mean_inventory_value"], p["fill_rate"],
                   s=46 if is_ours else 32,
                   color=SERIES[i % len(SERIES)],
                   marker="o" if is_ours else "s",
                   zorder=3, edgecolor="white", linewidth=0.6)
        ax.annotate(p["policy"].replace("stockmate_", "").replace("_", " "),
                    (p["mean_inventory_value"], p["fill_rate"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.0, color=INK)
    ax.set_xlabel("average inventory value held")
    ax.set_ylabel("fill rate")
    ax.set_title("Service against capital, same cash budget", loc="left")
    tidy(ax)
    save(fig, "fig_replenishment.png")


def fig_cost_of_a_sale():
    """Units bought per rupee is what separates the allocation rules."""
    d = load("replenishment.json")
    if not d:
        return
    pol = d["policies"]
    names = [p["policy"].replace("stockmate_", "sm ").replace("_", " ") for p in pol]
    fill = [p["fill_rate"] for p in pol]
    held = [p["mean_inventory_value"] for p in pol]
    base = max(held) or 1
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    y = np.arange(len(pol))
    ax.barh(y - 0.19, fill, 0.36, color=TEAL, label="fill rate")
    ax.barh(y + 0.19, [h / base for h in held], 0.36, color=AMBER,
            label="inventory held, relative")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_title("Higher service is bought with capital", loc="left")
    ax.legend(fontsize=6.4, loc="lower right")
    tidy(ax)
    save(fig, "fig_service_capital.png")


def main() -> int:
    print("drawing figures from results/")
    fig_architecture()
    fig_forecast()
    fig_audit_ablation()
    fig_price_creep()
    fig_replenishment()
    fig_cost_of_a_sale()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
