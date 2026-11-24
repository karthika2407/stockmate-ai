"""Turn the Online Retail II transaction log into daily demand per SKU.

Source
------
Online Retail II, UCI Machine Learning Repository (dataset 502),
DOI 10.24432/C5CG6D, CC BY 4.0.  1,067,371 transaction lines from a UK
non-store retailer between 1 December 2009 and 9 December 2011.

Why this dataset
----------------
The planner needs a demand signal with real intermittency, real returns and real
data-quality faults, and this is the largest openly licensed transaction-level
retail log that has all three.  It is a UK giftware business, not an Indian
kirana, so it is used here for demand dynamics only.  No claim about Indian
retail prices or margins is drawn from it, and the rupee figures the planner
reports in benchmarks are computed at the observed unit price rather than from
an assumed margin.

Cleaning decisions, all of which are reversible by the caller:
  * Invoices whose number starts with ``C`` are cancellations.  They carry
    negative quantities and are netted against demand rather than dropped, so a
    sale that was reversed does not inflate the forecast.
  * Lines with a non-positive price are administrative entries (postage,
    samples, bad debt adjustments) and are excluded from demand.
  * Stock codes that are not product codes (``POST``, ``BANK CHARGES``,
    ``TEST001`` and similar) are excluded.
  * Days with no transaction for a SKU are real zeros once that SKU has started
    selling, and are filled as zeros between its first and last observed sale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["RetailData", "load_online_retail", "daily_demand", "NON_PRODUCT_CODES"]

#: Stock codes in this dataset that are charges or adjustments, not goods.
NON_PRODUCT_CODES = {
    "POST", "D", "DOT", "M", "S", "AMAZONFEE", "m", "DCGSSBOY", "DCGSSGIRL",
    "PADS", "B", "CRUK", "C2", "BANK CHARGES", "ADJUST", "ADJUST2", "TEST001",
    "TEST002", "gift_0001_10", "gift_0001_20", "gift_0001_30", "gift_0001_40",
    "gift_0001_50", "SP1002",
}

_PRODUCT_CODE_RE = re.compile(r"^\d{5}[A-Za-z]{0,3}$")


@dataclass
class RetailData:
    """Cleaned transactions plus the audit of what cleaning removed."""

    transactions: pd.DataFrame
    report: dict

    def __len__(self) -> int:
        return len(self.transactions)


def _is_product(code: object) -> bool:
    s = str(code).strip()
    if s in NON_PRODUCT_CODES:
        return False
    return bool(_PRODUCT_CODE_RE.match(s))


def load_online_retail(
    path: str | Path,
    *,
    country: str | None = "United Kingdom",
    cache: bool = True,
) -> RetailData:
    """Read the raw workbook (or its parquet cache) and clean it.

    ``country`` restricts to one market so that a single demand calendar
    applies; pass None to keep all 43.
    """
    path = Path(path)
    cache_path = path.with_suffix(".clean.parquet")

    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            book = pd.ExcelFile(path)
            df = pd.concat([book.parse(s) for s in book.sheet_names], ignore_index=True)
        else:
            df = pd.read_parquet(path)
        # The workbook mixes types inside single columns: invoice numbers are
        # integers except for the "C"-prefixed cancellations, and a handful of
        # Description cells hold numbers. Normalise before anything touches them.
        for col in ("Invoice", "StockCode", "Description", "Country"):
            if col in df.columns:
                df[col] = df[col].astype("string").astype(object)
        if cache:
            df.to_parquet(cache_path, index=False)

    raw_rows = len(df)
    report: dict = {
        "source": "UCI Online Retail II (dataset 502), DOI 10.24432/C5CG6D",
        "raw_rows": raw_rows,
    }

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["is_cancellation"] = df["Invoice"].str.upper().str.startswith("C")
    report["cancellation_rows"] = int(df["is_cancellation"].sum())

    non_product = ~df["StockCode"].map(_is_product)
    report["non_product_rows"] = int(non_product.sum())
    df = df.loc[~non_product].copy()

    bad_price = df["Price"] <= 0
    report["non_positive_price_rows"] = int(bad_price.sum())
    df = df.loc[~bad_price].copy()

    if country is not None:
        in_market = df["Country"] == country
        report["rows_outside_market"] = int((~in_market).sum())
        report["market"] = country
        df = df.loc[in_market].copy()

    missing_desc = df["Description"].isna()
    report["missing_description_rows"] = int(missing_desc.sum())
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()

    df = df.rename(
        columns={
            "StockCode": "sku_id",
            "Description": "name",
            "Quantity": "units",
            "Price": "unit_price",
            "InvoiceDate": "ts",
        }
    )
    df["day"] = df["ts"].dt.normalize()
    df = df[["Invoice", "sku_id", "name", "units", "unit_price", "ts", "day", "is_cancellation"]]

    report["clean_rows"] = len(df)
    report["skus"] = int(df["sku_id"].nunique())
    report["invoices"] = int(df["Invoice"].nunique())
    report["first_day"] = str(df["day"].min().date())
    report["last_day"] = str(df["day"].max().date())
    report["gross_units"] = int(df.loc[~df["is_cancellation"], "units"].sum())
    report["returned_units"] = int(-df.loc[df["is_cancellation"], "units"].sum())

    return RetailData(transactions=df.reset_index(drop=True), report=report)


def daily_demand(
    data: RetailData,
    *,
    min_active_days: int = 180,
    min_total_units: int = 200,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Net daily units per SKU on a dense calendar.

    Returns a frame indexed by ``(sku_id, day)`` with columns ``units`` (net of
    returns, floored at zero for the day) and ``unit_price`` (the median price
    actually charged for that SKU, used as the cost proxy).

    SKUs are kept only when they have a long enough selling life to backtest
    against, because a product with eleven days of history cannot tell you
    anything about a forecaster.
    """
    tx = data.transactions
    net = (
        tx.groupby(["sku_id", "day"], as_index=False)
        .agg(units=("units", "sum"), name=("name", "first"))
    )
    # A day whose returns exceed its sales is a correction to an earlier day,
    # not negative demand; clip at zero and keep the raw value for the audit.
    net["units_raw"] = net["units"]
    net["units"] = net["units"].clip(lower=0)

    span = net.groupby("sku_id")["day"].agg(["min", "max", "count"])
    span["active_days"] = (span["max"] - span["min"]).dt.days + 1
    totals = net.groupby("sku_id")["units"].sum()

    keep = span.index[
        (span["active_days"] >= min_active_days) & (totals >= min_total_units)
    ]
    if top_n is not None:
        keep = totals.loc[keep].sort_values(ascending=False).head(top_n).index

    net = net[net["sku_id"].isin(keep)].copy()

    # Dense calendar per SKU, from its first to its last observed sale.
    frames = []
    prices = tx.groupby("sku_id")["unit_price"].median()
    names = tx.groupby("sku_id")["name"].agg(
        lambda s: s.value_counts().index[0] if len(s.value_counts()) else ""
    )
    for sku, grp in net.groupby("sku_id", sort=False):
        idx = pd.date_range(grp["day"].min(), grp["day"].max(), freq="D")
        s = grp.set_index("day")["units"].reindex(idx, fill_value=0)
        frames.append(
            pd.DataFrame(
                {
                    "sku_id": sku,
                    "day": idx,
                    "units": s.to_numpy(dtype=float),
                    "unit_price": float(prices.get(sku, np.nan)),
                    "name": names.get(sku, ""),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["sku_id", "day"]).reset_index(drop=True)
