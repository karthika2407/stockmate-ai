# Reference data: GST rate schedule

## What this folder holds

`gst_rates.csv` is the GST rate schedule that StockMate AI judges supplier bills
against. It is reference **data**, deliberately not constants in the code, for
one reason: Indian GST rates change by Council decision, and a schedule baked
into a Python module cannot be refreshed by the person who actually needs it
refreshed.

Columns:

| column | meaning |
| --- | --- |
| `hsn_prefix` | 2, 4, 6 or 8 digit HSN prefix. Lookup is longest-prefix: an 8 digit HSN on a bill falls back to 6, then 4, then 2 digits. |
| `description` | What the row is meant to cover, in shop language. |
| `rate_percent` | Total GST percent for that category. CGST and SGST are each half of it. |
| `effective_from` | First date the row applies, ISO format. |
| `effective_to` | Last date the row applies, or empty for open ended. |
| `source` | Where the value came from, written honestly. |
| `confidence` | `verified` only if the row was read off a primary CBIC document. Everything else is `unverified`. |

## Where these values came from, and what that means

**Every row in the shipped file is marked `unverified`.** None of them was read
off a primary document.

We tried to fetch the rate schedule from the primary and near-primary sources
(pib.gov.in press releases, taxinformation.cbic.gov.in, and the usual tax
commentary sites). Those fetches were blocked for automated clients, so we could
not open a notification, read an item-level entry, and copy it. Rather than
write a notification number we have not read, the rows carry a plain-English
source string saying they reflect secondary reporting of the GST Council's
decisions, and `confidence` is `unverified` throughout.

Two things in particular are not captured:

* **Compensation cess.** Tobacco and aerated beverage lines in India can carry
  cess on top of GST. This file holds the GST percent only, so the rule engine
  will under-state the tax due on those lines.
* **Packaging and branding conditions.** Several kirana categories are taxed
  differently depending on whether the goods are pre-packaged and labelled. The
  `description` column names the assumption each row makes; it does not model
  the condition.

## Rates changed recently, and will change again

* The **56th GST Council meeting** (3 September 2025) moved the structure to
  0 / 5 / 18 / 40, effective **22 September 2025**. The current rows in this file
  reflect secondary reporting of that decision.
* A **57th GST Council meeting** was announced on **8 October 2026**. We have no
  verified item-level rates from it, so this file holds **no rows for it at all**.
  If that meeting changed a rate you deal in, this file is already wrong for it.

The pre-22-September-2025 rows are kept so that an old bill is judged against the
schedule that was in force on its invoice date, not against today's.

## An operator must refresh this file before relying on it

Do not put this file in front of a supplier as it stands. Refresh it first:

1. Open the CBIC rate finder at `taxinformation.cbic.gov.in` and, for each
   `hsn_prefix` you actually trade in, look up the current rate. The GST Council
   press releases on `pib.gov.in` carry the decisions; the CBIC notifications
   carry the legally operative schedule, and the notification is what you cite.
2. For a row you confirmed, replace `source` with the exact notification or
   circular reference you read (number and date), and set `confidence` to
   `verified`. Only do this for a document you actually opened.
3. For a rate that changed, do **not** edit the old row's `rate_percent`. Close
   the old row by putting the day before the change in `effective_to`, and add a
   new row starting on the change date. The loader rejects overlapping date
   ranges for the same prefix, so a sloppy edit fails loudly instead of silently
   judging bills against two rates at once.
4. Add prefixes you trade in that are missing. A missing prefix makes
   `GST.RATE_KNOWN` report a gap in this file; it does not accuse the supplier.
5. Re-run the checks after editing:

       E:\Claude\StockMate_AI\.venv\Scripts\python.exe -m pytest tests/test_gst.py
       E:\Claude\StockMate_AI\.venv\Scripts\python.exe benchmarks/bench_gst_rules.py

   The loader validates slab legality, date ordering and overlaps on load, so a
   malformed edit raises with the CSV row number.

## How the code uses it

`stockmate.reference.gst_rates.GstSchedule` loads and validates the file and
exposes `schedule_version` and `source_note`. Every `GST.*` check carries that
version in its evidence, so a finding always says which schedule judged it. If
the schedule was wrong, the finding can be re-examined instead of defended.
