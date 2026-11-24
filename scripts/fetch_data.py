"""Download the two public datasets the benchmarks run on.

Neither dataset is committed.  Online Retail II is a 43 MB workbook and CORD-v2
is roughly 2 GB of photographed receipts, and both are redistributable only
under their own licences, so the repository records where they came from and
fetches them on demand instead of vendoring them.

Run:
    python scripts/fetch_data.py --all
    python scripts/fetch_data.py --retail          (43 MB, needed for forecasting)
    python scripts/fetch_data.py --receipts        (about 2 GB, needed for audit)
    python scripts/fetch_data.py --receipts --splits test validation   (330 MB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "_datasets"

RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
CORD_PARQUET_API = "https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/parquet"

SOURCES = {
    "online_retail_ii": {
        "name": "Online Retail II",
        "publisher": "UCI Machine Learning Repository (dataset 502)",
        "doi": "10.24432/C5CG6D",
        "licence": "CC BY 4.0",
        "url": RETAIL_URL,
        "used_for": "demand forecasting, replenishment simulation, price-creep history",
    },
    "cord_v2": {
        "name": "CORD-v2",
        "publisher": "NAVER CLOVA (naver-clova-ix/cord-v2 on Hugging Face)",
        "citation": "Park et al., CORD: A Consolidated Receipt Dataset for Post-OCR Parsing, 2019",
        "licence": "CC BY 4.0",
        "url": CORD_PARQUET_API,
        "used_for": "receipt extraction accuracy and bill-audit fault detection",
    },
}


def _open(url: str, timeout: int = 600):
    req = urllib.request.Request(url, headers={"User-Agent": "stockmate-research/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def _sha256(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    """Digest of the first chunk, enough to notice a truncated download."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def fetch_retail(root: Path) -> dict:
    out = root / "online_retail_ii"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "online_retail_ii.zip"

    if not zip_path.exists():
        print(f"downloading Online Retail II (43 MB) -> {zip_path}")
        t = time.time()
        with _open(RETAIL_URL) as r, open(zip_path, "wb") as fh:
            fh.write(r.read())
        print(f"  done in {time.time() - t:.0f}s")
    else:
        print("Online Retail II already present")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    xlsx = next(out.glob("*.xlsx"), None)
    if xlsx is None:
        raise RuntimeError("the archive did not contain the expected workbook")
    print(f"  ready: {xlsx} ({xlsx.stat().st_size / 1e6:.1f} MB)")
    return {"path": str(xlsx), "bytes": xlsx.stat().st_size, "sha256_head": _sha256(xlsx)}


def fetch_cord(root: Path, splits: tuple[str, ...]) -> dict:
    out = root / "cord_v2"
    out.mkdir(parents=True, exist_ok=True)

    print(f"listing CORD-v2 shards for splits {splits}")
    meta = json.load(_open(CORD_PARQUET_API, timeout=120))

    urls: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and node.endswith(".parquet"):
            urls.append(node)

    walk(meta)
    wanted = [u for u in urls if any(f"/{s}/" in u or f"_{s}_" in u for s in splits)]
    if not wanted:
        wanted = urls
    print(f"  {len(wanted)} shard(s) to fetch")

    files = []
    for url in wanted:
        name = "_".join(url.split("/")[-3:])
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip {name}")
            files.append(str(dest))
            continue
        for attempt in range(3):
            try:
                t = time.time()
                with _open(url) as r, open(dest, "wb") as fh:
                    fh.write(r.read())
                print(f"  {name}  {dest.stat().st_size / 1e6:.0f} MB  {time.time() - t:.0f}s")
                files.append(str(dest))
                break
            except Exception as exc:
                print(f"  retry {name}: {exc}")
                time.sleep(3)
        else:
            print(f"  FAILED {name}")
    return {"files": files, "count": len(files)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--retail", action="store_true")
    ap.add_argument("--receipts", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    args = ap.parse_args()

    if not (args.retail or args.receipts or args.all):
        ap.print_help()
        return 1

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    record = {"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": {}}

    if args.retail or args.all:
        record["sources"]["online_retail_ii"] = SOURCES["online_retail_ii"] | fetch_retail(root)
    if args.receipts or args.all:
        record["sources"]["cord_v2"] = (
            SOURCES["cord_v2"] | fetch_cord(root, tuple(args.splits)) | {"splits": args.splits}
        )

    prov = Path(__file__).resolve().parents[1] / "data" / "PROVENANCE.json"
    prov.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(prov.read_text(encoding="utf-8")) if prov.exists() else {}
    existing.update(record)
    prov.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nprovenance written to {prov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
