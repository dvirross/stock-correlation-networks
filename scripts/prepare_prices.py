"""Convert price exports into the canonical wide file and write a provenance + quality report.

Supported inputs
  --per-ticker-dir DIR   one CSV per ticker in the yfinance multi-header export format (rows: Price/Ticker/Date headers, then data),
                         e.g. the Drive folders data2003 and data_2004_2024
  --wide FILE            a wide CSV (date column plus one close column per ticker), e.g. close_data.csv

Usage (from research/stock-networks):
  python scripts/prepare_prices.py --per-ticker-dir path/to/data2003 --field Close --adjusted yes --source "yfinance, retrieved 2025-03-27" --out data/prices_adjusted_close.csv

Nothing is fabricated or repaired: empty or unreadable ticker files are listed as missing, and the quality report flags suspect jumps.
`--adjusted` records whether the Close field is split+dividend adjusted (yfinance auto_adjust=True is); it is written to provenance and
must be set truthfully because the return convention depends on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import data as dt  # noqa: E402


def read_yf_export(path: Path, field: str) -> pd.Series | None:
    try:
        df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)   # drops the Ticker and Date header rows
    except Exception:
        return None
    if field not in df.columns or df.empty:
        return None
    s = pd.to_numeric(df[field], errors="coerce").dropna()
    return s.rename(path.stem) if len(s) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--per-ticker-dir")
    g.add_argument("--wide")
    ap.add_argument("--field", default="Close")
    ap.add_argument("--adjusted", choices=["yes", "no", "unknown"], required=True)
    ap.add_argument("--source", required=True, help="free-text provenance, including retrieval date")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    missing: list[str] = []
    if a.per_ticker_dir:
        cols = {}
        for f in sorted(Path(a.per_ticker_dir).glob("*.csv")):
            s = read_yf_export(f, a.field)
            if s is None:
                missing.append(f.stem)
            else:
                cols[f.stem] = s
        prices = pd.DataFrame(cols).sort_index()
        prices.index.name = "date"
    else:
        prices = pd.read_csv(a.wide, parse_dates=[0], index_col=0).sort_index()
        prices.index.name = "date"
    prices = dt.validate_prices(prices)
    q = dt.quality_report(prices)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out)
    prov = dict(source=a.source, field=a.field, adjusted_for_splits_and_dividends=a.adjusted, first_date=str(prices.index.min().date()),
                last_date=str(prices.index.max().date()), n_dates=len(prices), n_tickers=prices.shape[1], unreadable_or_empty_tickers=missing,
                tickers_with_over_20pct_missing=[t for t, v in q.missing_share.items() if v > 0.2], short_history_tickers=q.short_history,
                suspect_jumps=len(q.suspect_jumps), suspect_jumps_reversed=int(q.suspect_jumps.reversed.sum()) if len(q.suspect_jumps) else 0,
                sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    q.suspect_jumps.to_csv(out.with_suffix(".suspect_jumps.csv"), index=False)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
