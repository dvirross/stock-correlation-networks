"""Audit the Drive price files against point-in-time S&P 500 membership and define the analysis panel.

Inputs (kept outside git, see data/README.md): a folder of yfinance per-ticker CSVs (default data/raw/per_ticker), the listing of
file sizes (listing_all.csv: ticker,file_id,file_size) and sp_500_historical_components.csv.

Panel rule (fixed before any network result was computed):
  1. S&P 500 member on every membership snapshot from 2003-01-06 to 2025-03-25 (continuous membership, same symbol throughout);
  2. at least 99% valid adjusted closes on all trading days 2003-01-02 to 2024-12-30.
Everything else is listed with its reason. This is a LONG-LIVED SURVIVOR panel; it supports structural and stability analysis but not
unbiased performance claims.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import data as dt  # noqa: E402


# Tickers excluded after inspecting every one-day move more than 25 points away from the cross-sectional median outside the 2008-09 and 2020 crises.
ARTIFACTS = {
    "JCI": "excluded: suspected unadjusted corporate action (-53% on 2007-07-02, consistent with the 2-for-1 split)",
    "DHR": "excluded: suspected corporate-action artifact (+61% on 2016-07-05, around the Fortive spin-off)",
}


def read_close(path):
    df = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True)
    return pd.to_numeric(df["Close"], errors="coerce").dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/per_ticker")
    ap.add_argument("--listing", default="data/raw/listing_all.csv")
    ap.add_argument("--membership", default="data/sp_500_historical_components.csv")
    ap.add_argument("--out", default="data/derived")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    mem = dt.load_membership(a.membership, start="2003-01-06")
    at_start = dt.members_at(mem, "2003-01-06")
    continuous = set.intersection(*[set(s) for s in mem.values])
    listing = pd.read_csv(a.listing).set_index("ticker")
    rows, series = [], {}
    for t in sorted(at_start):
        f = Path(a.data_dir) / f"{t}.csv"
        size = float(listing.file_size.get(t, 0)) if t in listing.index else float("nan")
        r = dict(ticker=t, drive_file_size=size, member_continuous=t in continuous, downloaded=f.exists())
        if f.exists():
            s = read_close(f)
            series[t] = s
            r.update(first=str(s.index.min().date()), last=str(s.index.max().date()), n_obs=len(s))
        rows.append(r)
    prices = pd.DataFrame(series).sort_index()
    prices = prices.loc["2003-01-02":"2024-12-30"]
    cal = prices.index
    aud = pd.DataFrame(rows).set_index("ticker")
    aud["valid_share_2003_2024"] = prices.notna().mean().reindex(aud.index)

    def reason(t, r):
        if r.drive_file_size <= 100:
            return "excluded: no data returned by Yahoo Finance (empty file; delisted, acquired or renamed)"
        if not r.downloaded:
            return "excluded: Drive file under 100 KB (about a year or less of history), not downloaded"
        if r.valid_share_2003_2024 < 0.99:
            return "excluded: partial history (starts late or ends early; listing and dates in the audit table)"
        if t in ARTIFACTS:
            return ARTIFACTS[t]
        if not r.member_continuous:
            return "excluded: full price history but not a continuous index member (left the index or symbol identity uncertain)"
        return "panel"
    aud["status"] = [reason(t, r) for t, r in aud.iterrows()]
    aud.to_csv(out / "universe_audit_2003_members.csv")
    panel = list(aud.index[aud.status == "panel"])
    px = prices[panel]
    px.index.name = "date"
    px.to_csv(out.parent / "panel_adjusted_close.csv")          # large; ignored by git (see .gitignore)
    q = dt.quality_report(px)
    q.suspect_jumps.to_csv(out / "panel_suspect_jumps.csv", index=False)
    summary = dict(members_2003_01_06=len(at_start), continuous_members=len(continuous & at_start), panel=len(panel),
                   status_counts=aud.status.value_counts().to_dict(), n_dates=len(px), first=str(px.index.min().date()), last=str(px.index.max().date()),
                   missing_share_max=float(px.isna().mean().max()), tickers_with_any_missing=int((px.isna().any()).sum()),
                   suspect_jumps=len(q.suspect_jumps), suspect_jumps_reversed=int(q.suspect_jumps.reversed.sum()) if len(q.suspect_jumps) else 0,
                   stale_run_max=int(q.stale_runs.max()))
    (out / "panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
