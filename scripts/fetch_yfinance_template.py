"""TEMPLATE for retrieving adjusted closes from Yahoo Finance. NOT RUN in the Phase A environment (market-data hosts are blocked there).

Run it locally, then pass the output through prepare_prices.py. yfinance has no delisted tickers, so a universe built from it is a
survivor universe; stocknet.data.coverage_by_date quantifies the resulting bias against the point-in-time membership file.

    pip install yfinance
    python scripts/fetch_yfinance_template.py --membership path/to/sp_500_historical_components.csv --start 2003-01-01 --end 2024-12-31 --out data/raw_yf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import data as dt  # noqa: E402


def main() -> None:
    import yfinance as yf  # imported here so the template can be read without the dependency
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", required=True)
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = dt.load_membership(a.membership, a.start, a.end)
    tickers = sorted(set().union(*m.values))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        df = yf.download(batch, start=a.start, end=a.end, auto_adjust=True, actions=False, progress=False, group_by="ticker")
        for t in batch:
            try:
                sub = df[t].dropna(how="all")
            except KeyError:
                continue
            if len(sub):
                sub.to_csv(out / f"{t}.csv")
    print(f"requested {len(tickers)} tickers; wrote files for those returned by Yahoo Finance (the rest are delisted or unavailable)")


if __name__ == "__main__":
    main()
