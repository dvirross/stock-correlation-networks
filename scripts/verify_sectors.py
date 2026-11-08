"""Verify the panel's sector labels against dated GICS snapshots of the S&P 500 constituents.

Source: the public dataset https://github.com/datasets/s-and-p-500-companies (file data/constituents.csv),
which is built from Wikipedia's "List of S&P 500 companies" (GICS Sector column). Every commit of that file is
a dated snapshot. Export the snapshots once with git, one CSV per commit named <YYYY-MM-DD>_<sha>.csv:

    git clone --filter=blob:none https://github.com/datasets/s-and-p-500-companies.git sp500ds
    mkdir snapshots
    git -C sp500ds log --format="%h %ad" --date=short --reverse --before=2025-01-01 -- data/constituents.csv |
      while read h d; do git -C sp500ds show $h:data/constituents.csv > snapshots/${d}_$h.csv; done

Then run (from this folder, not from inside the clone):

    python -I scripts/verify_sectors.py --snapshots <path>/snapshots

Writes data/derived/sector_verification.csv (one row per panel ticker: label used, GICS sector in the last snapshot
before 2025, whether they agree, first and last snapshot dates seen, every recorded sector change) and
data/derived/sector_verification_summary.json. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_snapshot(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    keys = list(rows[0].keys())
    tk = next((k for k in keys if k.strip() in ("Symbol", "Ticker symbol", "Ticker")), None)
    sk = next((k for k in keys if "Sector" in k), None)
    if tk is None or sk is None:
        return {}
    snap = {r[tk].strip().replace(".", "-"): (r.get(sk) or "").strip() for r in rows if r.get(tk) and r.get(sk)}
    # Some historical commits have shifted or mislabelled columns (sub-industries under the sector header).
    # A snapshot is used only if at least 95% of its values are GICS sector names; otherwise it is skipped.
    valid = sum(1 for v in snap.values() if v in GICS_SECTORS)
    return snap if snap and valid >= 0.95 * len(snap) else {}


# The 11 current GICS sectors, plus the pre-2018 name of Communication Services in both spellings seen in the source.
GICS_SECTORS = {
    "Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials", "Health Care",
    "Industrials", "Information Technology", "Materials", "Real Estate", "Utilities",
    "Telecommunication Services", "Telecommunications Services",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", type=Path, required=True)
    args = ap.parse_args()
    with (ROOT / "data/sectors.csv").open(newline="", encoding="utf-8") as f:
        labels = {r["ticker"]: r["sector"] for r in csv.DictReader(f)}
    files = sorted(p for p in args.snapshots.glob("*.csv") if p.name[:4].isdigit())
    history: dict[str, list[tuple[str, str]]] = {t: [] for t in labels}
    seen: dict[str, list[str]] = {t: [] for t in labels}
    used = 0
    for p in files:
        date = p.name[:10]
        snap = read_snapshot(p)
        if not snap:
            continue
        used += 1
        for t in labels:
            if t in snap:
                seen[t].append(date)
                if not history[t] or history[t][-1][1] != snap[t]:
                    history[t].append((date, snap[t]))
    rows = []
    for t, lab in sorted(labels.items()):
        h = history[t]
        last = h[-1][1] if h else ""
        rows.append({
            "ticker": t, "label_used": lab, "gics_last_snapshot": last, "agrees": bool(h) and last == lab,
            "first_seen": seen[t][0] if seen[t] else "", "last_seen": seen[t][-1] if seen[t] else "",
            "n_sector_changes": max(len(h) - 1, 0), "history": "; ".join(f"{d}: {s}" for d, s in h),
        })
    out = ROOT / "data/derived/sector_verification.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "source": "https://github.com/datasets/s-and-p-500-companies data/constituents.csv (from Wikipedia, GICS Sector)",
        "snapshots_used": used,
        "first_snapshot": files[0].name[:10] if files else None,
        "last_snapshot_before_2025": files[-1].name[:10] if files else None,
        "panel_tickers": len(rows),
        "found_in_snapshots": sum(1 for r in rows if r["last_seen"]),
        "agree_with_last_snapshot": sum(1 for r in rows if r["agrees"]),
        "tickers_with_sector_change": [r["ticker"] for r in rows if r["n_sector_changes"] > 0],
    }
    (ROOT / "data/derived/sector_verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
