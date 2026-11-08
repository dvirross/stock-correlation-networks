"""Rerun the sector-dependent rolling statistics with dated GICS labels instead of end-2024 labels.

Needs the private price panel (data/panel_adjusted_close.csv, see data/README.md) and
data/derived/sector_verification.csv from scripts/verify_sectors.py. Run from research/stock-networks:

    python scripts/run_pointintime_sectors.py

Label rule for a window ending on date e: the GICS sector of the last constituents snapshot dated on or before e.
Before the first snapshot (2012-12-27) no dated label exists, so the first snapshot's label is used and the row is
flagged `labels_dated=False`. The MSTs are rebuilt exactly as in run_part2.py (same windows, same correlation), so the
only difference from results/rolling_structure_monthly.csv and results/regime_table.csv is the labels.

Writes results/rolling_same_sector_pointintime.csv, results/regime_same_sector_pointintime.csv and
results/sector_pointintime_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import filters as fl, networks as nw, pipeline as pl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FIRST_SNAPSHOT = pd.Timestamp("2012-12-27")
REGIMES = {"Calm 2007": "2007-06-29", "Crisis 2008": "2008-12-31", "Calm 2017": "2017-12-29",
           "COVID 2020": "2020-06-30", "Rates 2022": "2022-12-30", "Latest 2024": "2024-12-30"}


def load_history() -> dict[str, list[tuple[pd.Timestamp, str]]]:
    v = pd.read_csv(ROOT / "data/derived/sector_verification.csv")
    hist = {}
    for t, h in zip(v.ticker, v.history):
        items = [x.split(": ", 1) for x in str(h).split("; ")]
        hist[t] = [(pd.Timestamp(d), s) for d, s in items]
    return hist


def labels_at(hist, end: pd.Timestamp) -> dict[str, str]:
    out = {}
    for t, h in hist.items():
        lab = h[0][1]
        for d, s in h:
            if d <= end:
                lab = s
        # Pre-2018 telecom name and the later Communication Services are different sectors; keep the dated name.
        out[t] = lab
    return out


def same_sector_share(g, labels) -> float:
    e = list(g.edges())
    return float(np.mean([labels[u] == labels[v] for u, v in e]))


def mst_at(returns, end, length=252):
    w = pl.window_ending(returns, end, length)
    c = nw.correlation(w)
    c = c.loc[sorted(c.index), sorted(c.columns)]
    return fl.mst_from_corr(c)


def main() -> None:
    _, returns, sec_2024 = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
    sec_2024 = {k: v for k, v in sec_2024.items() if k in returns.columns}
    hist = load_history()
    roll, graphs = pl.rolling_structure(returns, sec_2024)
    rows = []
    for e, g in graphs.items():
        lab = labels_at(hist, pd.Timestamp(e))
        rows.append(dict(date=e, same_sector_end2024_labels=same_sector_share(g, sec_2024),
                         same_sector_dated_labels=same_sector_share(g, lab), labels_dated=pd.Timestamp(e) >= FIRST_SNAPSHOT,
                         tickers_relabelled=sum(lab[t] != sec_2024[t] for t in sec_2024)))
    df = pd.DataFrame(rows).set_index("date")
    df["difference"] = df.same_sector_dated_labels - df.same_sector_end2024_labels
    df.round(5).to_csv(R / "rolling_same_sector_pointintime.csv")
    # consistency with the published series
    pub = pd.read_csv(R / "rolling_structure_monthly.csv", index_col=0, parse_dates=True)
    max_gap = float((pub.same_sector_edge_share - df.same_sector_end2024_labels.values).abs().max())
    reg = []
    for name, end in REGIMES.items():
        g = mst_at(returns, end)
        lab = labels_at(hist, pd.Timestamp(end))
        reg.append(dict(window=name, end=end, mst_same_sector_end2024=same_sector_share(g, sec_2024),
                        mst_same_sector_dated=same_sector_share(g, lab), labels_dated=pd.Timestamp(end) >= FIRST_SNAPSHOT))
    pd.DataFrame(reg).round(4).to_csv(R / "regime_same_sector_pointintime.csv", index=False)
    dated = df[df.labels_dated]
    summary = {
        "windows": int(len(df)), "windows_with_dated_labels": int(len(dated)),
        "max_abs_difference_dated_windows": float(dated.difference.abs().max()),
        "mean_difference_dated_windows": float(dated.difference.mean()),
        "dated_series_min": float(dated.same_sector_dated_labels.min()), "dated_series_max": float(dated.same_sector_dated_labels.max()),
        "reproduces_published_series_max_abs_gap": max_gap,
    }
    (R / "sector_pointintime_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(reg).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
