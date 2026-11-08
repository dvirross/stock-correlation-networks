"""Export the derived data behind the interactive tools on dvirross.com (no prices).

Needs the private price panel data/panel_adjusted_close.csv (see data/README.md) and data/derived/sector_verification.csv.
Run from the repository root:

    python scripts/export_tool_data.py            # everything (PMFG takes about 100 s per year)
    python scripts/export_tool_data.py --no-pmfg  # skip the PMFG graphs

Writes tool-data/ (all JSON, compact):
  meta.json                tickers (sorted, the index used by every other file), end-2024 sector, per-year dated sector,
                           window end dates, provenance, encoding notes
  corr/<year>.json         Pearson correlations of daily returns over the 252 trading days ending on the last trading day of
                           <year> (2004 to 2024), upper triangle row by row (i < j), as integers = round(1000 * rho)
  graphs/<year>.json       MST and PMFG edges as index pairs, and greedy-modularity communities of each (same function as the
                           article, stocknet.networks.communities)
  bootstrap/<end>.json     300 joint moving-block bootstrap MSTs (21-day blocks, seed 20241230, the article's settings) of the
                           windows ending 2024-12-30 and 2008-12-31, as index pairs, plus the original MST
  checks.json              validations: 2024 bootstrap inclusion reproduces results/bootstrap_edge_inclusion_mst.csv; the 2024
                           matrix reproduces the headline MST in results/headline_graph_edges.csv

Only derived statistics are written; no price or return series. Standard rounding to 3 decimals is enough for the tools
(the article's figures use the full-precision results).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import filters as fl, networks as nw, pipeline as pl, reliability as rl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tool-data"
BLOCK, SEED, N_BOOT, LENGTH = 21, 20241230, 300, 252
BOOT_ENDS = ["2024-12-30", "2008-12-31"]


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n", encoding="utf-8")


def pairs(g, index) -> list[list[int]]:
    return sorted([sorted([index[u], index[v]]) for u, v in g.edges()])


def dated_labels(tickers, year_end: pd.Timestamp) -> list[str]:
    v = pd.read_csv(ROOT / "data/derived/sector_verification.csv").set_index("ticker")
    out = []
    for t in tickers:
        items = [x.split(": ", 1) for x in str(v.loc[t, "history"]).split("; ")]
        lab = items[0][1]
        for d, s in items:
            if pd.Timestamp(d) <= year_end:
                lab = s
        out.append(lab)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-pmfg", action="store_true")
    args = ap.parse_args()
    _, returns, sec = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
    tickers = sorted(returns.columns)
    returns = returns[tickers]
    index = {t: i for i, t in enumerate(tickers)}
    iu = np.triu_indices(len(tickers), k=1)
    years, sector_dated = [], {}
    for year in range(2004, 2025):
        end = returns.loc[:f"{year}-12-31"].index[-1]
        w = pl.window_ending(returns, end, LENGTH)
        c = nw.correlation(w).loc[tickers, tickers]
        dump(OUT / f"corr/{year}.json", {"year": year, "end": str(end.date()), "days": LENGTH,
                                         "rho_x1000": np.rint(c.to_numpy()[iu] * 1000).astype(int).tolist()})
        mst = fl.mst_from_corr(c)
        graphs = {"mst": pairs(mst, index), "mst_communities": [nw.communities(mst)[t] for t in tickers]}
        if not args.no_pmfg:
            pm = fl.pmfg(c)
            graphs.update(pmfg=pairs(pm, index), pmfg_communities=[nw.communities(pm)[t] for t in tickers])
        dump(OUT / f"graphs/{year}.json", {"year": year, "end": str(end.date()), **graphs})
        years.append({"year": year, "end": str(end.date())})
        sector_dated[year] = dated_labels(tickers, end)
        print(year, end.date(), flush=True)
    checks = {}
    for end in BOOT_ENDS:
        w = pl.window_ending(returns, end, LENGTH)
        base = fl.mst_from_corr(nw.correlation(w))
        boots = list(rl.bootstrap_graphs(w, fl.mst_from_corr, nw.correlation, n_boot=N_BOOT, block=BLOCK, seed=SEED))
        dump(OUT / f"bootstrap/{end}.json", {"end": end, "block": BLOCK, "seed": SEED, "n": N_BOOT,
                                             "original": pairs(base, index), "resamples": [pairs(g, index) for g in boots]})
        if end == "2024-12-30":
            inc = rl.edge_inclusion(base, boots)
            pub = pd.read_csv(ROOT / "results/bootstrap_edge_inclusion_mst.csv")
            m = inc.merge(pub, on=["u", "v"], suffixes=("", "_pub"))
            checks["bootstrap_2024_matches_published"] = {"edges_matched": int(len(m)), "edges_published": int(len(pub)),
                                                          "max_abs_inclusion_gap": float((m.inclusion - m.inclusion_pub).abs().max())}
            head = pd.read_csv(ROOT / "results/headline_graph_edges.csv")
            pub_mst = {frozenset((u, v)) for u, v in zip(head[head.graph == "MST"].u, head[head.graph == "MST"].v)}
            checks["headline_mst_matches_published"] = pub_mst == {frozenset(e) for e in base.edges()}
    dump(OUT / "meta.json", {
        "tickers": tickers,
        "sector_end2024": [sec[t] for t in tickers],
        "sector_dated": {str(y): v for y, v in sector_dated.items()},
        "sector_dated_note": "GICS sector in the last constituents snapshot on or before the window end; before 2012-12-27 the "
                             "first snapshot is used (not truly dated). Source: datasets/s-and-p-500-companies.",
        "windows": years,
        "encoding": "corr/<year>.json rho_x1000 is the upper triangle (i<j, row by row) of the correlation matrix in the "
                    "order of tickers, as round(1000*rho)",
        "provenance": "Yahoo Finance daily closes (yfinance export, March 2025), split- and dividend-adjusted; survivor panel of "
                      "203 continuous S&P 500 members 2003-2024; simple returns; 252-day windows. No prices are included.",
        "generator": "scripts/export_tool_data.py",
    })
    dump(OUT / "checks.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
