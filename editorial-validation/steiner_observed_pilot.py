"""Reproduce the independent observed one-hub Steiner pilot from private historical closes.

Requires the user's existing close_data.csv; does not redistribute it.
Run from repository root:
python research/stock-networks/editorial-validation/steiner_observed_pilot.py \
  --input /path/to/close_data.csv --out /path/to/derived/results

The panel is explicitly survivorship-conditioned and selected retrospectively.
No inference about historical S&P 500 membership or traded portfolios is valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from validate_stocknet import TICKERS
from steiner_single_hub import compare_one_hub, graph_from_correlation

ENDS = ("2019-12-31", "2020-04-30", "2022-10-31", "2024-12-30")
GROUPS = {
    "mega_cap_tech_4": ("AAPL", "MSFT", "AMZN", "GOOGL"),
    "large_banks_4": ("JPM", "BAC", "GS", "MS"),
    "healthcare_4": ("UNH", "JNJ", "PFE", "MRK"),
    "cross_sector_4": ("AAPL", "JPM", "XOM", "JNJ"),
    "mega_cap_tech_8": ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "AMD", "ADBE", "CRM"),
    "banks_8": ("JPM", "BAC", "GS", "MS", "WFC", "USB", "PNC", "C"),
    "healthcare_8": ("UNH", "JNJ", "PFE", "MRK", "ABBV", "ABT", "BMY", "LLY"),
    "diverse_8": ("AAPL", "JPM", "XOM", "JNJ", "CAT", "PG", "DUK", "AMT"),
}
RANDOM_K = (3, 4, 6, 8, 12)
N_RANDOM_PER_CELL = 50
SEED = 20261008


def subgraph_mst_cost(dist: np.ndarray, idx: np.ndarray) -> float:
    """Small positive-weight dense graph, SciPy MST on its adjacency matrix."""
    sub = dist[np.ix_(idx, idx)]
    return float(minimum_spanning_tree(csr_matrix(sub)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    px = pd.read_csv(args.input, index_col=0, parse_dates=True).sort_index()
    px = px.loc["2016-01-01":"2024-12-30", TICKERS]
    if len(TICKERS) != 89 or px.isna().any().any():
        raise ValueError("Fixed 89-stock price panel incomplete; no imputation permitted.")
    ret = px.pct_change(fill_method=None).iloc[1:]
    if ret.isna().any().any():
        raise ValueError("Incomplete daily return observations.")

    rng = np.random.default_rng(SEED)
    preset, random_rows = [], []
    names = list(ret.columns)
    for end in ENDS:
        w = ret.loc[:end].tail(252)
        if len(w) != 252 or w.isna().any().any():
            raise ValueError("Incomplete endpoint lookback " + end)
        corr = w.corr().to_numpy()
        graph = graph_from_correlation(names, corr)
        dist = np.sqrt(np.maximum(0, 2 * (1 - corr)))
        np.fill_diagonal(dist, 0)

        for group, terms in GROUPS.items():
            result = compare_one_hub(graph, list(terms))
            preset.append(dict(window_end=end, group=group, terminal_count=len(terms),
                               terminal_tickers=" ".join(terms), **result))

        for k in RANDOM_K:
            for rep in range(N_RANDOM_PER_CELL):
                tt = rng.choice(len(names), size=k, replace=False)
                base = subgraph_mst_cost(dist, tt)
                best, best_hub = base, ""
                included = set(tt)
                for h in range(len(names)):
                    if h in included:
                        continue
                    c = subgraph_mst_cost(dist, np.r_[tt, h])
                    if c < best - 1e-10:
                        best, best_hub = c, names[h]
                random_rows.append(dict(window_end=end, terminals_n=k, replicate=rep,
                                        terminal_only_cost=base, best_single_hub_cost=best,
                                        gain_pct=100 * (base - best) / base,
                                        best_hub=best_hub))

    p = pd.DataFrame(preset)
    r = pd.DataFrame(random_rows)
    summary = (r.groupby(["window_end", "terminals_n"])
               .agg(n=("replicate", "size"),
                    positive=("gain_pct", lambda x: int((x > 1e-7).sum())),
                    mean_gain_pct=("gain_pct", "mean"),
                    maximum_gain_pct=("gain_pct", "max"))
               .reset_index())
    p.to_csv(args.out / "prespecified_group_results.csv", index=False)
    r.to_csv(args.out / "exploratory_random_group_results.csv", index=False)
    summary.to_csv(args.out / "exploratory_random_summary.csv", index=False)

    provenance = dict(source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      source_name=args.input.name,
                      n_assets=len(TICKERS), n_return_days=len(ret),
                      dates=ENDS, groups=GROUPS, sample_sizes=RANDOM_K,
                      random_sets_per_date_size=N_RANDOM_PER_CELL,
                      random_seed=SEED, window=252,
                      returns="simple pct_change(fill_method=None)",
                      interpretation="bounded at-most-one-stock Steiner diagnostic; retrospective survivor panel",
                      caveats=["not unrestricted Steiner optimum",
                               "not a point-in-time investable sample",
                               "terminal groups exploratory, 8-stock expansion after 4-stock pilot",
                               "random sets share data and cannot be treated as independent trials"])
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Preset experiments:", len(p), "positive:", int((p.gain_vs_terminal_mst_pct > 1e-7).sum()))
    print("Random controls:", len(r), "positive:", int((r.gain_pct > 1e-7).sum()))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
