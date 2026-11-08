"""Exploratory pooled-block bootstrap sensitivity for non-overlapping MST persistence.

For a 504-day observed return panel ending at a specified date, the first
252 observations and last 252 observations form two separate chronological
years. The observed MST edge Jaccard is compared to a pseudo-null generated
by drawing two INDEPENDENT 252-day block-bootstrap samples from the pooled
504-day observations. The same date blocks are drawn jointly for all stocks.

Hypothesis/limitation: exchangeability (approximate local stationarity) of the
two observed halves is assumed under the working null. Pooling two genuinely
different regimes imposes that null rather than verifying it. The one-sided
Monte Carlo tail fraction is descriptive conditional on this chosen sampling
scheme; neither causal attribution nor universal regime-change significance
follows. Vary the block length, the endpoint and universe.

Example:
 python research/stock-networks/editorial-validation/temporal_stationary_null.py \
   --prices-csv research/stock-networks/data/panel_adjusted_close.csv \
   --ends 2008-12-31 2017-12-29 2024-12-30 \
   --blocks 5 10 21 42 --replicates 400 --output /tmp/temporal_pooled_bootstrap.csv

For the independent 89-stock pilot add --pilot89 and pass the user's original
close_data.csv. No source CSV is exported or committed by this script.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def mst_edges(window: np.ndarray) -> set[tuple[int, int]]:
    c = np.corrcoef(window, rowvar=False)
    if not np.isfinite(c).all():
        raise ValueError("Non-finite correlation matrix; check constant or missing return series")
    d = np.sqrt(np.maximum(0.0, 2 * (1 - np.clip(c, -1, 1))))
    np.fill_diagonal(d, 0.0)
    iu = np.triu_indices(len(d), 1)
    if np.any(d[iu] == 0):
        raise ValueError("Perfectly correlated distinct series; sparse zero-length edge needs special handling")
    t = minimum_spanning_tree(csr_matrix(d)).tocoo()
    if t.nnz != len(d) - 1:
        raise RuntimeError("MST did not connect all names")
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(t.row, t.col)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def sample_blocks(n: int, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, math.ceil(length / block))
    return (starts[:, None] + np.arange(block)).ravel()[:length]


def compare_years(pooled_returns: np.ndarray, block: int, replicates: int, seed: int) -> dict:
    if len(pooled_returns) != 504:
        raise ValueError("Need exactly 504 returns: 252 earlier + 252 later")
    observed = jaccard(mst_edges(pooled_returns[:252]), mst_edges(pooled_returns[252:]))
    rng = np.random.default_rng(seed)
    null = np.empty(replicates)
    for b in range(replicates):
        left = pooled_returns[sample_blocks(504, 252, block, rng)]
        right = pooled_returns[sample_blocks(504, 252, block, rng)]
        null[b] = jaccard(mst_edges(left), mst_edges(right))
    return {
        "observed_nonoverlap_jaccard": round(float(observed), 6),
        "conditional_null_median": round(float(np.median(null)), 6),
        "conditional_null_p05": round(float(np.quantile(null, .05)), 6),
        "conditional_null_p95": round(float(np.quantile(null, .95)), 6),
        "conditional_lower_tail_fraction": round(float((1 + (null <= observed).sum()) / (replicates + 1)), 6),
        "replicates": replicates, "block_days": block,
        "working_null": "two independent 252-day samples from pooled 504-day stationary block distribution",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices-csv", type=Path, required=True)
    p.add_argument("--ends", nargs="+", default=["2008-12-31", "2017-12-29", "2024-12-30"])
    p.add_argument("--blocks", nargs="+", type=int, default=[21])
    p.add_argument("--replicates", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--pilot89", action="store_true",
                   help="Use the fixed 89-stock 2016–2024 survivor panel from validate_stocknet.py")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not all(1 <= block <= 252 for block in a.blocks) or a.replicates < 100:
        p.error("Require 1<=block<=252 and at least 100 replicates")
    prices = pd.read_csv(a.prices_csv, index_col=0, parse_dates=True).sort_index()
    if a.pilot89:
        from validate_stocknet import TICKERS
        prices = prices.loc["2016-01-01":"2024-12-30", TICKERS]
    returns = prices.pct_change(fill_method=None).iloc[1:]
    rows = []
    for end in a.ends:
        observed = returns.loc[:end].tail(504)
        if len(observed) != 504 or observed.isna().any().any():
            raise ValueError(f"504-day window incomplete at {end}; do not silently remove stocks")
        for block in a.blocks:
            result = compare_years(observed.to_numpy(dtype=float), block, a.replicates,
                                   a.seed + block + int(end[:4]))
            rows.append({"window_end": end, "n_stocks": len(observed.columns),
                         "earlier_end": str(observed.index[251].date()),
                         "later_start": str(observed.index[252].date()), **result})
            print(end, "block", block, result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        out.writeheader()
        out.writerows(rows)


if __name__ == "__main__":
    main()
