"""Independent real-data website diagnostics; no unbiased index or portfolio-return claims.
Run: python validate_stocknet.py --input close_data.csv --out results
Dependencies: numpy, pandas, scipy, matplotlib.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICKERS = ('AAPL MSFT AMZN GOOGL META NVDA AMD INTC CSCO ORCL IBM CRM ADBE QCOM TXN AVGO '
           'JPM BAC GS MS C WFC USB PNC BLK V MA AXP COF UNH JNJ PFE MRK ABBV ABT MDT TMO BMY AMGN ISRG LLY '
           'XOM CVX COP SLB EOG MPC PSX OXY VLO PG KO PEP WMT COST HD LOW MCD SBUX NKE TGT CL DUK SO NEE '
           'AEP EXC D DTE CAT DE HON GE BA UPS FDX UNP CSX RTX LMT AMT PLD SPG O EQIX EQR AVB PSA BXP').split()
LOOKBACKS = (63, 126, 252, 504)
OVERLAP_SHIFTS = (1, 5, 10, 21, 42, 63, 84, 126, 168, 210, 252)
B, BLOCK, SEED = 180, 5, 20261008
END = '2024-12-30'

def correlation(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == 'spearman':
        x = rankdata(x, axis=0)
    return np.corrcoef(x, rowvar=False)

def mst_edges(c: np.ndarray) -> set[tuple[int, int]]:
    n = len(c)
    dist = np.sqrt(np.maximum(0, 2 * (1 - np.clip(c, -1, 1))))
    np.fill_diagonal(dist, 0)
    assert np.isfinite(dist).all()
    assert (dist[np.triu_indices(n, 1)] > 0).all(), 'zero offdiagonal distances require special handling'
    tree = minimum_spanning_tree(csr_matrix(dist)).tocoo()
    assert tree.nnz == n - 1
    return {tuple(sorted((int(i), int(j)))) for i, j in zip(tree.row, tree.col)}

def bootstrap_reference_edge_support(x: np.ndarray, kind: str, seed: int) -> tuple[set, np.ndarray]:
    rng = np.random.default_rng(seed)
    reference = mst_edges(correlation(x, kind))
    count = {e: 0 for e in reference}
    t = len(x)
    for _ in range(B):
        starts = rng.integers(0, t - BLOCK + 1, size=math.ceil(t / BLOCK))
        indices = (starts[:, None] + np.arange(BLOCK)).ravel()[:t]
        replicate = mst_edges(correlation(x[indices], kind))
        for e in reference & replicate:
            count[e] += 1
    return reference, np.array([cnt / B for cnt in count.values()])

def render(x, y_series, filename, xlabel, ylabel, title, footnote):
    fig, ax = plt.subplots(figsize=(8.4, 5.4), layout='constrained')
    for label, values in y_series:
        ax.plot(x, values, marker='o', label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=.22)
    ax.set_ylim(0, 1)
    if len(y_series) > 1: ax.legend()
    ax.text(0, -.19, footnote, transform=ax.transAxes, fontsize=8.5)
    fig.savefig(filename, dpi=160, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    options = parser.parse_args()
    dest = Path(options.out)
    dest.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(options.input, index_col=0, parse_dates=True).sort_index()
    prices = raw.loc['2016-01-01':END, TICKERS]
    assert len(TICKERS) == 89
    assert prices.notna().all().all(), 'do not silently remove tickers or fill missing prices'
    daily = prices.pct_change(fill_method=None).iloc[1:]
    assert daily.notna().all().all()
    assert daily.index[-1] == pd.Timestamp(END)
    summaries = []
    for window in LOOKBACKS:
        obs = daily.tail(window).to_numpy()
        a, pearson = bootstrap_reference_edge_support(obs, 'pearson', seed=SEED + window)
        b, spearman = bootstrap_reference_edge_support(obs, 'spearman', seed=SEED + window)
        summaries.append(dict(window_days=window, n_assets=len(TICKERS), B=B, block_days=BLOCK,
            pearson_median_support=float(np.median(pearson)), spearman_median_support=float(np.median(spearman)),
            pearson_edges_support_ge_08=int((pearson >= .8).sum()),
            spearman_edges_support_ge_08=int((spearman >= .8).sum()),
            pearson_spearman_mst_jaccard=len(a & b)/len(a | b)))
    s = pd.DataFrame(summaries)
    s.to_csv(dest/'reliability_vs_window.csv', index=False)
    render(s.window_days,[('Pearson',s.pearson_median_support),('Spearman',s.spearman_median_support)],
        dest/'reliability_vs_window.png','Lookback (trading days)',
        'Median reference-edge bootstrap inclusion',
        'MST reliability is sensitive to estimation history',
        '89 fixed survivor-conditioned stocks; real returns ending 2024-12-30; B=180, 5-day blocks.\nLonger windows also include different market regimes.')
    ref = mst_edges(correlation(daily.tail(252).to_numpy(), 'pearson'))
    rows=[]
    for shift in OVERLAP_SHIFTS:
        x = daily.iloc[-252-shift:-shift].to_numpy()
        assert len(x) == 252
        old = mst_edges(correlation(x, 'pearson'))
        rows.append(dict(shift_days=shift, shared_observations=252-shift,
                         shared_fraction=(252-shift)/252, shared_MST_edges=len(ref & old),
                         edge_jaccard=len(ref & old)/len(ref | old)))
    overlaps = pd.DataFrame(rows)
    overlaps.to_csv(dest/'rolling_window_overlap.csv', index=False)
    render(overlaps.shared_fraction,[('Observed edge overlap',overlaps.edge_jaccard)],
        dest/'rolling_window_overlap.png',
        'Fraction of observations shared by two 252-day windows',
        'MST edge Jaccard', 'The overlapping-window stability trap',
        'Real observed returns, 89 survivor-conditioned stocks; both windows contain 252 days.\nDifferences mix sampling overlap with genuine changes in the return distribution.')
    metadata = {'origin':'historical Yahoo Finance export (March 2025)',
                'data_sha256':hashlib.sha256(Path(options.input).read_bytes()).hexdigest(),
                'analysis_start':'2016-01-01','analysis_end':END,'n_assets':len(TICKERS),
                'universe':'manually curated, survivor-conditioned subset, NOT point-in-time S&P index',
                'tickers':TICKERS,
                'price_adjustment':'likely yfinance auto_adjust=True in 2025, exact source version unpinned',
                'bootstrap':'joint moving block, same sampled dates across all stocks',
                'interpretation':'descriptive network reliability, not alpha/causality or unbiased market-wide inference'}
    (dest/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('DATA',len(daily),'returns',len(TICKERS),'assets; SHA256',metadata['data_sha256'])
    print('BOOTSTRAP\n'+s.round(4).to_string(index=False))
    print('OVERLAP\n'+overlaps.round(4).to_string(index=False))

if __name__ == '__main__':
    main()
