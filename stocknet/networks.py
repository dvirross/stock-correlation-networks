"""Correlation matrices, metric distances, minimum spanning trees, network metrics, communities and stability."""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


def correlation(window: pd.DataFrame, min_periods: int | None = None) -> pd.DataFrame:
    """Pearson correlation of simple returns over a window (pairwise-complete observations)."""
    return window.corr(min_periods=min_periods)


def distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna distance d = sqrt(2 (1 - rho)). It is a true metric on the standardized return vectors (triangle inequality holds)."""
    arr = np.sqrt(2.0 * (1.0 - corr.clip(-1, 1).to_numpy(dtype=float)))
    arr = np.array(arr, copy=True)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)


def mst_edges(dist: pd.DataFrame) -> pd.DataFrame:
    """Minimum spanning tree of the complete distance graph (n-1 edges). Ties are broken deterministically by scipy's algorithm
    given a fixed ordering of ``dist`` (sort tickers before calling for reproducibility)."""
    n = len(dist)
    if n < 2:
        return pd.DataFrame(columns=["u", "v", "distance"])
    w = dist.values.copy()
    w[w == 0] = 1e-12                                # scipy treats 0 as "no edge"
    np.fill_diagonal(w, 0.0)
    tree = minimum_spanning_tree(w).tocoo()
    names = dist.index
    return pd.DataFrame({"u": names[tree.row], "v": names[tree.col], "distance": tree.data}).sort_values(["u", "v"]).reset_index(drop=True)


def mst_graph(dist: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(dist.index)
    for u, v, d in mst_edges(dist).itertuples(index=False):
        g.add_edge(u, v, distance=float(d))
    return g


def network_metrics(g: nx.Graph, corr: pd.DataFrame) -> dict:
    """Window-level structure summaries of an MST plus the matrix they came from."""
    n = g.number_of_nodes()
    deg = dict(g.degree())
    total = sum(d["distance"] for _, _, d in g.edges(data=True))
    iu = np.triu_indices(len(corr), 1)
    eig = np.linalg.eigvalsh(corr.fillna(0).values)
    return dict(
        n=n,
        mean_corr=float(corr.values[iu].mean()),
        first_eigen_share=float(eig[-1] / eig.sum()),
        mst_total_length=total,
        mst_mean_edge=total / max(n - 1, 1),
        max_degree=max(deg.values()) if deg else 0,
        leaf_share=float(np.mean([d == 1 for d in deg.values()])) if deg else np.nan,
        mean_shortest_path=float(nx.average_shortest_path_length(g)) if n > 1 and nx.is_connected(g) else np.nan,
        degree_gini=_gini(np.array(list(deg.values()), dtype=float)))


def _gini(x: np.ndarray) -> float:
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def centralities(g: nx.Graph) -> pd.DataFrame:
    """Degree, betweenness and closeness (distance-weighted) per node. They answer different questions and are reported separately."""
    return pd.DataFrame({
        "degree": pd.Series(dict(g.degree())),
        "betweenness": pd.Series(nx.betweenness_centrality(g, weight="distance")),
        "closeness": pd.Series(nx.closeness_centrality(g, distance="distance"))}).sort_index()


def communities(g: nx.Graph, seed: int = 0) -> dict[str, int]:
    """Greedy-modularity communities of the MST using inverse-distance edge weights (deterministic)."""
    h = nx.Graph()
    h.add_nodes_from(g.nodes)
    for u, v, d in g.edges(data=True):
        h.add_edge(u, v, weight=1.0 / max(d["distance"], 1e-9))
    parts = nx.community.greedy_modularity_communities(h, weight="weight")
    return {n: i for i, comp in enumerate(sorted(parts, key=lambda c: (-len(c), sorted(c)[0]))) for n in comp}


def adjusted_rand_index(a: dict[str, int], b: dict[str, int]) -> float:
    """Adjusted Rand index between two labelings of the same nodes (1 = identical partitions, about 0 = chance agreement)."""
    keys = sorted(set(a) & set(b))
    la = np.array([a[k] for k in keys])
    lb = np.array([b[k] for k in keys])
    n = len(keys)
    if n < 2:
        return np.nan
    table = pd.crosstab(la, lb).values
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(table).sum()
    sum_a = comb(table.sum(1)).sum()
    sum_b = comb(table.sum(0)).sum()
    expected = sum_a * sum_b / comb(n)
    maxi = 0.5 * (sum_a + sum_b)
    return float((sum_ij - expected) / (maxi - expected)) if maxi != expected else 1.0


def modularity_of_labels(g: nx.Graph, labels: dict[str, int]) -> float:
    """Weighted modularity of an arbitrary partition (for example GICS sectors) on the MST, with inverse-distance weights."""
    h = nx.Graph()
    h.add_nodes_from(g.nodes)
    for u, v, d in g.edges(data=True):
        h.add_edge(u, v, weight=1.0 / max(d["distance"], 1e-9))
    groups: dict[int, set] = {}
    for n, lab in labels.items():
        if n in h:
            groups.setdefault(lab, set()).add(n)
    return float(nx.community.modularity(h, list(groups.values()), weight="weight"))


def edge_set(g: nx.Graph) -> set[frozenset]:
    return {frozenset(e) for e in g.edges}


def edge_jaccard(g1: nx.Graph, g2: nx.Graph) -> float:
    """Jaccard similarity of the edge sets of two trees on their common nodes (edge persistence between windows)."""
    common = set(g1.nodes) & set(g2.nodes)
    e1 = {e for e in edge_set(g1) if e <= common}
    e2 = {e for e in edge_set(g2) if e <= common}
    union = e1 | e2
    return len(e1 & e2) / len(union) if union else np.nan


def rolling_networks(returns: pd.DataFrame, rebalance_dates, lookback: int, universe_fn, min_valid_share: float = 0.95):
    """Yield ``(date, corr, graph)`` using only data up to each date. ``universe_fn(date)`` returns the eligible tickers."""
    for d in rebalance_dates:
        names = universe_fn(d)
        if len(names) < 5:
            continue
        win = returns.loc[:d, names].tail(lookback)
        win = win.dropna(axis=1, thresh=int(np.ceil(min_valid_share * lookback)))
        win = win.dropna(axis=0, how="any")        # common observation space: complete-case rows keep the correlation matrix positive semidefinite
        if len(win) < int(np.ceil(min_valid_share * lookback)):
            continue
        c = correlation(win).dropna(axis=0, how="any").dropna(axis=1, how="any")
        c = c.loc[sorted(c.index), sorted(c.columns)]
        if len(c) < 5:
            continue
        yield d, c, mst_graph(distance(c))
