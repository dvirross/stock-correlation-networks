"""Statistical reliability of an estimated network: joint moving-block bootstrap, edge inclusion, community stability,
window-length sensitivity and a labelled synthetic recoverability experiment.

Edge inclusion frequency is algorithmic stability under the chosen resampling scheme. It is not the probability that a link is
"true" or causal, and it is not a hypothesis-test p-value.
"""
from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from . import networks as nw


def block_indices(n_obs: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices: the same dates are drawn for every stock (cross-sectional dependence preserved)."""
    k = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs - block + 1, size=k)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n_obs]


def bootstrap_graphs(window: pd.DataFrame, graph_fn: Callable[[pd.DataFrame], nx.Graph], corr_fn: Callable[[pd.DataFrame], pd.DataFrame],
                     n_boot: int = 200, block: int = 21, seed: int = 0):
    """Yield one graph per bootstrap draw. ``corr_fn`` and ``graph_fn`` are the same estimator and filter as the point estimate."""
    rng = np.random.default_rng(seed)
    x = window.to_numpy()
    for _ in range(n_boot):
        idx = block_indices(len(x), block, rng)
        w = pd.DataFrame(x[idx], columns=window.columns)
        yield graph_fn(corr_fn(w))


def edge_inclusion(base: nx.Graph, boot_graphs) -> pd.DataFrame:
    """Share of bootstrap graphs containing each base-graph edge."""
    counts = {frozenset(e): 0 for e in base.edges}
    n = 0
    for g in boot_graphs:
        n += 1
        es = {frozenset(e) for e in g.edges}
        for e in counts:
            counts[e] += e in es
    return pd.DataFrame([dict(u=sorted(e)[0], v=sorted(e)[1], inclusion=c / n) for e, c in counts.items()])


def community_stability(base_labels: dict, boot_graphs, comm_fn: Callable[[nx.Graph], dict]) -> list[float]:
    """Adjusted Rand index between the base partition and each bootstrap partition (1 = same grouping)."""
    return [nw.adjusted_rand_index(base_labels, comm_fn(g)) for g in boot_graphs]


def window_length_overlap(returns: pd.DataFrame, date, names, lengths=(126, 252, 504), corr_fn=nw.correlation) -> pd.DataFrame:
    """Edge-set Jaccard of MSTs estimated from different lookbacks ending at the same date on the same stocks."""
    gs = {}
    for L in lengths:
        w = returns.loc[:date, names].tail(L)
        gs[L] = nw.mst_graph(nw.distance(corr_fn(w)))
    rows = [{"a": a, "b": b, "jaccard": nw.edge_jaccard(gs[a], gs[b])} for a in lengths for b in lengths]
    return pd.DataFrame(rows).pivot(index="a", columns="b", values="jaccard")


def planted_block_recovery(n_stocks: int, n_blocks: int, within: float, between: float, t_values, n_rep: int = 50, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC MECHANISM EXPERIMENT (not market data): returns drawn from a Gaussian block-correlation model with known sectors.
    For each sample length T it reports how well the MST sector structure and the true partition are recovered. Used only to show how
    estimation noise limits graph recovery as a function of T for the stated N; the correlation levels are inputs, not findings."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_blocks), int(np.ceil(n_stocks / n_blocks)))[:n_stocks]
    c = np.where(labels[:, None] == labels[None, :], within, between)
    np.fill_diagonal(c, 1.0)
    chol = np.linalg.cholesky(c)
    names = [f"S{i:03d}" for i in range(n_stocks)]
    truth = dict(zip(names, labels.tolist()))
    rows = []
    for T in t_values:
        for r in range(n_rep):
            x = rng.standard_normal((T, n_stocks)) @ chol.T
            corr = nw.correlation(pd.DataFrame(x, columns=names))
            g = nw.mst_graph(nw.distance(corr))
            same = np.mean([truth[u] == truth[v] for u, v in g.edges])
            ari = nw.adjusted_rand_index(nw.communities(g), truth)
            rows.append(dict(T=T, rep=r, same_block_edge_share=same, ari=ari))
    return pd.DataFrame(rows)
