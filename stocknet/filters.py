"""Graph filters on a correlation matrix: MST, PMFG, fixed-density (top-M edges) and the full weighted graph, plus graph statistics.

Properties imposed by construction (not findings): MST is a tree (N-1 edges, no cycles, zero triangles/clustering, every edge a bridge);
PMFG is planar with 3N-6 edges (N>=3) and contains the MST for a consistent edge order; the fixed-density graph can be disconnected.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from . import networks as nw


def _ordered_edges(corr: pd.DataFrame) -> list[tuple[float, str, str]]:
    names = list(corr.index)
    c = corr.to_numpy()
    iu = np.triu_indices(len(names), 1)
    order = np.lexsort((iu[1], iu[0], -c[iu]))          # descending correlation, deterministic ties
    return [(float(c[iu[0][k], iu[1][k]]), names[iu[0][k]], names[iu[1][k]]) for k in order]


def fixed_density_graph(corr: pd.DataFrame, n_edges: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(corr.index)
    for rho, u, v in _ordered_edges(corr)[:n_edges]:
        g.add_edge(u, v, rho=rho, distance=float(np.sqrt(max(2 * (1 - rho), 0.0))))
    return g


def pmfg(corr: pd.DataFrame) -> nx.Graph:
    """Planar Maximally Filtered Graph (Tumminello et al. 2005): add edges in descending correlation if the graph stays planar,
    until 3N-6 edges. Greedy by construction (not a globally optimal planar subgraph)."""
    n = len(corr)
    g = nx.Graph()
    g.add_nodes_from(corr.index)
    target = 3 * (n - 2)
    for rho, u, v in _ordered_edges(corr):
        g.add_edge(u, v, rho=rho, distance=float(np.sqrt(max(2 * (1 - rho), 0.0))))
        if not nx.check_planarity(g)[0]:
            g.remove_edge(u, v)
        if g.number_of_edges() >= target:
            break
    return g


def mst_from_corr(corr: pd.DataFrame) -> nx.Graph:
    return nw.mst_graph(nw.distance(corr))


def graph_stats(g: nx.Graph, groups: dict[str, str] | None = None) -> dict:
    """Structure summaries comparable across filters. Sector assortativity uses ``groups`` (ticker -> label) when supplied."""
    n, m = g.number_of_nodes(), g.number_of_edges()
    deg = np.array([d for _, d in g.degree()], dtype=float)
    comps = list(nx.connected_components(g))
    giant = max(comps, key=len)
    out = dict(n=n, edges=m, mean_degree=float(deg.mean()), max_degree=int(deg.max()), degree_gini=nw._gini(deg),
               components=len(comps), giant_share=len(giant) / n,
               triangles=int(sum(nx.triangles(g).values()) / 3), transitivity=float(nx.transitivity(g)),
               avg_clustering=float(nx.average_clustering(g)), max_core=max(nx.core_number(g).values()))
    h = g.subgraph(giant)
    out["mean_path_giant"] = float(nx.average_shortest_path_length(h)) if len(h) > 1 else np.nan
    if groups:
        lab = {k: groups[k] for k in g.nodes if k in groups}
        gg = g.subgraph(lab)
        out["sector_assortativity"] = float(nx.attribute_assortativity_coefficient(nx.relabel_nodes(gg, {k: k for k in gg}), "sector")) if _set_attr(gg, lab) else np.nan
        same = sum(lab[u] == lab[v] for u, v in gg.edges)
        out["same_sector_edge_share"] = same / max(gg.number_of_edges(), 1)
    return out


def _set_attr(g: nx.Graph, lab: dict) -> bool:
    for k in g.nodes:
        g.nodes[k]["sector"] = lab[k]
    return g.number_of_edges() > 0


def expected_same_group_share(groups: dict[str, str], nodes) -> float:
    """Share of same-sector pairs among all pairs of ``nodes``: what a label-blind graph would show on average."""
    labs = pd.Series({k: groups[k] for k in nodes if k in groups})
    counts = labs.value_counts().to_numpy()
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def permutation_pvalue_same_sector(g: nx.Graph, groups: dict[str, str], n_perm: int = 2000, seed: int = 0) -> dict:
    """Edge-endpoint same-sector share versus a null that permutes sector labels across stocks (graph fixed)."""
    nodes = [k for k in g.nodes if k in groups]
    edges = [(u, v) for u, v in g.edges if u in groups and v in groups]
    lab = np.array([groups[k] for k in nodes])
    idx = {k: i for i, k in enumerate(nodes)}
    e = np.array([(idx[u], idx[v]) for u, v in edges])
    obs = float(np.mean(lab[e[:, 0]] == lab[e[:, 1]]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(lab)
        null[i] = np.mean(p[e[:, 0]] == p[e[:, 1]])
    return dict(observed=obs, null_mean=float(null.mean()), null_p95=float(np.quantile(null, 0.95)), p_value=float((1 + (null >= obs).sum()) / (1 + n_perm)))
