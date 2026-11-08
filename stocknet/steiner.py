"""Steiner trees on correlation-distance graphs.

Given a small set of *terminal* stocks, a Steiner tree is the cheapest tree that connects them, possibly passing through extra
(non-terminal, "intermediate") stocks. Cost is the sum of correlation distances d = sqrt(2 (1 - rho)) along the tree edges.
This is a geometric summary of one estimated distance matrix. It carries no causal, transmission or investment meaning.

* ``tree_steiner``: exact minimum connecting subtree of a tree (prune non-terminal leaves). On an MST this is the Steiner tree
  *within the MST*, because a tree has a unique path between any two nodes.
* ``dreyfus_wagner``: exact Steiner tree on any connected weighted graph (Dreyfus and Wagner, 1971); exponential in the number of
  terminals (fine for <= ~12) and polynomial in the number of nodes.
* ``terminal_mst``: MST over the terminals alone, with no intermediate nodes allowed (metric closure of the graph); an upper bound.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


def tree_steiner(tree: nx.Graph, terminals) -> nx.Graph:
    """Minimal connecting subtree of ``tree`` containing ``terminals`` (exact)."""
    terms = set(terminals)
    if not terms <= set(tree.nodes):
        raise ValueError("terminals must be nodes of the tree")
    if not nx.is_tree(tree):
        raise ValueError("tree_steiner requires a tree (connected, acyclic)")
    h = tree.copy()
    leaves = [n for n in h.nodes if h.degree(n) <= 1 and n not in terms]
    while leaves:
        for n in leaves:
            if n in h:
                h.remove_node(n)
        leaves = [n for n in h.nodes if h.degree(n) <= 1 and n not in terms]
    return h


def _matrix(graph_or_dist, weight="distance"):
    if isinstance(graph_or_dist, nx.Graph):
        nodes = list(graph_or_dist.nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        w = np.full((len(nodes), len(nodes)), np.inf)
        np.fill_diagonal(w, 0.0)
        for u, v, d in graph_or_dist.edges(data=True):
            w[idx[u], idx[v]] = w[idx[v], idx[u]] = min(w[idx[u], idx[v]], d[weight])
        return nodes, w
    nodes = list(graph_or_dist.index)
    w = graph_or_dist.to_numpy(dtype=float).copy()
    np.fill_diagonal(w, 0.0)
    return nodes, w


def dreyfus_wagner(graph_or_dist, terminals, weight: str = "distance") -> nx.Graph:
    """Exact Steiner tree. ``graph_or_dist`` is a weighted ``networkx.Graph`` (edge attribute ``weight``) or a symmetric distance
    DataFrame (a complete graph). Returns a tree as a ``networkx.Graph`` whose edges carry ``distance``."""
    nodes, w = _matrix(graph_or_dist, weight)
    n = len(nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    terms = [idx[t] for t in terminals]
    if len(set(terms)) != len(terms) or len(terms) < 2:
        raise ValueError("need at least two distinct terminals")
    dist, pred = shortest_path(_dense_to_csr(w), directed=False, return_predecessors=True)
    if np.isinf(dist[np.ix_(terms, terms)]).any():
        raise ValueError("terminals are not connected in this graph")
    root, rest = terms[-1], terms[:-1]
    k = len(rest)
    full = (1 << k) - 1
    dp = np.full((1 << k, n), np.inf)
    sub_choice = np.zeros((1 << k, n), dtype=np.int64)
    via = np.zeros((1 << k, n), dtype=np.int64)
    for i, t in enumerate(rest):
        dp[1 << i] = dist[t]
        via[1 << i] = t
    for mask in range(1, 1 << k):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        best = np.full(n, np.inf)
        arg = np.zeros(n, dtype=np.int64)
        sub = (mask - 1) & mask
        while sub:
            if sub & low:                      # each unordered split once
                other = mask ^ sub
                if other:
                    cand = dp[sub] + dp[other]
                    better = cand < best
                    best = np.where(better, cand, best)
                    arg = np.where(better, sub, arg)
            sub = (sub - 1) & mask
        sub_choice[mask] = arg
        # dp[mask][v] = min_u best[u] + dist[u][v]
        tot = best[:, None] + dist
        u = tot.argmin(axis=0)
        dp[mask] = tot[u, np.arange(n)]
        via[mask] = u
    edges = set()

    def add_path(a, b):
        while b != a:
            p = pred[a, b]
            edges.add(tuple(sorted((int(p), int(b)))))
            b = p

    def build(mask, v):
        if mask & (mask - 1) == 0:
            add_path(int(via[mask][v]), v)
            return
        u = int(via[mask][v])
        add_path(u, v)
        s = int(sub_choice[mask][u])
        build(s, u)
        build(mask ^ s, u)

    build(full, root)
    g = nx.Graph()
    for a, b in edges:
        g.add_edge(nodes[a], nodes[b], distance=float(w[a, b]))
    h = nx.minimum_spanning_tree(g, weight="distance")      # remove any cycle created by tied shortest paths
    return tree_steiner(h, [nodes[t] for t in terms])


def _dense_to_csr(w):
    from scipy.sparse import csr_matrix
    m = np.where(np.isinf(w), 0.0, w)
    np.fill_diagonal(m, 0.0)
    return csr_matrix(m)


def terminal_mst(dist, terminals) -> nx.Graph:
    """MST over the terminals only, on the complete graph with the given distance DataFrame (no intermediate nodes)."""
    sub = dist.loc[list(terminals), list(terminals)]
    g = nx.Graph()
    for a, b in itertools.combinations(sub.index, 2):
        g.add_edge(a, b, distance=float(sub.loc[a, b]))
    return nx.minimum_spanning_tree(g, weight="distance")


def summarize_tree(tree: nx.Graph, terminals) -> dict:
    terms = set(terminals)
    inter = sorted(set(tree.nodes) - terms)
    return dict(cost=float(sum(d["distance"] for _, _, d in tree.edges(data=True))), edges=tree.number_of_edges(), nodes=tree.number_of_nodes(),
                intermediates=len(inter), intermediate_nodes=inter)
