"""Steiner-tree tests use hand-made graphs and fixed-seed random weights (TEST FIXTURES ONLY, not market data)."""
import itertools

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stocknet import steiner as st


def _brute_force(g, terminals):
    others = [n for n in g.nodes if n not in set(terminals)]
    best = np.inf
    for r in range(len(others) + 1):
        for extra in itertools.combinations(others, r):
            h = g.subgraph(list(terminals) + list(extra))
            if nx.is_connected(h):
                t = nx.minimum_spanning_tree(h, weight="distance")
                best = min(best, sum(d["distance"] for *_, d in t.edges(data=True)))
    return best


def test_star_uses_the_steiner_point():
    g = nx.Graph()
    for t in "ABC":
        g.add_edge("hub", t, distance=1.0)
        for u in "ABC":
            if t < u:
                g.add_edge(t, u, distance=1.9)
    tree = st.dreyfus_wagner(g, ["A", "B", "C"])
    s = st.summarize_tree(tree, ["A", "B", "C"])
    assert s["cost"] == pytest.approx(3.0) and s["intermediate_nodes"] == ["hub"]      # 3 < 3.8 (terminals only)


def test_tree_steiner_prunes_non_terminal_leaves_and_is_exact_on_trees():
    t = nx.Graph()
    t.add_weighted_edges_from([("a", "b", 1), ("b", "c", 1), ("c", "d", 1), ("b", "x", 1), ("d", "y", 1)], weight="distance")
    sub = st.tree_steiner(t, ["a", "c"])
    assert set(sub.nodes) == {"a", "b", "c"}
    with pytest.raises(ValueError):
        st.tree_steiner(nx.cycle_graph(4), [0, 1])


def test_dreyfus_wagner_matches_brute_force_on_random_graphs():
    for seed in range(6):
        rng = np.random.default_rng(seed)
        g = nx.gnp_random_graph(9, 0.6, seed=seed)
        if not nx.is_connected(g):
            continue
        for u, v in g.edges:
            g[u][v]["distance"] = float(rng.uniform(0.5, 2.0))
        terms = list(rng.choice(9, size=4, replace=False))
        tree = st.dreyfus_wagner(g, terms)
        cost = st.summarize_tree(tree, terms)["cost"]
        assert nx.is_tree(tree) and set(terms) <= set(tree.nodes)
        assert cost == pytest.approx(_brute_force(g, terms), rel=1e-9)


def test_steiner_on_complete_metric_graph_never_exceeds_terminal_mst_and_is_at_least_mst_subtree_bound():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(25, 3))
    d = pd.DataFrame(np.linalg.norm(x[:, None] - x[None], axis=2), index=[f"s{i}" for i in range(25)], columns=[f"s{i}" for i in range(25)])
    terms = ["s0", "s3", "s7", "s11", "s19"]
    steiner = st.summarize_tree(st.dreyfus_wagner(d, terms), terms)["cost"]
    tmst = st.summarize_tree(st.terminal_mst(d, terms), terms)["cost"]
    full = nx.from_pandas_adjacency(d)
    for u, v in full.edges:
        full[u][v]["distance"] = full[u][v]["weight"]
    mst = nx.minimum_spanning_tree(full, weight="distance")
    in_mst = st.summarize_tree(st.tree_steiner(mst, terms), terms)["cost"]
    assert steiner <= tmst + 1e-12 and steiner <= in_mst + 1e-12                     # Steiner on the larger edge set cannot cost more
    assert tmst / steiner < 2.0                                                      # Steiner ratio bound (<= 2 for any metric)


def test_requires_connected_terminals():
    g = nx.Graph()
    g.add_edge("a", "b", distance=1.0)
    g.add_edge("c", "d", distance=1.0)
    with pytest.raises(ValueError):
        st.dreyfus_wagner(g, ["a", "c"])


def test_single_hub_beats_terminal_only_tree_on_a_psd_correlation_example():
    """QA toy case: three terminals with pairwise correlation 0.4, each correlated 0.75 with an optional hub (valid PSD matrix).
    Terminal-only MST costs 2*sqrt(1.2); the hub star costs 3*sqrt(0.5), which is lower. The exact solver must find the hub."""
    names = ["T1", "T2", "T3", "H"]
    corr = np.array([[1, .4, .4, .75], [.4, 1, .4, .75], [.4, .4, 1, .75], [.75, .75, .75, 1]])
    assert np.linalg.eigvalsh(corr).min() > 0
    d = pd.DataFrame(np.sqrt(2 * (1 - corr)), index=names, columns=names)
    tree = st.dreyfus_wagner(d, ["T1", "T2", "T3"])
    s = st.summarize_tree(tree, ["T1", "T2", "T3"])
    assert s["intermediate_nodes"] == ["H"]
    assert s["cost"] == pytest.approx(3 * np.sqrt(0.5))
    assert st.summarize_tree(st.terminal_mst(d, ["T1", "T2", "T3"]), ["T1", "T2", "T3"])["cost"] == pytest.approx(2 * np.sqrt(1.2))
    assert s["cost"] < 2 * np.sqrt(1.2)
