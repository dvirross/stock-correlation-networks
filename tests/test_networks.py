import itertools

import numpy as np
import pandas as pd
import pytest

from stocknet import networks as nw


def test_distance_is_a_metric(factor_returns):
    d = nw.distance(nw.correlation(factor_returns))
    assert np.allclose(d.values, d.values.T) and np.allclose(np.diag(d.values), 0)
    v = d.values
    n = len(v)
    for i, j, k in itertools.combinations(range(n), 3):
        assert v[i, j] <= v[i, k] + v[k, j] + 1e-9


def test_distance_known_values():
    c = pd.DataFrame([[1, 0.5, -1], [0.5, 1, 0], [-1, 0, 1]], index=list("abc"), columns=list("abc"))
    d = nw.distance(c)
    assert d.loc["a", "b"] == pytest.approx(1.0)               # sqrt(2 * 0.5)
    assert d.loc["a", "c"] == pytest.approx(2.0)               # perfect negative correlation: maximum distance


def test_mst_matches_bruteforce_on_small_case(factor_returns):
    cols = list(factor_returns.columns[:6])
    d = nw.distance(nw.correlation(factor_returns[cols]))
    tree = nw.mst_edges(d)
    assert len(tree) == len(cols) - 1
    best = None
    pairs = list(itertools.combinations(cols, 2))
    for edges in itertools.combinations(pairs, len(cols) - 1):
        import networkx as nx
        g = nx.Graph(edges)
        if g.number_of_nodes() == len(cols) and nx.is_tree(g):
            w = sum(d.loc[u, v] for u, v in edges)
            best = w if best is None else min(best, w)
    assert tree.distance.sum() == pytest.approx(best)


def test_metrics_and_communities_recover_blocks(factor_returns):
    c = nw.correlation(factor_returns)
    g = nw.mst_graph(nw.distance(c))
    m = nw.network_metrics(g, c)
    assert m["n"] == 12 and 0 < m["first_eigen_share"] < 1 and m["mst_total_length"] > 0
    comm = nw.communities(g)
    truth = {n: int(n[1]) for n in c.index}
    assert nw.adjusted_rand_index(comm, truth) > 0.7           # three planted blocks are recovered
    assert nw.modularity_of_labels(g, truth) > 0.2


def test_adjusted_rand_index_properties():
    a = {"x": 0, "y": 0, "z": 1, "w": 1}
    assert nw.adjusted_rand_index(a, {"x": 5, "y": 5, "z": 9, "w": 9}) == pytest.approx(1.0)   # label names are irrelevant
    assert nw.adjusted_rand_index(a, {"x": 0, "y": 1, "z": 0, "w": 1}) < 0.1


def test_edge_jaccard():
    import networkx as nx
    g1 = nx.Graph([("a", "b"), ("b", "c")])
    g2 = nx.Graph([("a", "b"), ("a", "c")])
    assert nw.edge_jaccard(g1, g1) == 1.0
    assert nw.edge_jaccard(g1, g2) == pytest.approx(1 / 3)


def test_rolling_networks_use_only_past_data(factor_returns):
    cols = list(factor_returns.columns)
    t = factor_returns.index[300]
    base = list(nw.rolling_networks(factor_returns, [t], 100, lambda d: cols))[0]
    changed = factor_returns.copy()
    changed.loc[changed.index > t] = 0.123                      # destroy everything after t
    after = list(nw.rolling_networks(changed, [t], 100, lambda d: cols))[0]
    assert np.allclose(base[1].values, after[1].values) and nw.edge_jaccard(base[2], after[2]) == 1.0


def test_rolling_networks_use_complete_cases_and_stay_psd(factor_returns):
    from stocknet import networks as nw
    ret = factor_returns.copy()
    rng = np.random.default_rng(3)
    for c in ret.columns:                                              # different missingness per column (fixture), 1% of days
        ret.loc[ret.index[rng.choice(len(ret), 6, replace=False)], c] = np.nan
    dates = [ret.index[400], ret.index[500]]
    out = list(nw.rolling_networks(ret, dates, 200, lambda d: list(ret.columns), min_valid_share=0.8))
    assert out
    for _, c, _g in out:
        assert np.linalg.eigvalsh(c.to_numpy()).min() > -1e-9
