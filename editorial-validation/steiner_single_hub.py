"""Single-auxiliary-stock Steiner diagnostic for a complete correlation-distance graph.

Exploratory graph objective only; no causal, investable, or point-in-time claims.
Example:
    python research/stock-networks/editorial-validation/steiner_single_hub.py --self-test
    python research/stock-networks/editorial-validation/steiner_single_hub.py \
      --prices-csv research/stock-networks/data/panel_adjusted_close.csv \
      --end 2024-12-30 --window 252 --terminals AAPL MSFT AMZN GOOGL

Only the terminal set must be prespecified; an auxiliary candidate may be any
other stock in the same fixed observation panel. This searches the optimal tree
using AT MOST ONE extra stock, not the unrestricted NP-hard Steiner optimum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np


def graph_from_correlation(names: list[str], corr: np.ndarray) -> nx.Graph:
    """Complete undirected graph with Mantegna distances; reject non-PSD input."""
    c = np.asarray(corr, dtype=float)
    n = len(names)
    if c.shape != (n, n) or not np.isfinite(c).all():
        raise ValueError("Correlation matrix has wrong shape or non-finite entries.")
    if not np.allclose(c, c.T, atol=1e-9) or not np.allclose(np.diag(c), 1, atol=1e-9):
        raise ValueError("Matrix is not a symmetric correlation matrix.")
    if np.min(np.linalg.eigvalsh(c)) < -1e-8:
        raise ValueError("Correlation matrix is not positive semidefinite.")
    if np.any(np.abs(c) > 1 + 1e-9):
        raise ValueError("Correlation outside [-1,1].")
    g = nx.Graph()
    g.add_nodes_from(names)
    for i in range(n):
        for j in range(i + 1, n):
            rho = float(np.clip(c[i, j], -1, 1))
            g.add_edge(names[i], names[j], weight=float(np.sqrt(2 * (1 - rho))), rho=rho)
    return g


def weight(g: nx.Graph) -> float:
    return float(sum(data["weight"] for _, _, data in g.edges(data=True)))


def prune_nonterminal_leaves(g: nx.Graph, terminals: set[str]) -> nx.Graph:
    t = g.copy()
    while True:
        dead = [node for node in t if node not in terminals and t.degree(node) <= 1]
        if not dead:
            return t
        t.remove_nodes_from(dead)


def compare_one_hub(g: nx.Graph, terminals: list[str]) -> dict:
    """Exact optimum among trees with <=1 auxiliary node, for complete weighted G."""
    s = set(terminals)
    if len(s) < 2 or len(s) != len(terminals) or not s.issubset(g):
        raise ValueError("Require at least two distinct terminals present in graph.")
    if nx.number_connected_components(g) != 1:
        raise ValueError("This diagnostic expects the complete connected graph.")

    terminal_only = nx.minimum_spanning_tree(g.subgraph(s), weight="weight")
    best = terminal_only
    best_hub = None
    baseline_cost = weight(terminal_only)
    best_cost = baseline_cost
    for hub in sorted(set(g) - s):
        candidate = nx.minimum_spanning_tree(g.subgraph(s | {hub}), weight="weight")
        candidate = prune_nonterminal_leaves(candidate, s)
        c = weight(candidate)
        if c < best_cost - 1e-10:
            best, best_cost, best_hub = candidate, c, hub

    full_mst = nx.minimum_spanning_tree(g, weight="weight")
    backbone = prune_nonterminal_leaves(full_mst, s)
    # Paths along the market-wide MST are not generally the optimum on the full G.
    result = {
        "terminals": terminals, "market_nodes": g.number_of_nodes(),
        "terminal_only_mst_cost": baseline_cost,
        "market_mst_terminal_subtree_cost": weight(backbone),
        "best_at_most_one_hub_cost": best_cost,
        "best_single_hub": best_hub,
        "gain_vs_terminal_mst_pct": 100 * (baseline_cost - best_cost) / baseline_cost,
        "gain_vs_global_mst_subtree_pct": 100 * (weight(backbone) - best_cost) / weight(backbone),
        "auxiliary_count": len(set(best) - s),
        "edges": [{"u": min(u, v), "v": max(u, v), "distance": float(d["weight"]),
                   "correlation": float(d["rho"])}
                  for u, v, d in best.edges(data=True)],
        "scope": "Descriptive best tree with at most one extra observed stock (NOT full Steiner optimum).",
    }
    if result["best_at_most_one_hub_cost"] > result["terminal_only_mst_cost"] + 1e-8:
        raise AssertionError("One-hub optimum exceeds terminal-only baseline.")
    return result


def self_test() -> None:
    """PSD four-stock counterexample to shortest-path-based Steiner approximations."""
    names = ["A", "B", "C", "H"]
    c = np.full((4, 4), 0.4)
    np.fill_diagonal(c, 1)
    c[:3, 3] = c[3, :3] = 0.75
    assert np.linalg.eigvalsh(c).min() > 0
    g = graph_from_correlation(names, c)
    report = compare_one_hub(g, names[:3])
    expected_term = 2 * np.sqrt(1.2)
    expected_hub = 3 * np.sqrt(0.5)
    assert np.isclose(report["terminal_only_mst_cost"], expected_term)
    assert np.isclose(report["best_at_most_one_hub_cost"], expected_hub)
    assert report["best_single_hub"] == "H"
    # Shortest paths in a Euclidean metric complete graph are direct edges.
    # Kou's standard metric-closure approximation therefore misses this hub.
    for method in ("kou", "mehlhorn"):
        approx = nx.algorithms.approximation.steinertree.steiner_tree(
            g, names[:3], weight="weight", method=method)
        assert np.isclose(weight(approx), expected_term), method
        assert "H" not in approx, method
    print("PASS: PSD metric toy; best single hub improves cost; Kou/Mehlhorn miss hub.")
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prices-csv", type=Path)
    ap.add_argument("--end", default="2024-12-30")
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--terminals", nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.prices_csv is None or args.terminals is None:
        ap.error("Specify --self-test, or both --prices-csv and --terminals.")
    import pandas as pd
    prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True).sort_index()
    daily = prices.pct_change(fill_method=None).iloc[1:]
    window = daily.loc[:args.end].tail(args.window)
    if len(window) != args.window or window.isna().any().any():
        raise ValueError("Incomplete or missing-return panel; do not impute or silently drop stocks.")
    names = list(window.columns)
    if any(t not in names for t in args.terminals):
        raise ValueError("A terminal is missing from the panel.")
    graph = graph_from_correlation(names, window.corr().to_numpy())
    result = compare_one_hub(graph, args.terminals)
    result.update(window_end=str(window.index[-1].date()), window_days=args.window,
                  price_file=str(args.prices_csv),
                  data_note="Historically selected survivor panel; no out-of-sample performance inference.")
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
