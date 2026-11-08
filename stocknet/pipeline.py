"""Analysis steps shared by the driver script and the notebook. All inputs are the real panel; nothing here simulates market data
except ``recoverability`` (explicitly labelled synthetic mechanism test in reliability.planted_block_recovery)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from . import dependence as dp
from . import filters as fl
from . import networks as nw
from . import reliability as rl

ESTIMATOR_FNS = dp.ESTIMATORS


def load_panel(prices_csv, sectors_csv):
    px = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    sec = pd.read_csv(sectors_csv).set_index("ticker").sector.to_dict()
    returns = px.pct_change(fill_method=None).iloc[1:]
    return px, returns, sec


def _codes(labels: dict) -> dict:
    ids = {v: i for i, v in enumerate(sorted(set(labels.values())))}
    return {k: ids[v] for k, v in labels.items()}


def window_ending(returns: pd.DataFrame, end, length: int) -> pd.DataFrame:
    return returns.loc[:end].tail(length)


def build_graphs(corr: pd.DataFrame) -> dict[str, nx.Graph]:
    n = len(corr)
    return {"MST": fl.mst_from_corr(corr), "PMFG": fl.pmfg(corr), "Fixed density (3N-6 edges)": fl.fixed_density_graph(corr, 3 * (n - 2))}


def structure_table(corr: pd.DataFrame, sectors: dict, graphs: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """One row per graph type on the identical correlation matrix."""
    graphs = graphs or build_graphs(corr)
    rows = []
    for name, g in graphs.items():
        s = fl.graph_stats(g, sectors)
        p = fl.permutation_pvalue_same_sector(g, sectors, n_perm=1000, seed=seed)
        comm = nw.communities(g)
        s.update(graph=name, same_sector_null=p["null_mean"], same_sector_perm_p=p["p_value"],
                 community_count=len(set(comm.values())),
                 ari_vs_sector=nw.adjusted_rand_index(comm, _codes({k: v for k, v in sectors.items() if k in comm})),
                 modularity_sector=nw.modularity_of_labels(g, _codes({k: v for k, v in sectors.items() if k in g})))
        rows.append(s)
    cols = ["graph", "edges", "mean_degree", "max_degree", "triangles", "transitivity", "components", "giant_share", "mean_path_giant",
            "same_sector_edge_share", "same_sector_null", "same_sector_perm_p", "sector_assortativity", "community_count", "ari_vs_sector", "modularity_sector"]
    return pd.DataFrame(rows)[cols].set_index("graph")


def full_matrix_summary(corr: pd.DataFrame, sectors: dict) -> dict:
    names = list(corr.index)
    c = corr.to_numpy()
    iu = np.triu_indices(len(names), 1)
    lab = np.array([sectors[n] for n in names])
    same = lab[iu[0]] == lab[iu[1]]
    eig = np.linalg.eigvalsh(c)
    return dict(n=len(names), pairs=len(iu[0]), mean_corr=float(c[iu].mean()), mean_within_sector=float(c[iu][same].mean()), mean_between_sector=float(c[iu][~same].mean()),
                first_eigen_share=float(eig[-1] / eig.sum()), second_eigen_share=float(eig[-2] / eig.sum()))


def estimator_comparison(window: pd.DataFrame, sectors: dict, include_kendall: bool = True):
    """Same window, four dependence estimates; MST/PMFG/fixed-density overlap between estimators and sector alignment."""
    corrs = {}
    for name, fn in ESTIMATOR_FNS.items():
        if name == "kendall" and not include_kendall:
            continue
        c = fn(window)
        corrs[name] = c.loc[sorted(c.index), sorted(c.columns)]
    names = list(corrs)
    iu = np.triu_indices(len(corrs[names[0]]), 1)
    corr_agreement = pd.DataFrame({a: {b: float(np.corrcoef(corrs[a].to_numpy()[iu], corrs[b].to_numpy()[iu])[0, 1]) for b in names} for a in names})
    graphs = {n: build_graphs(c) for n, c in corrs.items()}
    overlap = {}
    for gname in next(iter(graphs.values())):
        overlap[gname] = pd.DataFrame({a: {b: nw.edge_jaccard(graphs[a][gname], graphs[b][gname]) for b in names} for a in names})
    rows = []
    for n in names:
        for gname, g in graphs[n].items():
            s = fl.graph_stats(g, sectors)
            rows.append(dict(estimator=n, graph=gname, same_sector_edge_share=s["same_sector_edge_share"], max_degree=s["max_degree"], triangles=s["triangles"],
                             edges=s["edges"], components=s["components"]))
    return corrs, graphs, corr_agreement, overlap, pd.DataFrame(rows)


def regime_table(returns: pd.DataFrame, sectors: dict, ends: dict, length: int = 252) -> pd.DataFrame:
    rows = []
    for label, end in ends.items():
        w = window_ending(returns, end, length)
        c = nw.correlation(w)
        c = c.loc[sorted(c.index), sorted(c.columns)]
        g = build_graphs(c)
        full = full_matrix_summary(c, sectors)
        t = structure_table(c, sectors, g)
        mst_len = np.mean([d["distance"] for _, _, d in g["MST"].edges(data=True)])
        rows.append(dict(window=label, end=str(w.index[-1].date()), mean_corr=full["mean_corr"], first_eigen_share=full["first_eigen_share"],
                         mean_within_sector=full["mean_within_sector"], mean_between_sector=full["mean_between_sector"], mst_mean_edge=mst_len,
                         mst_max_degree=int(t.loc["MST", "max_degree"]), mst_same_sector=t.loc["MST", "same_sector_edge_share"],
                         pmfg_same_sector=t.loc["PMFG", "same_sector_edge_share"], pmfg_triangles=int(t.loc["PMFG", "triangles"]),
                         fixed_components=int(t.loc["Fixed density (3N-6 edges)", "components"]), mst_ari_vs_sector=t.loc["MST", "ari_vs_sector"],
                         annualised_market_vol=float(w.mean(axis=1).std(ddof=1) * np.sqrt(252))))
    return pd.DataFrame(rows).set_index("window")


def rolling_structure(returns: pd.DataFrame, sectors: dict, length: int = 252, step: int = 21, start="2004-01-01") -> tuple[pd.DataFrame, dict]:
    """Monthly rolling MST metrics and the MSTs themselves (for persistence)."""
    idx = returns.index
    ends = [idx[i] for i in range(length, len(idx), step) if idx[i] >= pd.Timestamp(start)]
    rows, graphs = [], {}
    for e in ends:
        w = window_ending(returns, e, length)
        c = nw.correlation(w)
        c = c.loc[sorted(c.index), sorted(c.columns)]
        g = fl.mst_from_corr(c)
        graphs[e] = g
        full = full_matrix_summary(c, sectors)
        s = fl.graph_stats(g, sectors)
        rows.append(dict(date=e, mean_corr=full["mean_corr"], first_eigen_share=full["first_eigen_share"], mst_mean_edge=np.mean([d["distance"] for _, _, d in g.edges(data=True)]),
                         max_degree=s["max_degree"], same_sector_edge_share=s["same_sector_edge_share"], market_vol=float(w.mean(axis=1).std(ddof=1) * np.sqrt(252))))
    df = pd.DataFrame(rows).set_index("date")
    dates = list(graphs)
    df["jaccard_prev"] = [np.nan] + [nw.edge_jaccard(graphs[a], graphs[b]) for a, b in zip(dates[:-1], dates[1:])]
    return df, graphs


def persistence_vs_gap(graphs: dict, step: int = 21, length: int = 252, gaps_months=(1, 3, 6, 12, 24, 60)) -> pd.DataFrame:
    """Edge Jaccard between MSTs whose windows are ``gap`` months apart. Gaps of 12 months or more use non-overlapping windows (length 252 days)."""
    dates = list(graphs)
    rows = []
    for gm in gaps_months:
        lag = gm  # monthly step == 21 trading days ~ 1 month
        vals = [nw.edge_jaccard(graphs[dates[i]], graphs[dates[i + lag]]) for i in range(len(dates) - lag)]
        rows.append(dict(gap_months=gm, median=float(np.median(vals)), p10=float(np.quantile(vals, .1)), p90=float(np.quantile(vals, .9)), n=len(vals),
                         overlapping=gm * step < length))
    return pd.DataFrame(rows)
