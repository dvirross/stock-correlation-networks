"""Part 5: (a) bootstrap stability vs temporal persistence benchmark; (b) narrowly scoped Steiner-tree experiment.

Terminal set fixed BEFORE any Steiner computation: one widely known large-cap from each of eight sectors present in the panel:
AAPL (IT), JPM (Financials), XOM (Energy), JNJ (Health Care), PG (Staples), CAT (Industrials), HD (Discretionary), SO (Utilities).
Geometric summary only: no causal, transmission or investment meaning."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import pipeline as pl, networks as nw, filters as fl, reliability as rl, steiner as st

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
px, returns, sec = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
sec = {k: v for k, v in sec.items() if k in returns.columns}
TERMINALS = ["AAPL", "JPM", "XOM", "JNJ", "PG", "CAT", "HD", "SO"]
ENDS = {"2008": "2008-12-31", "2017": "2017-12-29", "2024": "2024-12-30"}
BLOCK, SEED = 21, 20241230
t0 = time.time()

def jac(a, b):
    ea = {frozenset(e) for e in a.edges}; eb = {frozenset(e) for e in b.edges}
    return len(ea & eb) / len(ea | eb)

# ---------------- (a) bootstrap stability vs temporal persistence ----------------
rows = []
for lab, end in ENDS.items():
    w = pl.window_ending(returns, end, 252); w = w[sorted(w.columns)]
    base = fl.mst_from_corr(nw.correlation(w))
    boots = list(rl.bootstrap_graphs(w, fl.mst_from_corr, nw.correlation, n_boot=300, block=BLOCK, seed=SEED))
    j_base = [jac(base, b) for b in boots]
    j_pairs = [jac(boots[2 * i], boots[2 * i + 1]) for i in range(150)]           # two independent resamples of the SAME year: "no true change" noise floor
    prev = pl.window_ending(returns[returns.index < w.index[0]], returns.index[returns.index < w.index[0]][-1], 252)
    prev = prev[sorted(prev.columns)]
    j_prev = jac(base, fl.mst_from_corr(nw.correlation(prev)))                       # previous, NON-overlapping 252-day window
    shift = {}
    for k in (1, 21, 63, 126, 252):                                                  # shifted windows: overlap fraction = (252 - k)/252
        wk = pl.window_ending(returns[returns.index <= w.index[-1]], returns.index[returns.index <= w.index[-1]][-1 - k], 252) if k else w
        shift[k] = jac(base, fl.mst_from_corr(nw.correlation(wk[sorted(wk.columns)])))
    rows.append(dict(window_end=end, bootstrap_vs_original_median=float(np.median(j_base)), bootstrap_vs_original_p10=float(np.quantile(j_base, .1)), bootstrap_vs_original_p90=float(np.quantile(j_base, .9)),
                     bootstrap_pair_noise_floor_median=float(np.median(j_pairs)), bootstrap_pair_p10=float(np.quantile(j_pairs, .1)), bootstrap_pair_p90=float(np.quantile(j_pairs, .9)),
                     previous_nonoverlapping_window=j_prev, **{f"shift_{k}d_overlap_{(252 - k) / 252:.3f}": v for k, v in shift.items()}))
    print(lab, rows[-1], round(time.time() - t0), flush=True)
pd.DataFrame(rows).round(4).to_csv(R / "bootstrap_vs_temporal.csv", index=False)

# ---------------- (b) Steiner-tree experiment ----------------
def trees_for(corr, pm, terminals):
    d = nw.distance(corr)
    mst = fl.mst_from_corr(corr)
    out = {}
    out["MST subtree (exact, inside the MST)"] = st.tree_steiner(mst, terminals)
    out["Steiner on PMFG (exact)"] = st.dreyfus_wagner(pm, terminals)
    out["Steiner on full distance graph (exact)"] = st.dreyfus_wagner(d, terminals)
    out["Terminals only (MST of the 8 stocks, no intermediates)"] = st.terminal_mst(d, terminals)
    return out, mst

main_rows, detail = [], {}
for lab, end in ENDS.items():
    w = pl.window_ending(returns, end, 252); w = w[sorted(w.columns)]
    corr = nw.correlation(w)
    pm = fl.pmfg(corr)
    trees, mst = trees_for(corr, pm, TERMINALS)
    base_cost = st.summarize_tree(trees["MST subtree (exact, inside the MST)"], TERMINALS)["cost"]
    for name, t in trees.items():
        s = st.summarize_tree(t, TERMINALS)
        s.update(window_end=end, tree=name, cost_vs_mst_subtree=s["cost"] / base_cost, sector_of_intermediates=";".join(f"{n}:{sec[n]}" for n in s["intermediate_nodes"]),
                 edges_in_mst=sum(frozenset(e) in {frozenset(x) for x in mst.edges} for e in t.edges))
        s["intermediate_nodes"] = ";".join(s["intermediate_nodes"])
        main_rows.append(s)
    detail[lab] = (corr, pm, mst)
    print("steiner", lab, round(time.time() - t0), flush=True)
pd.DataFrame(main_rows).round(4).to_csv(R / "steiner_primary_terminals.csv", index=False)
tbl = pd.DataFrame(main_rows)
print(tbl[["window_end", "tree", "cost", "edges", "intermediates", "cost_vs_mst_subtree", "edges_in_mst", "intermediate_nodes"]].round(3).to_string())

# sensitivity to terminal selection (headline window)
corr, pm, mst = detail["2024"]
d = nw.distance(corr)
rng = np.random.default_rng(5)
names = list(corr.index)
SECS8 = ["Information Technology", "Financials", "Energy", "Health Care", "Consumer Staples", "Industrials", "Consumer Discretionary", "Utilities"]
by_sec = {s: [n for n in names if sec[n] == s] for s in SECS8}
def one(terms):
    a = st.tree_steiner(mst, terms); b = st.dreyfus_wagner(pm, terms); c = st.dreyfus_wagner(d, terms); e = st.terminal_mst(d, terms)
    sa, sb, sc, se = (st.summarize_tree(x, terms) for x in (a, b, c, e))
    return dict(cost_mst=sa["cost"], cost_pmfg=sb["cost"], cost_full=sc["cost"], cost_terminals_only=se["cost"], inter_mst=sa["intermediates"], inter_pmfg=sb["intermediates"], inter_full=sc["intermediates"])
sens = []
for i in range(300):
    sens.append(dict(design="random 8 stocks", **one(list(rng.choice(names, 8, replace=False)))))
for i in range(300):
    sens.append(dict(design="one random stock from each of the 8 named sectors", **one([str(rng.choice(by_sec[s])) for s in SECS8])))
for tdrop in TERMINALS:                                                                    # leave-one-terminal-out around the predetermined set
    sens.append(dict(design=f"predetermined set without {tdrop}", **one([t for t in TERMINALS if t != tdrop])))
sens = pd.DataFrame(sens); sens.round(4).to_csv(R / "steiner_terminal_sensitivity.csv", index=False)
summ = {}
for des, g in sens.groupby("design"):
    if des.startswith("predetermined"): continue
    summ[des] = dict(n=len(g), pmfg_over_mst_median=float((g.cost_pmfg / g.cost_mst).median()), pmfg_over_mst_p10=float((g.cost_pmfg / g.cost_mst).quantile(.1)), pmfg_over_mst_p90=float((g.cost_pmfg / g.cost_mst).quantile(.9)),
                     full_over_mst_median=float((g.cost_full / g.cost_mst).median()), full_over_mst_p10=float((g.cost_full / g.cost_mst).quantile(.1)), full_over_mst_p90=float((g.cost_full / g.cost_mst).quantile(.9)),
                     full_over_terminals_only_median=float((g.cost_full / g.cost_terminals_only).median()),
                     inter_mst_median=float(g.inter_mst.median()), inter_mst_p10=float(g.inter_mst.quantile(.1)), inter_mst_p90=float(g.inter_mst.quantile(.9)),
                     inter_pmfg_median=float(g.inter_pmfg.median()), inter_full_median=float(g.inter_full.median()), inter_full_p10=float(g.inter_full.quantile(.1)), inter_full_p90=float(g.inter_full.quantile(.9)),
                     share_pmfg_equals_mst_cost=float((np.isclose(g.cost_pmfg, g.cost_mst)).mean()))
loo = sens[sens.design.str.startswith("predetermined")]
summ["leave_one_terminal_out"] = loo.round(4).to_dict("records")
(R / "steiner_sensitivity_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
print(json.dumps({k: v for k, v in summ.items() if k != "leave_one_terminal_out"}, indent=1))
print(loo.round(3).to_string())

# bootstrap sensitivity of the primary Steiner trees (MST subtree and full-graph Steiner; PMFG skipped for cost)
w = pl.window_ending(returns, ENDS["2024"], 252); w = w[sorted(w.columns)]
base_m = st.tree_steiner(mst, TERMINALS); base_f = st.dreyfus_wagner(d, TERMINALS)
rng = np.random.default_rng(SEED)
inter_m, inter_f, jm, jf, cm, cf = {}, {}, [], [], [], []
B = 200
for b in range(B):
    ix = rl.block_indices(len(w), BLOCK, rng)
    wb = w.iloc[ix]; cb = nw.correlation(wb); db = nw.distance(cb)
    tm = st.tree_steiner(fl.mst_from_corr(cb), TERMINALS); tf = st.dreyfus_wagner(db, TERMINALS)
    jm.append(jac(base_m, tm)); jf.append(jac(base_f, tf))
    cm.append(st.summarize_tree(tm, TERMINALS)["cost"]); cf.append(st.summarize_tree(tf, TERMINALS)["cost"])
    for n in set(tm.nodes) - set(TERMINALS): inter_m[n] = inter_m.get(n, 0) + 1
    for n in set(tf.nodes) - set(TERMINALS): inter_f[n] = inter_f.get(n, 0) + 1
bs = dict(B=B, block=BLOCK,
          mst_subtree=dict(edge_jaccard_median=float(np.median(jm)), cost_p10=float(np.quantile(cm, .1)), cost_median=float(np.median(cm)), cost_p90=float(np.quantile(cm, .9)), base_cost=st.summarize_tree(base_m, TERMINALS)["cost"],
                           intermediate_frequency={n: v / B for n, v in sorted(inter_m.items(), key=lambda kv: -kv[1])[:10]}),
          full_steiner=dict(edge_jaccard_median=float(np.median(jf)), cost_p10=float(np.quantile(cf, .1)), cost_median=float(np.median(cf)), cost_p90=float(np.quantile(cf, .9)), base_cost=st.summarize_tree(base_f, TERMINALS)["cost"],
                            intermediate_frequency={n: v / B for n, v in sorted(inter_f.items(), key=lambda kv: -kv[1])[:10]}))
(R / "steiner_bootstrap.json").write_text(json.dumps(bs, indent=2) + "\n")
print(json.dumps(bs, indent=1), round(time.time() - t0))
# also write the primary trees' edges for the figure
rows = []
for lab, (corr_, pm_, mst_) in detail.items():
    d_ = nw.distance(corr_)
    for name, t in trees_for(corr_, pm_, TERMINALS)[0].items():
        for u, v, dd in t.edges(data=True): rows.append(dict(window_end=ENDS[lab], tree=name, u=u, v=v, distance=round(dd["distance"], 6)))
pd.DataFrame(rows).to_csv(R / "steiner_primary_tree_edges.csv", index=False)
