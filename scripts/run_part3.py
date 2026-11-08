"""Reliability: joint moving-block bootstrap, community stability, window-length sensitivity, labelled synthetic recoverability."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import pipeline as pl, networks as nw, filters as fl, reliability as rl

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
px, returns, sec = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
sec = {k: v for k, v in sec.items() if k in returns.columns}
HEAD, BLOCK, SEED = "2024-12-30", 21, 20241230
t0 = time.time()
w = pl.window_ending(returns, HEAD, 252)
w = w[sorted(w.columns)]
n = w.shape[1]
corr_fn = nw.correlation
fns = {"MST": lambda c: fl.mst_from_corr(c), "Fixed density (3N-6 edges)": lambda c: fl.fixed_density_graph(c, 3 * (n - 2))}
out = {}
for name, gfn in fns.items():
    base = gfn(corr_fn(w))
    boots = list(rl.bootstrap_graphs(w, gfn, corr_fn, n_boot=300, block=BLOCK, seed=SEED))
    inc = rl.edge_inclusion(base, boots)
    inc["same_sector"] = [sec[u] == sec[v] for u, v in zip(inc.u, inc.v)]
    inc["distance"] = [base[u][v]["distance"] for u, v in zip(inc.u, inc.v)]
    inc.round(4).to_csv(R / f"bootstrap_edge_inclusion_{name.split(' ')[0].lower()}.csv", index=False)
    comm_base = nw.communities(base)
    ari = rl.community_stability(comm_base, boots[:100], nw.communities)
    out[name] = dict(edges=len(inc), median_inclusion=float(inc.inclusion.median()), share_ge_90=float((inc.inclusion >= .9).mean()), share_ge_50=float((inc.inclusion >= .5).mean()),
                     share_lt_25=float((inc.inclusion < .25).mean()),
                     median_incl_within_sector=float(inc[inc.same_sector].inclusion.median()), median_incl_between_sector=float(inc[~inc.same_sector].inclusion.median()),
                     corr_distance_inclusion=float(np.corrcoef(inc.distance, inc.inclusion)[0, 1]),
                     community_ari_median=float(np.nanmedian(ari)), community_ari_p10=float(np.nanquantile(ari, .1)), community_ari_p90=float(np.nanquantile(ari, .9)), n_boot=300, block=BLOCK)
    print(name, json.dumps(out[name], indent=1), "t", time.time() - t0, flush=True)
# PMFG bootstrap skipped: greedy planarity filtering costs ~100 s per graph on this panel (documented limitation)
(R / "bootstrap_summary.json").write_text(json.dumps(out, indent=2) + "\n")
# window-length sensitivity (same stocks, same end date)
names = sorted(returns.columns)
wl = {}
for label, end in {"2024": HEAD, "2017": "2017-12-29", "2008": "2008-12-31"}.items():
    m = rl.window_length_overlap(returns, end, names, lengths=(126, 252, 504))
    m.round(4).to_csv(R / f"window_length_jaccard_{label}.csv"); wl[label] = m.round(3).to_dict()
print(json.dumps(wl, indent=1))
# SYNTHETIC mechanism experiment: block-correlation Gaussian model, inputs taken from the real matrices
full = pl.full_matrix_summary(corr_fn(w), sec)
settings = {"2024-like (within %.2f, between %.2f)" % (full["mean_within_sector"], full["mean_between_sector"]): (full["mean_within_sector"], full["mean_between_sector"]),
            "2008-like (within 0.63, between 0.54)": (0.63, 0.537), "2017-like (within 0.35, between 0.11)": (0.352, 0.114)}
rows = []
for lab, (wi, be) in settings.items():
    r = rl.planted_block_recovery(n, 11, wi, be, [63, 126, 252, 504, 1008, 2016], n_rep=20, seed=7)
    g = r.groupby("T")[["same_block_edge_share", "ari"]].mean().reset_index(); g["setting"] = lab; rows.append(g)
syn = pd.concat(rows); syn.round(4).to_csv(R / "synthetic_recoverability.csv", index=False)
print(syn.round(3).to_string(), "t", time.time() - t0)
