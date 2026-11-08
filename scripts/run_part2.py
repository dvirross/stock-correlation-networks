import json, sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import pipeline as pl, networks as nw

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
px, returns, sec = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
sec = {k: v for k, v in sec.items() if k in returns.columns}
HEAD = "2024-12-30"
t0 = time.time()
# --- dependence measures on the identical window ---
w = pl.window_ending(returns, HEAD, 252)
corrs, graphs, agree, overlap, rows = pl.estimator_comparison(w, sec)
agree.round(4).to_csv(R / "estimator_pairwise_correlation_agreement.csv")
for g, m in overlap.items():
    m.round(4).to_csv(R / f"estimator_edge_jaccard_{g.split(' ')[0].lower()}.csv")
rows.round(4).to_csv(R / "estimator_graph_stats.csv")
print(agree.round(3).to_string()); print(overlap["MST"].round(3).to_string()); print(overlap["PMFG"].round(3).to_string()); print(rows.round(3).to_string())
print("t", time.time() - t0, flush=True)
# --- rolling monthly MST structure and persistence ---
roll, mst_graphs = pl.rolling_structure(returns, sec)
roll.round(5).to_csv(R / "rolling_structure_monthly.csv")
per = pl.persistence_vs_gap(mst_graphs)
per.round(4).to_csv(R / "mst_persistence_by_gap.csv")
print(per.round(3).to_string()); print(roll.describe().round(3).T.to_string())
print("corr(market vol, mean corr)", roll[["market_vol", "mean_corr"]].corr().iloc[0, 1], "corr(mean corr, jaccard)", roll[["mean_corr", "jaccard_prev"]].corr().iloc[0, 1])
print("t", time.time() - t0, flush=True)
