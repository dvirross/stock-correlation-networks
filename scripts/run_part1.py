import json, sys, time
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
w = pl.window_ending(returns, HEAD, 252)
c = nw.correlation(w); c = c.loc[sorted(c.index), sorted(c.columns)]
full = pl.full_matrix_summary(c, sec)
print("full matrix", json.dumps(full, indent=1))
st = pl.structure_table(c, sec)
st.round(4).to_csv(R / "structure_headline.csv"); print(st.round(3).T.to_string())
print("t", time.time() - t0)
ends = {"Calm 2007": "2007-06-29", "Crisis 2008": "2008-12-31", "Calm 2017": "2017-12-29", "COVID 2020": "2020-06-30", "Rates 2022": "2022-12-30", "Latest 2024": HEAD}
rt = pl.regime_table(returns, sec, ends)
rt.round(4).to_csv(R / "regime_table.csv"); print(rt.round(3).T.to_string())
print("t", time.time() - t0)
