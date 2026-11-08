"""Record optimizer status for every minimum-variance rebalance window used in the walk-forward comparison (QA request:
a silent equal-weight fallback would contaminate the comparison). Same windows as run_part4.py (252 days, month-ends from 2005-01)."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import pipeline as pl, portfolios as pf, backtest as bt

ROOT = Path(__file__).resolve().parent.parent
px, returns, sec = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
rows = []
for t in bt.month_end_dates(returns.index, start="2005-01-01"):
    win = returns.loc[:t].tail(252)
    _, st = pf.min_variance_solve(win)
    rows.append(dict(date=pd.Timestamp(t).date(), **st))
d = pd.DataFrame(rows)
d.to_csv(ROOT / "results/min_variance_solver_status.csv", index=False)
summ = dict(windows=len(d), converged=int(d.success.sum()), fallbacks=int(d.fallback.sum()), max_iterations=int(d.n_iter.max()))
(ROOT / "results/min_variance_solver_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
print(summ)
