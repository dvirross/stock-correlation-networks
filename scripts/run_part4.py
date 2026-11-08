"""Walk-forward risk comparison on the SURVIVOR panel (descriptive; no performance-superiority claim).
Signal at the month-end close; primary convention: trade at the NEXT close (execution_lag=1). lag 0 (same-close, not implementable) is shown for contrast."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stocknet import pipeline as pl, portfolios as pf, backtest as bt, metrics as mt

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
px, returns, sec = pl.load_panel(ROOT / "data/panel_adjusted_close.csv", ROOT / "data/sectors.csv")
membership = pd.Series([frozenset(returns.columns)], index=pd.DatetimeIndex([returns.index[0]]))
rebal = bt.month_end_dates(returns.index, start="2005-01-01")
def _cached(rule):
    cache = {}
    def f(window):
        key = (window.index[0], window.index[-1], tuple(window.columns))
        if key not in cache:
            cache[key] = rule(window)
        return cache[key]
    return f
rules = {k: _cached(v) for k, v in pf.RULES.items()}  # weights depend only on the window, so reuse them across cost levels
res, t0 = {}, time.time()
for lag in (1, 0):
    for cost in (0, 10, 25):
        for name, rule in rules.items():
            res[(name, cost, lag)] = bt.run_backtest(returns, membership, rule, rebal, lookback=252, cost_bps=cost, min_names=50, execution_lag=lag)
            print(name, cost, lag, round(time.time() - t0), flush=True)
rows = []
for (name, cost, lag), r in res.items():
    s = mt.summarize(r.net, r.turnover)
    w = list(r.weights.values())
    s.update(strategy=name, cost_bps=cost, execution_lag=lag, avg_effective_n=float(np.mean([pf.effective_n(x) for x in w])), avg_names=float(np.mean([len(x) for x in w])),
             annual_one_way_turnover=float(r.turnover.mean() * 12), missing_events=r.missing_events, skipped_rebalances=len(r.log), n_rebalances=len(r.weights))
    rows.append(s)
summ = pd.DataFrame(rows).set_index(["strategy", "execution_lag", "cost_bps"]); summ.round(5).to_csv(R / "walkforward_summary.csv"); print(summ.round(4).to_string())
net10 = pd.DataFrame({n: res[(n, 10, 1)].net for n in rules}); net10.round(7).to_csv(R / "walkforward_daily_net_10bps.csv")
vol_stat = lambda x: x.std(ddof=1) * np.sqrt(252)
bs = {}
for n in rules:
    if n != "equal_weight":
        bs[n] = mt.block_bootstrap_diff(net10[n], net10["equal_weight"], stat=vol_stat, block=21, n_boot=2000, seed=11)
        print(n, bs[n])
(R / "walkforward_vol_difference_vs_equal_weight.json").write_text(json.dumps(bs, indent=2) + "\n")
# subperiods: annualised volatility by calendar regime for each strategy
sub = {"2005-2007": ("2005-01-01", "2007-12-31"), "2008-2009": ("2008-01-01", "2009-12-31"), "2010-2019": ("2010-01-01", "2019-12-31"), "2020-2024": ("2020-01-01", "2024-12-31")}
tab = pd.DataFrame({n: {k: float(net10[n].loc[a:b].std(ddof=1) * np.sqrt(252)) for k, (a, b) in sub.items()} for n in rules})
tab.round(4).to_csv(R / "walkforward_vol_by_subperiod.csv"); print(tab.round(3).to_string())
