# Methodology: stock-correlation networks on real returns (Phases A and B)

Status: Phase A (design, tooling, tests on fixtures) and Phase B (real-data analysis for the website article) were completed on 2026-10-08; the sector verification was added on 2026-11-08. All results in `results/` come from real observed Yahoo Finance adjusted closes of a **survivor panel of 203 stocks** (2003-01-02 to 2024-12-30), except `synthetic_recoverability.csv`, which is an explicitly labelled mechanism experiment, and the hand-made arrays in `tests/`, which are fixtures and never feed a result. Raw prices are not committed (see `data/README.md`).

## 1. Prior work searched first
Before writing code, existing repositories, notebooks and data were searched for a reusable correlation-network analysis. None computed correlation networks, MSTs, communities or HRP, so those parts are new. Reused: a point-in-time S&P 500 membership history (derived from the public [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents); licence unverified, not committed) and an archive of per-ticker yfinance exports from March 2025. The structure follows [`dvirross/leveraged-etf-compounding`](https://github.com/dvirross/leveraged-etf-compounding) (provenance, Plotly exports with static fallbacks, tests).

## 2. Data and conventions
* **Source (used):** one yfinance CSV per ticker from an archive of March 2025 exports; January 2003 membership from the hanshof history (licence unverified, not committed). The adjustment flag of the original export is not recorded in the files; the adjusted-close convention was checked against Interactive Brokers raw closes for KO and spot-checked for AAPL, KO and MSFT in the independent review (consistent with dividend and split adjustment, not a vendor audit). Provenance, exclusions and quality checks: `data/README.md`, `data/derived/`.
* **Adjustment convention:** split- and dividend-adjusted close, so close-to-close changes are total returns. Adjusted *levels* are retrospective (they change when later dividends occur); only returns are used.
* **Returns:** simple returns r_t = P_t / P_{t-1} - 1; a return is missing if either price is missing. No forward fill, no zero fill.
* **Corporate-action safeguards:** `quality_report` flags one-day moves beyond +-40% with next-day reversal (typical of adjustment errors), stale quotes (longest run of identical prices) and short histories. Flags are reported for manual review, never auto-repaired.
* **Missing data:** a stock enters a window only with at least 95% valid returns; remaining windows use complete columns only.

## 3. Universe and survivorship
* In Phase B the panel rule is stricter than this walk-forward rule: a stock must be a continuous index member from 2003-01-06 to 2024-12-30 with a complete, plausible price history (203 of the 458 January 2003 members). This is a survivor panel by construction, and the walk-forward comparison treats all 203 as always eligible (`membership` set to the panel). A stock is eligible at formation date t if it was an index member on t (last snapshot on or before t, no look-ahead) and has enough valid history ending at t.
* **Survivorship cannot be eliminated with Yahoo data** because delisted names are missing. It is measured by the panel rule above (203 of 458) and by `coverage_by_date`, which now counts a member as covered only if it has a non-null price on the date (or within an explicit lag), not a stale carried-forward price. Network *descriptions* are less sensitive than portfolio *performance*, which is the part most at risk.
* The backtest raises an error when a held ticker loses its returns (delisting or gap) instead of dropping it silently. Diagnostic modes (`zero`, `drop`) exist and are counted.
* Better data (Phase B decision): a vendor that includes delisted securities and delisting returns (for example CRSP, Norgate, Tiingo or EODHD). Dvir's existing IBKR access does not generally provide delisted history.

## 4. Windows
* Correlation lookback: 252 trading days (sensitivity 126, 504). Monthly rebalance at the last trading day.
* Forward window: returns of the trading days strictly after the rebalance close, until the next rebalance close.
* Overlapping windows are strongly dependent; inference is descriptive, with moving-block bootstraps on observed strategy returns. Persistence between windows is only reported against non-overlapping windows (or with the overlap fraction stated); bootstrap stability and temporal persistence are different quantities and are never mixed.

## 5. Network construction and metrics
* Pearson correlation of returns; Mantegna distance d = sqrt(2 (1 - rho)), a true metric (tested).
* Minimum spanning tree via scipy (n - 1 edges; brute-force check on a small case).
* Metrics per window: mean correlation, first-eigenvalue share (market-mode strength), MST total/mean edge length, maximum degree, leaf share, degree Gini, mean shortest path; per-node degree, betweenness and closeness (kept separate, they answer different questions).
* Communities: greedy modularity on the MST (inverse-distance weights) compared with sector labels by adjusted Rand index and modularity.
* Stability: edge Jaccard between windows; adjusted Rand index across time for communities.

## 6. Walk-forward evaluation (leakage-free by construction)
1. Signal date t (month-end close): eligible universe and weights use data up to and including t.
2. **Execution at the next close** (`execution_lag=1`, the default): the return of day t+1 is earned by the previous weights; the rebalance trades at the close of t+1 paying `cost_bps` on traded notional |target - drifted previous weights|. `execution_lag=0` (same-close, not implementable) exists only for contrast.
3. Every trading day from the first execution to the end of the sample is accounted for. A skipped rebalance continues the previous weights, so no holding-period return is dropped. The final month-end signal (2024-12-30) has no later close to trade at, so it is not executed: `walkforward_summary.csv` shows 239 executed rebalances and `skipped_rebalances=1` for `execution_lag=1` (and 240 and 0 for `execution_lag=0`). No rebalance was skipped for lack of eligible names.
Tests scramble all data after t and assert identical weights; cost, turnover, execution-lag and skipped-rebalance accounting are verified by hand-computed cases.

Strategies: equal weight; long-only minimum variance with a Ledoit-Wolf linear shrinkage toward a scaled identity (analytic intensity); Hierarchical Risk Parity (Lopez de Prado); exploratory MST inverse-degree. `walkforward_summary.csv` holds returns, volatility, Sharpe (no risk-free rate), drawdown, effective N and one-way turnover for costs of 0, 10 and 25 bps and both execution lags; **the article reports only risk, drawdown, turnover and effective N** because returns on a survivor panel without delisting returns are not evidence. Minimum-variance optimizer status was recorded for every rebalance window (`min_variance_solver_status.csv`); see the summary file for the fallback count. Null hypothesis stated up front: network-based rules do not beat the classical baselines after costs.

Phase B results (10 bps, next-close, 5,012 days from 2005-02-01): annualized volatility 19.7% (equal weight), 13.2% (minimum variance), 16.3% (HRP), 19.5% (MST inverse degree); worst drawdown -53.4%, -35.9%, -46.0%, -53.1%; 95% block-bootstrap ranges for the volatility difference against equal weight are in `walkforward_vol_difference_vs_equal_weight.json`.

## Phase B (empirical article) decisions

1. **Data:** survivor panel of 203 stocks (see `data/README.md`); no survivor-free claim is made. The portfolio comparison is descriptive and risk-focused.
2. **Filters compared on identical windows:** MST (N-1 edges), greedy PMFG (3N-6 edges), fixed-density top-(3N-6) pairs, plus the full weighted matrix summary. By-construction properties (tree: no triangles, every edge a bridge, connected; PMFG: connected, planar, contains MST; fixed density: may fragment) are reported as such.
3. **Estimators on identical windows:** Pearson, Spearman, Kendall (converted by sin(pi tau / 2)), and Pearson after removing the in-window equal-weight market return. They are different estimands; agreement is reported as pairwise-value correlation and edge Jaccard.
4. **Reliability:** joint moving-block bootstrap (block 21 days, B=300 for MST and fixed density; community ARI on the first 100 resamples; PMFG not bootstrapped because greedy planarity filtering costs ~100 s per graph); edge inclusion frequency is algorithmic stability, not truth probability; window-length overlap (126/252/504) for three end dates; month-to-month and gap persistence.
5. **Synthetic experiment (labelled):** eleven unequal-sized planted blocks (ten of 19 stocks and one of 13), with constant within- and between-block correlations calibrated from the real-data averages; reports same-block MST-edge share (the upper bound is (N-k)/(N-1)) and ARI as the sample length T varies. This is a mechanism check only, not a market-data result.
6. **Tail dependence and partial correlation:** not run; no estimand in the article requires them.
7. **Seeds:** bootstrap seed 20241230, synthetic seed 7, permutation test seed 0 (1,000 shuffles).

## 7. Resolved and open decisions
Resolved: price source (Yahoo adjusted closes from the March 2025 exports, survivor panel, disclosed); article reports structure, reliability and a risk-only portfolio check, not performance; reproducibility lives in this public repository.
Open: delisted-name prices and delisting returns (needs a licensed dataset such as CRSP or Norgate); membership-file licence; point-in-time sector labels before 2024 (labels are the GICS sectors of 2024-12-27, verified on 2026-11-08 with `scripts/verify_sectors.py`; dated-label rerun of the rolling statistics in `scripts/run_pointintime_sectors.py`, needs the price panel); the original export's `auto_adjust` flag; whether to move this folder to a public reproducibility repository.

## 8. Not done
No licensed or delisting-inclusive data; no point-in-time sectors before 2024 (dated labels exist only from 2012-12-27); no stationary-null calibration of year-to-year Jaccard differences; PMFG not bootstrapped (about 100 s per graph); tail dependence and full partial correlation not run. Test fixtures in `tests/` are hand-made or fixed-seed arithmetic inputs and are never used for results.
