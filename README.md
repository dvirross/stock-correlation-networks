# Stock correlation networks on real returns

Reproducible analysis behind the article [From Correlation Matrices to Stock Networks](https://dvirross.com/quant-research/notes/stock-networks-correlation-sna) on dvirross.com ([Hebrew version](https://dvirross.com/he/quant-research/notes/stock-networks-correlation-sna)).

The question: what does a network built from a stock correlation matrix reveal, which of its features are artifacts of the construction, and how much of it is stable? The data are real daily returns of 203 stocks that were continuous S&P 500 members from 2003 to 2024 (a survivor panel, disclosed and measured). Nothing in this repository is investment advice, and the portfolio check reports risk only, not returns.

## What is here

```
stocknet/             data checks, correlation measures, MST / PMFG / fixed-density filters, bootstrap,
                      exact Steiner trees (Dreyfus-Wagner), portfolio rules, walk-forward accounting, Plotly builders
tests/                49 unit tests on small hand-made inputs with known answers (no market data)
scripts/              audit_universe.py (panel definition), run_part1.py ... run_part5.py, make_figures.py,
                      verify_sectors.py, run_pointintime_sectors.py, check_min_variance_solver.py,
                      prepare_prices.py, fetch_yfinance_template.py
notebooks/            stock-correlation-networks-analysis.ipynb runs the whole analysis from the price panel to the figures
results/              every number quoted in the article (CSV and JSON)
figures/              the published Plotly specifications (charts/) and static PNG fallbacks (images/)
data/                 sector labels, derived audit files and README.md (how to rebuild the panel; no prices committed)
editorial-validation/ independent checks on the saved results (block-length sensitivity of the portfolio intervals,
                      claims checker, Steiner pilot, stationary-null sketch)
METHODOLOGY.md        data conventions, panel rule, every method, seeds, and what was not done
```

## Run

```bash
pip install -r requirements.txt
pytest -q                                                   # 49 tests, no data needed
node editorial-validation/portfolio_block_sensitivity.mjs   # runs on the committed results
```

The full analysis needs the price panel, which is not redistributed. `data/README.md` explains how to rebuild it and which tickers were excluded and why; then run `scripts/run_part1.py` to `run_part5.py` and `scripts/make_figures.py`, or the notebook.

## Data and provenance

- Prices: Yahoo Finance daily closes exported with yfinance in March 2025, split- and dividend-adjusted (checked against Interactive Brokers raw closes for a sample stock). Simple returns, no gaps filled.
- Universe: the 458 S&P 500 members on 2003-01-06 from the point-in-time history in [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents); 203 were continuous members through 2024 with complete, plausible prices. The 255 exclusions are listed with reasons in `data/derived/universe_audit_2003_members.csv`.
- Sectors: GICS sectors. All 203 agree with the S&P 500 constituents list of 2024-12-27 in [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies); 21 changed sector there between 2012 and 2024, so the labels are not point-in-time for earlier windows (`data/derived/sector_verification.csv`).

## Limitations

A survivor panel (no delisted names or delisting returns), Yahoo adjusted prices rather than an institutional source, sector labels that are not point-in-time before 2024, one universe, one lookback and one rebalance frequency in the portfolio check. See `METHODOLOGY.md`, sections 7 and 8.

## Licence

The code is released under the MIT licence (`LICENSE`). Third-party data keep their own terms: the constituents history of [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents) and [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies), and Yahoo Finance prices, which are not redistributed here.

## Data for the interactive tools

`scripts/export_tool_data.py` writes `tool-data/` from the private price panel: yearly correlation matrices (2004–2024, rounded to 3 decimals), MST and PMFG edges with communities, and 300 bootstrap MSTs for the windows ending 2008 and 2024. These are derived statistics only, used by the interactive tools on dvirross.com; `tool-data/checks.json` records that they reproduce the published results.
