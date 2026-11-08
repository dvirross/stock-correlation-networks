# Data

**No market data is committed.** The price panel and the membership file are not redistributed: Yahoo Finance data and the membership history have redistribution terms that were not verified. The analysis scripts stop with a clear message until the files below exist. Nothing is simulated except the clearly labelled synthetic mechanism experiment.

| File | Content | How to create |
|---|---|---|
| `prices_adjusted_close.csv` | `date` plus one column per ticker, split- and dividend-adjusted daily close, NaN where unavailable | From per-ticker yfinance exports with `scripts/prepare_prices.py`, or retrieve with `scripts/fetch_yfinance_template.py` (writes a provenance file and a suspect-jump report). A fresh retrieval may differ slightly from the March 2025 exports used here, because vendors revise adjustments |
| `sp_500_historical_components.csv` | `date,tickers` point-in-time S&P 500 membership | From the public [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents) history (check its licence before republishing). Used here: 3,344 snapshots, 458 members on 2003-01-06 |
| `sectors.csv` | `ticker,sector,source` | Committed: GICS sectors, verified as described below |

The per-ticker exports used here were retrieved with yfinance in March 2025. Delisted tickers (for example WYE, WLP and SEBL) came back as empty files: Yahoo returned no data, which is direct evidence of the survivorship problem. The export did not record whether `auto_adjust` was on, so the adjustment convention was checked against Interactive Brokers raw closes (below).

## Phase B: data actually used (all real; none committed except small derived files)

| Item | Detail |
|---|---|
| Price source | One yfinance CSV per ticker from the March 2025 exports (324 files). Close is split- and dividend-adjusted (checked against Interactive Brokers raw closes for KO, consistent with dividend and split adjustment). Simple returns, no forward fill. |
| Membership | S&P 500 membership on 2003-01-06 from `sp_500_historical_components.csv` (history derived from [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents); redistribution licence **unverified**, so the file is not committed). 458 members. |
| Panel rule | Continuous member 2003-01-06 to 2024-12-30 **and** complete plausible Yahoo history over 5,536 trading days: 205 continuous members, 203 after removing two artifacts. Retained 44% of the 2003 members; the panel is a survivor panel by construction. |
| Exclusions (255 of 458) | 106 Yahoo returned no data (empty file); 62 full history but not a continuous index member; 57 partial history; 28 export files under 100 KB (about a year or less of history) not used; DHR (+61% on 2016-07-05, Fortive spin-off adjustment artifact) and JCI (−53% on 2007-07-02, unadjusted 2-for-1 split) excluded rather than repaired. Per-ticker status: `data/derived/universe_audit_2003_members.csv`. |
| Other checks | `data/derived/panel_summary.json` (no missing values, longest stale run 4 days); `data/derived/panel_suspect_jumps.csv` (25 daily moves beyond ±40%: 23 in 2008–2009 and 2 in 2020, in AIG, regional banks, insurers and a few energy and industrial names; none reversed the next day, so they look like genuine crisis moves and were kept). |
| Sector labels | `data/sectors.csv`: GICS sectors. Written first from general knowledge (2026-10-08), then **verified on 2026-11-08**: all 203 agree with the S&P 500 constituents list dated 2024-12-27 in the public dataset [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies) (built from Wikipedia's constituents table). Between 2012-12-27 (first snapshot) and 2024 21 of the 203 changed sector there, mostly through GICS reorganizations (2016, 2018, 2023). So the labels are correct for the headline window and **not point-in-time** for earlier windows. Verification: `scripts/verify_sectors.py` (writes `data/derived/sector_verification.csv` and `_summary.json`); dated-label rerun of the rolling statistics: `scripts/run_pointintime_sectors.py` (needs the price panel). |
| Regeneration | Put the yfinance files in a folder, run `scripts/audit_universe.py` (writes `data/panel_adjusted_close.csv`, gitignored, and `data/derived/*`), then `scripts/run_part1.py` to `run_part4.py` and `scripts/make_figures.py`. |

Not available and not claimed: delisted-name prices (Yahoo returned none), point-in-time sectors before 2012-12-27, institutional-grade (CRSP/Norgate) adjustment audit.
