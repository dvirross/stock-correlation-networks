/**
 * Independently replicate the stock-network portfolio volatility-difference
 * block-bootstrap sensitivity from the committed DAILY NET RETURN CSV.
 *
 * No market-price input, external packages, model training or browser required.
 * The observed return series are taken as fixed; the resampling exercise is
 * conditional on the strategy paths and does NOT eliminate survivorship bias
 * or account for re-estimating portfolio rules under new market histories.
 *
 * Run from the repository root (reads results/walkforward_daily_net_10bps.csv):
 *   node editorial-validation/portfolio_block_sensitivity.mjs
 * or
 *   node editorial-validation/portfolio_block_sensitivity.mjs \
 *     --input /path/to/walkforward_daily_net_10bps.csv \
 *     --output /path/to/portfolio_block_sensitivity.csv
 *
 * Reproducible seeded LCG, 600 paired bootstrap replications per block length.
 * All stocks/strategies are resampled on the SAME calendar-date indices.
 */
import fs from 'node:fs';
import path from 'node:path';

const arg = (flag, fallback) => {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : fallback;
};
const input = arg('--input', 'results/walkforward_daily_net_10bps.csv');
const output = arg('--output', '');
const blocks = [5, 10, 21, 42, 63, 126];
const bootstrapN = 600;
let seed = 237881; // UINT32, matches the independent V8-checkpoint calculation
const random = () => {
  seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
  return seed / 4294967296;
};
const quantile = (sorted, p) => {
  const j = (sorted.length - 1) * p;
  const i = Math.floor(j);
  return sorted[i] + (j - i) * (sorted[Math.ceil(j)] - sorted[i]);
};
const raw = fs.readFileSync(input, 'utf8').trim().split(/\r?\n/);
const columns = raw.shift().split(',');
const expected = ['', 'equal_weight', 'min_variance', 'hrp', 'mst_inverse_degree'];
if (columns.length !== expected.length || columns.some((s, i) => s !== expected[i])) {
  throw new Error('Unexpected strategy CSV columns: ' + columns.join(','));
}
const observations = raw.map(line => {
  const x = line.split(',');
  if (x.length !== 5) throw new Error('Malformed CSV row ' + line);
  const returns = x.slice(1).map(Number);
  if (returns.some(v => !Number.isFinite(v))) throw new Error('Non-numeric strategy return');
  return returns;
});
const n = observations.length;
if (n < 1000) throw new Error('Unexpectedly short time series');
const result = [];
for (const block of blocks) {
  const draws = [[], [], []];
  for (let rep = 0; rep < bootstrapN; rep++) {
    const sums = new Float64Array(4);
    const squares = new Float64Array(4);
    let collected = 0;
    while (collected < n) {
      const start = Math.floor(random() * (n - block + 1));
      for (let d = 0; d < block && collected < n; d++, collected++) {
        const vals = observations[start + d];
        for (let k = 0; k < 4; k++) {
          sums[k] += vals[k];
          squares[k] += vals[k] * vals[k];
        }
      }
    }
    const annualizedVol = Array.from({ length: 4 }, (_, k) =>
      Math.sqrt(Math.max(0, (squares[k] - sums[k] ** 2 / n) / (n - 1))) * Math.sqrt(252));
    for (let k = 1; k < 4; k++) draws[k - 1].push(100 * (annualizedVol[k] - annualizedVol[0]));
  }
  for (let j = 0; j < 3; j++) {
    const sorted = draws[j].sort((a, b) => a - b);
    result.push({
      block_days: block,
      strategy: expected[j + 2],
      resamples: bootstrapN,
      p025_diff_annual_vol_pp: quantile(sorted, .025).toFixed(6),
      median_diff_annual_vol_pp: quantile(sorted, .5).toFixed(6),
      p975_diff_annual_vol_pp: quantile(sorted, .975).toFixed(6),
      fraction_negative: (sorted.filter(x => x < 0).length / bootstrapN).toFixed(6),
    });
  }
}
const fieldnames = Object.keys(result[0]);
const table = [fieldnames.join(','), ...result.map(r => fieldnames.map(k => r[k]).join(','))].join('\n') + '\n';
if (output) {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, table);
}
process.stdout.write(table);
