# Validation and benchmark notes

Everything below came from runs on my machine: macOS 26.4.1, Apple Silicon (arm64), Python 3.9.6, numpy 2.0.2, single thread. Raw output is committed in `results/validation.json` and `results/bench.json`. Reproduce with:

```
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/bench.py
.venv/bin/python -m pytest --cov=poolflow --cov-report=term
```

## Annuity exactness

Grid of 13 rates (0, 1e-12, 1e-9, 1e-6, then 0.5% through 18%) x 8 terms (6 to 360 months), 104 combos, $250k principal each.

- Closed-form balance after the full term: max residual 0.00 dollars across the grid. The balance formula uses the annuity-ratio form with expm1/log1p; the naive textbook form cancels catastrophically near zero rates, which I hit and fixed during development.
- Integer-cent schedules: 0 of 104 combos violated exact principal conservation (schedule principal sums to the cent, final balance exactly 0).
- Max final-payment adjustment vs the level payment: 5005 cents ($50.05), at the 18% x 360-month corner. That is compounded sub-cent rounding drift absorbed by the final payment, which is how a servicer's payoff quote works. At 6% x 360 months the adjustment is $3.66.

## PSA fixtures

19 hand-computed fixtures (100/150/200 PSA at ages 1 through 360), computed straight from the published formulas and stored in `fixtures/psa_fixtures.json`. Engine agreement: CPR max abs error 0.0, SMM max abs error 4.8e-13. The month-5 SMM of 0.000837 and month-30 SMM of 0.005143 at 100 PSA match the worked textbook values.

## Pool projection (seed 20260817, 5,000 loans, $1,365,965,009.39)

- Cash conservation at a half-cent tolerance: 0 violations out of 5,000 loans. Worst per-loan gap between total principal paid and starting balance: 9.3e-10 dollars, pure float64 roundoff.
- Final pool balance after 360 months: exactly 0.0.
- WAL at 100 PSA: 6.386873066655031 years. Independent pure-Python oracle: 6.386873066655034. Difference 2.7e-15 years.
- WAL sanity across speeds: 11.83 years at 0 PSA, 3.90 years at 300 PSA. Monotone as expected.

## Roll rates (seed 20260817)

- Iterative cohort evolution vs analytic matrix power, horizons 1 to 360 months: max deviation 1.8e-15.
- Monte Carlo with 200,000 simulated loans over 24 months vs the analytic distribution: max deviation 5.6e-4, consistent with sampling noise of order sqrt(p(1-p)/200000) which is about 1.1e-3 at p=0.5.
- Lifetime default via the fundamental matrix: 1.0 analytic vs 1.0 from 20,000 months of iteration, difference 0.0. This is expected, not impressive: default is the only absorbing state, so lifetime absorption is 1 by theorem. The informative numbers are the finite-horizon default shares: 2.1% at 12 months, 18.1% at 60, 34.5% at 120, 73.2% at 360 for a cohort starting current.

## Throughput

5,000 loans x 360-month horizon = 1.8M loan-months per projection, 7 repetitions after a warmup:

- median 7.5 ms per full projection, best 7.4 ms
- 239.0M loan-months/sec median, 242.0M best

Single thread, vectorized numpy, Apple Silicon. The monthly loop is Python; the per-loan work inside each month is where numpy earns the throughput. A pure-Python loop over loans would be orders of magnitude slower; I did not benchmark that variant.

## Tests

167 tests pass, coverage 100% of `poolflow` (196 statements):

```
.venv/bin/python -m pytest --cov=poolflow --cov-report=term
167 passed
```
