# PoolFlow

I wanted to understand mortgage-backed securities from the ground up, so I built the machinery myself: a small engine that takes a pool of fixed-rate mortgages and projects every dollar of cash flow, month by month, until the last loan pays off. Along the way it handles the things that make agency-style pools interesting: borrowers prepaying, borrowers going delinquent, and the servicing and guarantee-fee strips that get carved out of interest before investors see it.

The rule I set for myself: every formula gets checked against something independent of my own code. Amortization against the closed-form annuity formula. Prepayment against hand-computed fixtures from published conventions. Cohort delinquency against analytic matrix powers. Weighted-average life against a second implementation written a different way. If two roads do not arrive at the same number, something is wrong.

## What is inside

**Loan-level amortization** (`poolflow/amortization.py`). The closed-form level payment, computed with expm1/log1p so it stays stable at rates arbitrarily close to zero, plus an integer-cent schedule generator that bills the way a servicer does: rounded payment, rounded monthly interest accrual, and a final payment adjusted so the loan retires exactly. The cent schedule conserves principal exactly, by construction and by test.

**Prepayment conventions** (`poolflow/prepayment.py`). SMM and CPR conversions and the standard PSA benchmark: CPR ramping 0.2% per month of loan age to 6% at month 30, then flat, with a linear multiplier. Formulas follow the open literature (Fabozzi's mortgage pass-through chapter is the reference I worked from); `fixtures/psa_fixtures.json` holds 19 hand-computed fixture values the tests must reproduce.

**Pool projection** (`poolflow/pool.py`). A seeded 5,000-loan synthetic pool projected to full runoff, vectorized in numpy: scheduled principal, prepayments at the PSA SMM, gross interest, servicing and guarantee-fee strips, and net investor interest. Every loan carries a cash-conservation invariant: total principal paid must equal the starting balance to within half a cent. Weighted-average life comes out of the same flows.

**Delinquency roll rates** (`poolflow/rollrates.py`). A five-state monthly Markov chain (current, 30, 60, 90+ days, default) with a seeded, realistically shaped transition matrix. Cohort evolution runs three ways: iterative propagation, analytic matrix powers, and Monte Carlo over individual loans. Lifetime absorption uses the fundamental-matrix formula N = (I - Q)^-1 from standard Markov-chain theory.

## Numbers from my machine

Full details and reproduce commands live in `RESULTS.md`; committed raw output in `results/`. Highlights from the seeded 5,000-loan pool (about $1.37B balance, Apple Silicon, single thread):

- 0 cash-conservation violations across all 5,000 loans; worst per-loan error was under a billionth of a dollar
- WAL at 100 PSA: 6.39 years, matching the independent oracle to 2.7e-15 years
- iterative cohort evolution matches matrix powers to 1.8e-15; Monte Carlo with 200k loans lands within 5.6e-4
- roughly 239 million loan-months per second of projection throughput

## Running it

```
python3 -m venv .venv && .venv/bin/pip install -U pip numpy pytest pytest-cov
.venv/bin/python -m pytest --cov=poolflow
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/bench.py
```

## Limitations

- All loan data is synthetic (seeded RNG). No real loan tapes, so the pool's rate/term/balance distributions are plausible but invented.
- Prepayment is the deterministic PSA convention only. There is no interest-rate model, no refinancing incentive, no burnout, no OAS.
- The roll-rate chain has default as its only absorbing state, so the lifetime default rate is 1.0 by construction; the meaningful outputs are the finite-horizon default shares and the speed of convergence, both of which are validated. A payoff state would fix this but is out of scope here.
- Delinquency and the cash-flow engine are validated side by side but not coupled: delinquent loans do not stop paying in the pool projection.
- The cent-level schedule shows honest final-payment drift at extreme rates: at 18% over 360 months the final payment absorbs about $50 of compounded sub-cent rounding. At realistic rates it stays within a few dollars, and principal conservation is exact everywhere.
- No tranching, no CMO structuring, no servicer advance modeling.
