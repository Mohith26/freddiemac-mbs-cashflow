"""End-to-end validation run. Writes results/validation.json.

Everything here is recomputed from scratch on each run: annuity exactness
over a rate x term grid, PSA fixture agreement, pool cash conservation,
WAL vs an independent oracle, and roll-rate deviations vs matrix-power
and absorbing-chain oracles.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poolflow.amortization import balance_after, monthly_payment, schedule
from poolflow.pool import conservation_violations, generate_pool, project_pool, wal_years
from poolflow.prepayment import psa_cpr, psa_smm
from poolflow.rollrates import (
    evolve,
    evolve_oracle,
    lifetime_default_rate,
    seeded_matrix,
    simulate_cohort,
)

SEED = 20260817
ROOT = Path(__file__).resolve().parent.parent


def annuity_checks():
    rates = [0.0, 1e-12, 1e-9, 1e-6, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.0875, 0.12, 0.18]
    terms = [6, 12, 60, 120, 180, 240, 300, 360]
    max_residual = 0.0
    cent_violations = 0
    max_final_adj_cents = 0
    combos = 0
    for rate in rates:
        for term in terms:
            combos += 1
            # closed-form balance after full term must be ~0 (float only)
            max_residual = max(max_residual, abs(balance_after(300_000, rate, term, term)))
            # cent schedule must conserve principal exactly
            rows = schedule(25_000_000, rate, term)
            if sum(r.principal_c for r in rows) != 25_000_000 or rows[-1].balance_c != 0:
                cent_violations += 1
            pmt_c = round(monthly_payment(250_000, rate, term) * 100)
            max_final_adj_cents = max(max_final_adj_cents, abs(rows[-1].payment_c - pmt_c))
    return {
        "grid_combos": combos,
        "max_closed_form_residual_dollars": max_residual,
        "cent_schedule_conservation_violations": cent_violations,
        "max_final_payment_adjustment_cents": max_final_adj_cents,
    }


def psa_checks():
    fixtures = json.loads((ROOT / "fixtures" / "psa_fixtures.json").read_text())["fixtures"]
    max_cpr_err = 0.0
    max_smm_err = 0.0
    for fx in fixtures:
        max_cpr_err = max(max_cpr_err, abs(psa_cpr(fx["age"], fx["psa"]) - fx["cpr"]))
        max_smm_err = max(max_smm_err, abs(psa_smm(fx["age"], fx["psa"]) - fx["smm"]))
    return {
        "fixture_count": len(fixtures),
        "max_cpr_abs_error": max_cpr_err,
        "max_smm_abs_error": max_smm_err,
    }


def pool_checks():
    pool = generate_pool(5000, SEED)
    res = project_pool(pool, psa=100.0)
    diff = np.abs(res["per_loan_principal_total"] - res["per_loan_initial_balance"])
    wal = wal_years(res)
    # independent WAL oracle: pure-Python accumulation
    principal = (res["scheduled_principal"] + res["prepaid_principal"]).tolist()
    num = den = 0.0
    for i, p in enumerate(principal):
        num += (i + 1) * p
        den += p
    wal_oracle = num / den / 12.0
    return {
        "n_loans": 5000,
        "seed": SEED,
        "initial_balance_dollars": float(res["initial_balance"]),
        "horizon_months": int(res["ending_balance"].size),
        "conservation_violations_at_half_cent": conservation_violations(res),
        "max_per_loan_conservation_error_dollars": float(diff.max()),
        "final_pool_balance_dollars": float(res["ending_balance"][-1]),
        "wal_years_psa100": wal,
        "wal_oracle_years": wal_oracle,
        "wal_abs_diff_years": abs(wal - wal_oracle),
        "wal_years_psa0": wal_years(project_pool(pool, psa=0.0)),
        "wal_years_psa300": wal_years(project_pool(pool, psa=300.0)),
    }


def rollrate_checks():
    m = seeded_matrix(SEED)
    start = np.array([1.0, 0, 0, 0, 0])
    max_dev_iter = 0.0
    for n in (1, 6, 12, 60, 120, 360):
        max_dev_iter = max(
            max_dev_iter, float(np.abs(evolve(start, m, n) - evolve_oracle(start, m, n)).max())
        )
    mc = simulate_cohort(start, m, 24, 200_000, seed=1)
    mc_dev = float(np.abs(mc - evolve_oracle(start, m, 24)).max())
    analytic = lifetime_default_rate(start, m)
    iterated = float(evolve(start, m, 20_000)[4])
    # finite-horizon default shares (lifetime is provably 1.0 here because
    # default is the only absorbing state in the 5-state chain)
    horizon_defaults = {
        str(n): float(evolve(start, m, n)[4]) for n in (12, 60, 120, 360)
    }
    return {
        "seed": SEED,
        "max_iterative_vs_matrix_power_deviation": max_dev_iter,
        "monte_carlo_loans": 200_000,
        "monte_carlo_horizon_months": 24,
        "max_monte_carlo_vs_analytic_deviation": mc_dev,
        "lifetime_default_rate_analytic": analytic,
        "lifetime_default_rate_iterated_20000m": iterated,
        "lifetime_default_abs_diff": abs(analytic - iterated),
        "default_share_at_horizon_months": horizon_defaults,
    }


def main():
    t0 = time.time()
    out = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "annuity": annuity_checks(),
        "psa": psa_checks(),
        "pool": pool_checks(),
        "rollrates": rollrate_checks(),
    }
    out["elapsed_sec"] = round(time.time() - t0, 2)
    dest = ROOT / "results" / "validation.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
