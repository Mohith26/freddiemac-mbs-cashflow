"""Pool-level cash-flow projection.

A pool is a set of level-payment loans projected jointly, month by month,
fully vectorized in numpy. Each month, for every active loan:

1. gross interest accrues on the beginning balance at the note rate / 12,
2. the scheduled principal is the level payment minus interest (capped at
   the balance, and forced to clear the balance in the final month),
3. a fraction SMM of the remaining balance prepays (PSA convention by
   default, applied to the balance net of scheduled principal, which is
   the standard convention),
4. servicing and guarantee-fee strips accrue on the beginning balance and
   are carved out of gross interest; the investor receives the rest.

The projection reports monthly aggregate flows plus per-loan principal
totals so a cash-conservation invariant can be checked loan by loan.
"""

from typing import Dict, Optional

import numpy as np

from .prepayment import psa_smm


def generate_pool(n_loans: int, seed: int) -> Dict[str, np.ndarray]:
    """Deterministic synthetic pool. All loans are new (age 0) fixed-rate.

    Balances 50k to 500k, note rates 3% to 7.5%, terms 120 to 360 months.
    """
    if n_loans <= 0:
        raise ValueError("n_loans must be positive")
    rng = np.random.default_rng(seed)
    balance = np.round(rng.uniform(50_000, 500_000, n_loans), 2)
    rate = np.round(rng.uniform(0.03, 0.075, n_loans), 4)
    term = rng.integers(120, 361, n_loans)
    r = rate / 12.0
    payment = balance * r / (1.0 - (1.0 + r) ** (-term))
    return {"balance": balance, "rate": rate, "term": term, "payment": payment}


def project_pool(
    pool: Dict[str, np.ndarray],
    psa: float = 100.0,
    servicing_rate: float = 0.0025,
    gfee_rate: float = 0.0050,
    smm_override: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Project pool cash flows to full runoff.

    smm_override, if given, is a per-month SMM vector (indexed by loan age,
    1-indexed at position 0) that replaces the PSA curve; used for tests.
    Returns monthly aggregates and per-loan principal totals.
    """
    bal = pool["balance"].astype(float).copy()
    rate = pool["rate"].astype(float)
    term = pool["term"].astype(int)
    pmt = pool["payment"].astype(float)
    r = rate / 12.0
    n_loans = bal.size
    horizon = int(term.max())

    initial_balance = bal.sum()
    per_loan_principal = np.zeros(n_loans)

    sched_prin_m = np.zeros(horizon)
    prepay_m = np.zeros(horizon)
    gross_int_m = np.zeros(horizon)
    servicing_m = np.zeros(horizon)
    gfee_m = np.zeros(horizon)
    net_int_m = np.zeros(horizon)
    end_bal_m = np.zeros(horizon)

    for m in range(1, horizon + 1):
        active = bal > 0.0
        if not active.any():
            break
        b0 = np.where(active, bal, 0.0)
        interest = b0 * r
        sched = np.where(active, pmt - interest, 0.0)
        # final scheduled month or rounding: never amortize below zero
        final_month = active & (m >= term)
        sched = np.where(final_month, b0, np.minimum(sched, b0))
        sched = np.maximum(sched, 0.0)
        after_sched = b0 - sched
        smm = float(smm_override[m - 1]) if smm_override is not None else psa_smm(m, psa)
        prepay = smm * after_sched
        bal = after_sched - prepay

        servicing = b0 * (servicing_rate / 12.0)
        gfee = b0 * (gfee_rate / 12.0)

        per_loan_principal += sched + prepay
        i = m - 1
        sched_prin_m[i] = sched.sum()
        prepay_m[i] = prepay.sum()
        gross_int_m[i] = interest.sum()
        servicing_m[i] = servicing.sum()
        gfee_m[i] = gfee.sum()
        net_int_m[i] = (interest - servicing - gfee).sum()
        end_bal_m[i] = bal.sum()

    return {
        "initial_balance": initial_balance,
        "scheduled_principal": sched_prin_m,
        "prepaid_principal": prepay_m,
        "gross_interest": gross_int_m,
        "servicing_fee": servicing_m,
        "guarantee_fee": gfee_m,
        "net_interest": net_int_m,
        "ending_balance": end_bal_m,
        "per_loan_principal_total": per_loan_principal,
        "per_loan_initial_balance": pool["balance"].astype(float),
    }


def conservation_violations(result: Dict[str, np.ndarray], tol: float = 0.005) -> int:
    """Number of loans whose total principal paid differs from the initial
    balance by more than tol dollars (default: half a cent)."""
    diff = np.abs(result["per_loan_principal_total"] - result["per_loan_initial_balance"])
    return int((diff > tol).sum())


def wal_years(result: Dict[str, np.ndarray]) -> float:
    """Weighted-average life in years: principal-weighted mean payoff time.

    WAL = sum_m (m / 12) * P_m / sum_m P_m  with m the 1-indexed month.
    """
    principal = result["scheduled_principal"] + result["prepaid_principal"]
    months = np.arange(1, principal.size + 1)
    total = principal.sum()
    if total == 0:
        raise ValueError("no principal flows")
    return float((months / 12.0 * principal).sum() / total)
