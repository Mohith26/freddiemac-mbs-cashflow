"""Standard prepayment conventions: SMM, CPR and the PSA benchmark.

Formulas are the industry-standard ones documented in open literature
(Fabozzi, "Bond Markets, Analysis, and Strategies", mortgage pass-through
chapter; the PSA benchmark was published by the Public Securities
Association, now SIFMA):

    SMM = 1 - (1 - CPR) ** (1/12)
    CPR = 1 - (1 - SMM) ** 12

    100% PSA: CPR ramps linearly from 0.2% at loan age 1 month,
    increasing 0.2% per month, reaching 6% at month 30, then stays
    at 6% for the remaining life. X% PSA scales that CPR by X/100.

Loan age here is 1-indexed: age 1 is the loan's first month.
"""

import numpy as np


def cpr_to_smm(cpr):
    """Annual CPR to single monthly mortality. Accepts scalars or arrays."""
    cpr = np.asarray(cpr, dtype=float)
    if np.any(cpr < 0) or np.any(cpr > 1):
        raise ValueError("CPR must be in [0, 1]")
    out = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)
    return float(out) if out.ndim == 0 else out


def smm_to_cpr(smm):
    """Single monthly mortality to annual CPR. Accepts scalars or arrays."""
    smm = np.asarray(smm, dtype=float)
    if np.any(smm < 0) or np.any(smm > 1):
        raise ValueError("SMM must be in [0, 1]")
    out = 1.0 - (1.0 - smm) ** 12
    return float(out) if out.ndim == 0 else out


def psa_cpr(age_months, psa: float = 100.0):
    """CPR under the PSA benchmark at a given loan age (1-indexed).

    100 PSA is 0.2% CPR per month of age, capped at 6% from month 30 on.
    The multiplier scales CPR linearly; results are clipped to 100% CPR.
    """
    if psa < 0:
        raise ValueError("psa multiplier must be non-negative")
    age = np.asarray(age_months)
    if np.any(age < 1):
        raise ValueError("loan age is 1-indexed and must be >= 1")
    base = 0.002 * np.minimum(age, 30)
    out = np.clip(base * (psa / 100.0), 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def psa_smm(age_months, psa: float = 100.0):
    """SMM implied by the PSA benchmark at a given loan age."""
    return cpr_to_smm(psa_cpr(age_months, psa))
