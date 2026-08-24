"""Level-payment mortgage amortization.

The monthly payment on a fixed-rate, fully amortizing loan is the standard
annuity formula:

    payment = P * r / (1 - (1 + r) ** -n)    for monthly rate r > 0
    payment = P / n                          for r == 0

where P is the principal, r the monthly rate (annual rate / 12) and n the
term in months. This is the textbook formula, see for example Fabozzi,
"Bond Markets, Analysis, and Strategies", chapter on residential mortgage
loans, or any fixed-income reference.

Two layers live here:

* exact float arithmetic (payment, remaining balance) used for closed-form
  validation, computed with expm1/log1p so rates arbitrarily close to zero
  stay numerically stable, and
* an integer-cent schedule generator that mirrors how a servicer actually
  bills: the level payment is rounded to the cent, interest each month is
  the rounded accrual on the outstanding balance, and the final payment is
  adjusted so the loan pays off exactly.
"""

import math
from typing import List, NamedTuple


class ScheduleRow(NamedTuple):
    month: int
    payment_c: int
    interest_c: int
    principal_c: int
    balance_c: int


def monthly_rate(annual_rate: float) -> float:
    """Nominal annual rate to monthly rate (simple division by 12)."""
    return annual_rate / 12.0


def monthly_payment(principal: float, annual_rate: float, n_months: int) -> float:
    """Closed-form level payment for a fully amortizing loan.

    Uses expm1/log1p so the formula degrades gracefully to P / n as the
    rate approaches zero instead of blowing up on catastrophic cancellation.
    """
    _check_loan_args(principal, annual_rate, n_months)
    r = monthly_rate(annual_rate)
    if r == 0.0:
        return principal / n_months
    # 1 - (1+r)^-n  ==  -expm1(-n * log1p(r)), stable for tiny r
    denom = -math.expm1(-n_months * math.log1p(r))
    return principal * r / denom


def balance_after(principal: float, annual_rate: float, n_months: int, k: int) -> float:
    """Closed-form remaining balance after k level payments.

    Uses the annuity-ratio form, which is numerically stable for rates
    arbitrarily close to zero (the textbook P*(1+r)^k - pmt*((1+r)^k-1)/r
    form cancels catastrophically there):

        balance_k = P * (1 - (1+r)^-(n-k)) / (1 - (1+r)^-n)
    """
    _check_loan_args(principal, annual_rate, n_months)
    if k < 0 or k > n_months:
        raise ValueError("k must be in [0, n_months]")
    r = monthly_rate(annual_rate)
    if r == 0.0:
        return principal * (1.0 - k / n_months)
    log1pr = math.log1p(r)
    num = -math.expm1(-(n_months - k) * log1pr)
    den = -math.expm1(-n_months * log1pr)
    return principal * num / den


def schedule(principal_cents: int, annual_rate: float, n_months: int) -> List[ScheduleRow]:
    """Integer-cent amortization schedule.

    Interest each month is round(balance * r) in cents (Python round, i.e.
    banker's rounding at exact halves). The scheduled payment is the
    closed-form payment rounded to the cent; the final payment is whatever
    clears the remaining balance plus its interest, so the schedule always
    sums exactly to the original principal.
    """
    if principal_cents <= 0:
        raise ValueError("principal_cents must be positive")
    _check_loan_args(float(principal_cents), annual_rate, n_months)
    r = monthly_rate(annual_rate)
    pmt_c = round(monthly_payment(principal_cents / 100.0, annual_rate, n_months) * 100)
    rows = []
    bal = principal_cents
    for m in range(1, n_months + 1):
        interest_c = round(bal * r)
        principal_c = pmt_c - interest_c
        if m == n_months or principal_c >= bal:
            principal_c = bal  # final payment clears the balance exactly
        payment_c = interest_c + principal_c
        bal -= principal_c
        rows.append(ScheduleRow(m, payment_c, interest_c, principal_c, bal))
        if bal == 0:
            break
    return rows


def _check_loan_args(principal: float, annual_rate: float, n_months: int) -> None:
    if principal <= 0:
        raise ValueError("principal must be positive")
    if annual_rate < 0:
        raise ValueError("annual_rate must be non-negative")
    if n_months <= 0:
        raise ValueError("n_months must be a positive integer")
