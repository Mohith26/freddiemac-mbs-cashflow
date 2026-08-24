import math

import pytest

from poolflow.amortization import (
    balance_after,
    monthly_payment,
    monthly_rate,
    schedule,
)


class TestClosedFormPayment:
    def test_textbook_payment_100k_6pct_30yr(self):
        # 100000 * 0.005 / (1 - 1.005^-360) = 599.5505...
        assert round(monthly_payment(100_000, 0.06, 360), 2) == 599.55

    def test_textbook_payment_200k_5pct_15yr(self):
        # independent recomputation: 200000 * (0.05/12) / (1 - (1+0.05/12)^-180)
        r = 0.05 / 12
        expected = 200_000 * r / (1 - (1 + r) ** -180)
        assert monthly_payment(200_000, 0.05, 180) == pytest.approx(expected, abs=1e-9)

    def test_zero_rate_payment_is_principal_over_term(self):
        assert monthly_payment(120_000, 0.0, 240) == pytest.approx(500.0)

    @pytest.mark.parametrize("tiny", [1e-15, 1e-12, 1e-9, 1e-7])
    def test_near_zero_rates_stay_stable(self, tiny):
        pmt = monthly_payment(100_000, tiny, 360)
        assert pmt == pytest.approx(100_000 / 360, rel=1e-4)
        assert pmt >= 100_000 / 360  # any positive rate costs at least P/n

    def test_payment_monotone_in_rate(self):
        rates = [0.0, 0.001, 0.01, 0.03, 0.06, 0.10, 0.15]
        pmts = [monthly_payment(250_000, r, 360) for r in rates]
        assert pmts == sorted(pmts)
        assert pmts[-1] > pmts[0]

    def test_payment_decreases_with_term(self):
        assert monthly_payment(100_000, 0.06, 360) < monthly_payment(100_000, 0.06, 180)

    @pytest.mark.parametrize(
        "principal,rate,term",
        [(-1, 0.05, 360), (0, 0.05, 360), (100_000, -0.01, 360), (100_000, 0.05, 0)],
    )
    def test_invalid_args_raise(self, principal, rate, term):
        with pytest.raises(ValueError):
            monthly_payment(principal, rate, term)

    def test_monthly_rate(self):
        assert monthly_rate(0.06) == pytest.approx(0.005)


class TestClosedFormBalance:
    @pytest.mark.parametrize("rate", [0.0, 1e-10, 0.02, 0.045, 0.06, 0.0875, 0.12])
    @pytest.mark.parametrize("term", [12, 120, 180, 360])
    def test_balance_after_full_term_is_zero(self, rate, term):
        residual = balance_after(300_000, rate, term, term)
        assert abs(residual) < 1e-5  # dollars, pure float roundoff

    def test_balance_at_time_zero_is_principal(self):
        assert balance_after(250_000, 0.05, 360, 0) == pytest.approx(250_000)

    def test_balance_zero_rate_linear(self):
        assert balance_after(240_000, 0.0, 240, 60) == pytest.approx(180_000)

    def test_balance_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            balance_after(100_000, 0.05, 360, 361)

    def test_balance_matches_recursive_amortization(self):
        p, rate, n = 175_000.0, 0.0525, 240
        r = rate / 12
        pmt = monthly_payment(p, rate, n)
        bal = p
        for k in range(1, 61):
            bal = bal * (1 + r) - pmt
        assert balance_after(p, rate, n, 60) == pytest.approx(bal, abs=1e-6)


class TestCentSchedule:
    @pytest.mark.parametrize("rate", [0.0, 0.001, 0.03, 0.06, 0.0999, 0.15])
    @pytest.mark.parametrize("term", [12, 120, 360])
    def test_principal_conserved_exactly_in_cents(self, rate, term):
        principal_c = 25_000_000  # 250k dollars
        rows = schedule(principal_c, rate, term)
        assert sum(row.principal_c for row in rows) == principal_c
        assert rows[-1].balance_c == 0

    def test_all_but_final_payment_match_closed_form_cents(self):
        rows = schedule(10_000_000, 0.06, 360)
        pmt_c = round(monthly_payment(100_000, 0.06, 360) * 100)
        assert all(row.payment_c == pmt_c for row in rows[:-1])

    def test_final_payment_adjustment_is_small(self):
        rows = schedule(10_000_000, 0.06, 360)
        pmt_c = round(monthly_payment(100_000, 0.06, 360) * 100)
        assert abs(rows[-1].payment_c - pmt_c) <= 200  # within 2 dollars

    def test_interest_is_rounded_accrual_on_balance(self):
        rows = schedule(10_000_000, 0.048, 180)
        bal = 10_000_000
        for row in rows[:5]:
            assert row.interest_c == round(bal * 0.048 / 12)
            bal = row.balance_c

    def test_balances_strictly_decreasing(self):
        rows = schedule(5_000_000, 0.07, 120)
        bals = [10_000_000] + [row.balance_c for row in rows]
        assert all(b1 > b2 for b1, b2 in zip(bals, bals[1:]))

    def test_payment_sums_to_principal_plus_interest(self):
        rows = schedule(20_000_000, 0.055, 240)
        total_pay = sum(r.payment_c for r in rows)
        total_int = sum(r.interest_c for r in rows)
        assert total_pay == 20_000_000 + total_int

    def test_zero_rate_schedule(self):
        rows = schedule(1_200_000, 0.0, 12)
        assert all(row.interest_c == 0 for row in rows)
        assert sum(row.principal_c for row in rows) == 1_200_000

    def test_invalid_principal_raises(self):
        with pytest.raises(ValueError):
            schedule(0, 0.05, 360)

    def test_schedule_month_indexing(self):
        rows = schedule(10_000_000, 0.06, 360)
        assert rows[0].month == 1
        assert rows[-1].month == len(rows)
