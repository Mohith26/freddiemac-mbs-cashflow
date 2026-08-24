import numpy as np
import pytest

from poolflow.amortization import schedule
from poolflow.pool import conservation_violations, generate_pool, project_pool, wal_years

SEED = 20260817


@pytest.fixture(scope="module")
def pool5k():
    return generate_pool(5000, SEED)


@pytest.fixture(scope="module")
def result5k(pool5k):
    return project_pool(pool5k, psa=100.0)


class TestPoolGeneration:
    def test_deterministic_given_seed(self):
        a = generate_pool(100, 7)
        b = generate_pool(100, 7)
        np.testing.assert_array_equal(a["balance"], b["balance"])
        np.testing.assert_array_equal(a["term"], b["term"])

    def test_different_seeds_differ(self):
        a = generate_pool(100, 1)
        b = generate_pool(100, 2)
        assert not np.array_equal(a["balance"], b["balance"])

    def test_sizes_and_ranges(self, pool5k):
        assert pool5k["balance"].size == 5000
        assert pool5k["balance"].min() >= 50_000 and pool5k["balance"].max() <= 500_000
        assert pool5k["rate"].min() >= 0.03 and pool5k["rate"].max() <= 0.075
        assert pool5k["term"].min() >= 120 and pool5k["term"].max() <= 360

    def test_payment_matches_annuity_formula(self, pool5k):
        r = pool5k["rate"][0] / 12
        expected = pool5k["balance"][0] * r / (1 - (1 + r) ** -int(pool5k["term"][0]))
        assert pool5k["payment"][0] == pytest.approx(expected, abs=1e-9)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            generate_pool(0, 1)


class TestProjection:
    def test_cash_conservation_zero_violations(self, result5k):
        assert conservation_violations(result5k) == 0

    def test_total_principal_equals_initial_balance(self, result5k):
        total = result5k["scheduled_principal"].sum() + result5k["prepaid_principal"].sum()
        assert total == pytest.approx(result5k["initial_balance"], abs=1e-4)

    def test_ending_balance_reaches_zero(self, result5k):
        assert result5k["ending_balance"][-1] == pytest.approx(0.0, abs=1e-6)

    def test_balances_never_negative(self, result5k):
        assert np.all(result5k["ending_balance"] >= -1e-9)

    def test_gross_interest_splits_into_net_and_strips(self, result5k):
        recombined = (
            result5k["net_interest"]
            + result5k["servicing_fee"]
            + result5k["guarantee_fee"]
        )
        np.testing.assert_allclose(recombined, result5k["gross_interest"], atol=1e-6)

    def test_zero_psa_matches_pure_scheduled_amortization(self):
        pool = generate_pool(50, 3)
        res = project_pool(pool, psa=0.0)
        assert res["prepaid_principal"].sum() == 0.0
        assert conservation_violations(res) == 0

    def test_single_loan_matches_cent_schedule(self):
        pool = {
            "balance": np.array([100_000.0]),
            "rate": np.array([0.06]),
            "term": np.array([360]),
            "payment": np.array([100_000 * 0.005 / (1 - 1.005 ** -360)]),
        }
        res = project_pool(pool, psa=0.0)
        rows = schedule(10_000_000, 0.06, 360)
        # engine works in float dollars, servicer schedule in cents: agree < 1 dollar
        for i in (0, 119, 359):
            assert res["scheduled_principal"][i] == pytest.approx(
                rows[i].principal_c / 100, abs=1.0
            )

    def test_higher_psa_shortens_wal(self):
        pool = generate_pool(500, 11)
        wal_slow = wal_years(project_pool(pool, psa=50.0))
        wal_base = wal_years(project_pool(pool, psa=100.0))
        wal_fast = wal_years(project_pool(pool, psa=300.0))
        assert wal_fast < wal_base < wal_slow

    def test_smm_override_full_prepay_first_month(self):
        pool = generate_pool(20, 5)
        smm = np.zeros(400)
        smm[0] = 1.0
        res = project_pool(pool, smm_override=smm)
        assert res["ending_balance"][0] == pytest.approx(0.0, abs=1e-9)
        assert conservation_violations(res) == 0

    def test_fees_accrue_on_beginning_balance(self):
        pool = generate_pool(10, 9)
        res = project_pool(pool, psa=0.0, servicing_rate=0.0025, gfee_rate=0.0050)
        b0 = pool["balance"].sum()
        assert res["servicing_fee"][0] == pytest.approx(b0 * 0.0025 / 12, abs=1e-6)
        assert res["guarantee_fee"][0] == pytest.approx(b0 * 0.0050 / 12, abs=1e-6)


class TestWAL:
    def test_wal_matches_independent_oracle(self, result5k):
        # oracle: pure-Python accumulation, written independently of wal_years
        principal = (
            result5k["scheduled_principal"] + result5k["prepaid_principal"]
        ).tolist()
        num = 0.0
        den = 0.0
        for i, p in enumerate(principal):
            num += (i + 1) * p
            den += p
        oracle = num / den / 12.0
        assert wal_years(result5k) == pytest.approx(oracle, abs=1e-10)

    def test_wal_of_immediate_payoff_is_one_month(self):
        pool = generate_pool(20, 5)
        smm = np.zeros(400)
        smm[0] = 1.0
        res = project_pool(pool, smm_override=smm)
        assert wal_years(res) == pytest.approx(1 / 12, abs=1e-12)

    def test_wal_within_pool_term_bounds(self, result5k):
        wal = wal_years(result5k)
        assert 0 < wal < 30

    def test_wal_raises_on_empty_flows(self):
        with pytest.raises(ValueError):
            wal_years(
                {
                    "scheduled_principal": np.zeros(10),
                    "prepaid_principal": np.zeros(10),
                }
            )
