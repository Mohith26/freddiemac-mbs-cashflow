import numpy as np
import pytest

from poolflow.rollrates import (
    N_STATES,
    STATES,
    absorption_probabilities,
    evolve,
    evolve_oracle,
    lifetime_default_rate,
    seeded_matrix,
    simulate_cohort,
    validate_matrix,
)

SEED = 20260817
START = np.array([1.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def matrix():
    return seeded_matrix(SEED)


class TestMatrix:
    def test_states_layout(self):
        assert STATES == ("current", "d30", "d60", "d90p", "default")
        assert N_STATES == 5

    def test_seeded_matrix_deterministic(self):
        np.testing.assert_array_equal(seeded_matrix(3), seeded_matrix(3))

    def test_rows_sum_to_one(self, matrix):
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(5), atol=1e-14)

    def test_default_is_absorbing(self, matrix):
        np.testing.assert_array_equal(matrix[4], [0, 0, 0, 0, 1.0])

    def test_validate_rejects_bad_shapes_and_rows(self):
        with pytest.raises(ValueError):
            validate_matrix(np.eye(4))
        bad = np.eye(5)
        bad[0, 0] = 0.5
        with pytest.raises(ValueError):
            validate_matrix(bad)
        neg = np.eye(5)
        neg[0, 0], neg[0, 1] = 1.5, -0.5
        with pytest.raises(ValueError):
            validate_matrix(neg)


class TestEvolution:
    @pytest.mark.parametrize("n", [1, 6, 12, 60, 120, 360])
    def test_iterative_matches_matrix_power_oracle(self, matrix, n):
        a = evolve(START, matrix, n)
        b = evolve_oracle(START, matrix, n)
        assert np.abs(a - b).max() < 1e-12

    def test_distribution_stays_on_simplex(self, matrix):
        x = evolve(START, matrix, 120)
        assert np.all(x >= 0)
        assert x.sum() == pytest.approx(1.0, abs=1e-12)

    def test_default_share_monotone_nondecreasing(self, matrix):
        prev = 0.0
        for n in range(1, 61):
            cur = evolve(START, matrix, n)[4]
            assert cur >= prev - 1e-15
            prev = cur

    def test_identity_matrix_freezes_distribution(self):
        ident = np.eye(5)
        np.testing.assert_array_equal(evolve(START, ident, 50), START)

    def test_monte_carlo_close_to_analytic(self, matrix):
        n_loans = 200_000
        mc = simulate_cohort(START, matrix, 24, n_loans, seed=1)
        analytic = evolve_oracle(START, matrix, 24)
        # sampling error ~ sqrt(p(1-p)/n) < 0.0012; allow 4 sigma
        assert np.abs(mc - analytic).max() < 0.005

    def test_monte_carlo_deterministic_given_seed(self, matrix):
        a = simulate_cohort(START, matrix, 12, 1000, seed=9)
        b = simulate_cohort(START, matrix, 12, 1000, seed=9)
        np.testing.assert_array_equal(a, b)


class TestAbsorption:
    def test_fundamental_matrix_on_hand_example(self):
        # 2 transient states embedded in 5-state layout is overkill for a
        # hand check, so verify N = (I-Q)^-1 satisfies N (I-Q) = I instead.
        m = seeded_matrix(4)
        n, _ = absorption_probabilities(m)
        q = m[:4, :4]
        np.testing.assert_allclose(n @ (np.eye(4) - q), np.eye(4), atol=1e-12)

    def test_lifetime_default_matches_long_horizon_iteration(self, matrix):
        analytic = lifetime_default_rate(START, matrix)
        iterated = evolve(START, matrix, 20_000)[4]
        assert analytic == pytest.approx(iterated, abs=1e-9)

    def test_all_transient_states_eventually_absorb(self, matrix):
        _, b = absorption_probabilities(matrix)
        # default is the only absorbing state, so every row absorbs there
        np.testing.assert_allclose(b[:, 0], np.ones(4), atol=1e-9)

    def test_lifetime_default_rate_bounds(self, matrix):
        rate = lifetime_default_rate(START, matrix)
        assert 0.0 <= rate <= 1.0

    def test_cohort_starting_in_default_stays_defaulted(self, matrix):
        start = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        assert lifetime_default_rate(start, matrix) == pytest.approx(1.0)

    def test_deeper_delinquency_defaults_sooner(self, matrix):
        # after a fixed horizon, a cohort starting at d90+ has more defaults
        cur = evolve(np.array([1.0, 0, 0, 0, 0]), matrix, 24)[4]
        d90 = evolve(np.array([0.0, 0, 0, 1.0, 0]), matrix, 24)[4]
        assert d90 > cur
