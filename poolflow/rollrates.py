"""Delinquency roll rates as a discrete-time Markov chain.

States, in order: current, d30, d60, d90p, default. Default is absorbing.
A cohort distribution row-vector evolves as x_{t+1} = x_t @ T.

Lifetime absorption probabilities come from standard absorbing-chain math
(see any Markov chains text, e.g. Grinstead and Snell, "Introduction to
Probability", chapter 11): partition T into transient block Q and
absorbing block R, the fundamental matrix is N = (I - Q)^-1 and the
absorption probability matrix is B = N @ R.
"""

from typing import Tuple

import numpy as np

STATES = ("current", "d30", "d60", "d90p", "default")
N_STATES = len(STATES)
ABSORBING = (4,)  # default
TRANSIENT = (0, 1, 2, 3)


def seeded_matrix(seed: int) -> np.ndarray:
    """Seeded, realistically shaped monthly roll-rate matrix.

    Structure: current mostly stays current with a small roll to d30;
    delinquent states can cure back to current, stay, or roll deeper;
    d90p can roll to default; default is absorbing. Random perturbations
    are drawn from the seed and rows are renormalized to sum to one.
    """
    rng = np.random.default_rng(seed)
    base = np.array(
        [
            #  cur    d30    d60    d90p   def
            [0.980, 0.020, 0.000, 0.000, 0.000],
            [0.300, 0.450, 0.250, 0.000, 0.000],
            [0.150, 0.100, 0.400, 0.350, 0.000],
            [0.080, 0.020, 0.050, 0.600, 0.250],
            [0.000, 0.000, 0.000, 0.000, 1.000],
        ]
    )
    noise = rng.uniform(0.9, 1.1, base.shape)
    m = base * noise
    m[4] = [0, 0, 0, 0, 1.0]  # keep default absorbing
    m /= m.sum(axis=1, keepdims=True)
    return m


def validate_matrix(m: np.ndarray) -> None:
    if m.shape != (N_STATES, N_STATES):
        raise ValueError("matrix must be 5x5")
    if np.any(m < 0):
        raise ValueError("probabilities must be non-negative")
    if not np.allclose(m.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("rows must sum to 1")


def evolve(dist: np.ndarray, matrix: np.ndarray, n_months: int) -> np.ndarray:
    """Iterative month-by-month cohort evolution (the engine path)."""
    validate_matrix(matrix)
    x = np.asarray(dist, dtype=float).copy()
    for _ in range(n_months):
        x = x @ matrix
    return x


def evolve_oracle(dist: np.ndarray, matrix: np.ndarray, n_months: int) -> np.ndarray:
    """Independent oracle: distribution times an analytic matrix power."""
    validate_matrix(matrix)
    return np.asarray(dist, dtype=float) @ np.linalg.matrix_power(matrix, n_months)


def simulate_cohort(
    dist: np.ndarray, matrix: np.ndarray, n_months: int, n_loans: int, seed: int
) -> np.ndarray:
    """Monte Carlo cohort: n_loans individual paths, returns the empirical
    state distribution after n_months."""
    validate_matrix(matrix)
    rng = np.random.default_rng(seed)
    states = rng.choice(N_STATES, size=n_loans, p=np.asarray(dist, dtype=float))
    cum = np.cumsum(matrix, axis=1)
    for _ in range(n_months):
        u = rng.random(n_loans)
        states = (u[:, None] > cum[states]).sum(axis=1)
    return np.bincount(states, minlength=N_STATES) / n_loans


def absorption_probabilities(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fundamental-matrix absorption math.

    Returns (N, B): N = (I - Q)^-1 over transient states, B = N @ R the
    probability each transient state is eventually absorbed in default.
    """
    validate_matrix(matrix)
    q = matrix[np.ix_(TRANSIENT, TRANSIENT)]
    r = matrix[np.ix_(TRANSIENT, ABSORBING)]
    n = np.linalg.inv(np.eye(len(TRANSIENT)) - q)
    return n, n @ r


def lifetime_default_rate(dist: np.ndarray, matrix: np.ndarray) -> float:
    """Lifetime default probability of a cohort starting at dist.

    Clipped to [0, 1]: the matrix inverse can overshoot 1.0 by a few
    machine epsilons when default is the only absorbing state.
    """
    _, b = absorption_probabilities(matrix)
    x = np.asarray(dist, dtype=float)
    raw = float(x[list(TRANSIENT)] @ b[:, 0] + x[list(ABSORBING)].sum())
    return min(max(raw, 0.0), 1.0)
