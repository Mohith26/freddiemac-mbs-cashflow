"""poolflow: a small mortgage-pool cash-flow engine.

Loan-level amortization, standard prepayment conventions (SMM/CPR/PSA),
pool aggregation with servicing and guarantee-fee strips, delinquency
roll-rate Markov chains, and weighted-average life.
"""

from . import amortization, pool, prepayment, rollrates  # noqa: F401

__version__ = "1.0.0"
