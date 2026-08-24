"""Throughput benchmark: loan-months per second for the pool projection.

Projects the seeded 5,000-loan pool to full runoff several times and
reports loan-months/sec, where loan-months counts every (loan, month)
cell actually evaluated (active loans only would undercount vectorized
work, so this uses n_loans x horizon, the shape the engine computes on).
Writes results/bench.json.
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poolflow.pool import generate_pool, project_pool

SEED = 20260817
REPS = 7


def main():
    pool = generate_pool(5000, SEED)
    # warmup
    project_pool(pool, psa=100.0)
    times = []
    horizon = None
    for _ in range(REPS):
        t0 = time.perf_counter()
        res = project_pool(pool, psa=100.0)
        times.append(time.perf_counter() - t0)
        horizon = res["ending_balance"].size
    loan_months = 5000 * horizon
    best = min(times)
    median = statistics.median(times)
    out = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "reps": REPS,
        "n_loans": 5000,
        "horizon_months": horizon,
        "loan_months_per_projection": loan_months,
        "times_sec": [round(t, 4) for t in times],
        "median_sec": round(median, 4),
        "best_sec": round(best, 4),
        "loan_months_per_sec_median": round(loan_months / median),
        "loan_months_per_sec_best": round(loan_months / best),
    }
    dest = Path(__file__).resolve().parent.parent / "results" / "bench.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
