"""Compare bisection and Newton's method for the upper kl-inverse.

Run from the repository root:

    uv run python experiments/compare_methods.py
"""

import time

from pac_bayes_root_finding.bounds import calculate_e, pinsker_bound
from pac_bayes_root_finding.root_finding import kl_inv_bisection, kl_inv_newton

DELTA = 0.05
KL_QP = 20.0
SAMPLE_SIZES = (100, 1_000, 10_000, 100_000)
EMPIRICAL_ERROR = 0.05
REPEATS = 2_000


def time_per_call(fn, *args):
    start = time.perf_counter()
    for _ in range(REPEATS):
        fn(*args)
    return (time.perf_counter() - start) / REPEATS * 1e6


def main():
    header = (
        f"{'n':>8} {'e':>9} {'Pinsker':>9} {'bisection':>11} {'newton':>11} "
        f"{'|diff|':>9} {'bisect us':>10} {'newton us':>10}"
    )
    print(f"Empirical error p = {EMPIRICAL_ERROR}, KL(Q||P) = {KL_QP}, delta = {DELTA}\n")
    print(header)
    print("-" * len(header))
    for n in SAMPLE_SIZES:
        e = calculate_e(KL_QP, n, DELTA)
        bis = kl_inv_bisection(EMPIRICAL_ERROR, e)
        new = kl_inv_newton(EMPIRICAL_ERROR, e)
        print(
            f"{n:>8} {e:>9.5f} {pinsker_bound(EMPIRICAL_ERROR, e):>9.5f} "
            f"{bis:>11.8f} {new:>11.8f} {abs(bis - new):>9.1e} "
            f"{time_per_call(kl_inv_bisection, EMPIRICAL_ERROR, e):>10.1f} "
            f"{time_per_call(kl_inv_newton, EMPIRICAL_ERROR, e):>10.1f}"
        )


if __name__ == "__main__":
    main()
