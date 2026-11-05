"""Root-finding methods for the upper kl-inverse.

Each kl_inv function finds the largest q in [p, 1) with
kl_divergence(p, q) <= e, (the root of g(q) = kl_divergence(p, q) - e.)
"""

from pac_bayes_root_finding.bounds import pinsker_bound
from pac_bayes_root_finding.kl_divergence import kl_divergence, kl_divergence_dq


def kl_inv_bisection(p, e, tol=1e-12):
    a, b = p, 1 - 1e-15
    while b - a > tol:
        mid = (a + b) / 2
        if kl_divergence(p, mid) > e:
            b = mid
        else:
            a = mid
    return a


def kl_inv_newton(p, e, tol=1e-12, max_iter=100):
    """Upper kl-inverse via Newton's method.

    Starts at the Pinsker bound, which lies at or above the root. Since
    kl_divergence(p, q) is convex and increasing in q on (p, 1), the
    iterates decrease monotonically toward the root and stay in the interval.
    The result may sit slightly above the root.

    Stops on the residual |g(q)| rather than the step size: near q = 1 the
    derivative is huge, so steps can be tiny even far from the root.
    """
    q = min(pinsker_bound(p, e), 1 - 1e-15)
    for _ in range(max_iter):
        g = kl_divergence(p, q) - e
        if abs(g) < tol:
            break
        q -= g / kl_divergence_dq(p, q)
    return q