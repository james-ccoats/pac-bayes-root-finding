import pytest

from pac_bayes_root_finding.bounds import pinsker_bound
from pac_bayes_root_finding.kl_divergence import kl_divergence
from pac_bayes_root_finding.root_finding import kl_inv_bisection, kl_inv_newton

CASES = [
    (0.0, 0.01),
    (0.1, 0.05),
    (0.5, 0.2),
    (0.02, 1.5),
    (0.3, 1e-4),
    (0.9, 0.05),
]


@pytest.mark.parametrize("p,e", CASES)
def test_bisection_solves_kl_equation(p, e):
    q = kl_inv_bisection(p, e)
    assert kl_divergence(p, q) == pytest.approx(e, abs=1e-6)


@pytest.mark.parametrize("p,e", CASES)
def test_newton_agrees_with_bisection(p, e):
    assert kl_inv_newton(p, e) == pytest.approx(kl_inv_bisection(p, e), abs=1e-9)


@pytest.mark.parametrize("p,e", CASES)
def test_inverse_lies_between_p_and_pinsker_bound(p, e):
    q = kl_inv_bisection(p, e)
    assert p <= q <= pinsker_bound(p, e)


def test_zero_budget_returns_observed_rate():
    assert kl_inv_bisection(0.2, 0.0) == pytest.approx(0.2, abs=1e-6)


def test_inverse_is_monotone_in_budget():
    q_small = kl_inv_newton(0.1, 0.01)
    q_large = kl_inv_newton(0.1, 0.1)
    assert q_small < q_large
