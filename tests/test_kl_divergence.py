import numpy as np
import pytest

from pac_bayes_root_finding.kl_divergence import kl_divergence, kl_divergence_dq


def test_is_zero_when_distributions_match():
    assert kl_divergence(0.3, 0.3) == pytest.approx(0.0)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_is_non_negative(p):
    for q in np.linspace(0.01, 0.99, 25):
        assert kl_divergence(p, q) >= -1e-15


def test_handles_boundary_observed_rates():
    assert kl_divergence(0.0, 0.5) == pytest.approx(np.log(2))
    assert kl_divergence(1.0, 0.5) == pytest.approx(np.log(2))


def test_matches_closed_form():
    p, q = 0.2, 0.6
    expected = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    assert kl_divergence(p, q) == pytest.approx(expected)


@pytest.mark.parametrize("p,q", [(0.1, 0.4), (0.5, 0.7), (0.0, 0.3)])
def test_derivative_matches_finite_difference(p, q):
    h = 1e-6
    numeric = (kl_divergence(p, q + h) - kl_divergence(p, q - h)) / (2 * h)
    assert kl_divergence_dq(p, q) == pytest.approx(numeric, rel=1e-5)
