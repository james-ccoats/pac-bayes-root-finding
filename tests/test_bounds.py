import numpy as np
import pytest

from pac_bayes_root_finding.bounds import calculate_e, pinsker_bound


def test_calculate_e_matches_formula():
    kl_qp, n, delta = 5.0, 1000, 0.05
    expected = (kl_qp + np.log(2 * np.sqrt(n) / delta)) / n
    assert calculate_e(kl_qp, n, delta) == pytest.approx(expected)


def test_calculate_e_shrinks_with_more_samples():
    assert calculate_e(5.0, 10_000, 0.05) < calculate_e(5.0, 1_000, 0.05)


def test_calculate_e_grows_with_posterior_prior_divergence():
    assert calculate_e(10.0, 1_000, 0.05) > calculate_e(1.0, 1_000, 0.05)


def test_pinsker_bound_matches_formula():
    assert pinsker_bound(0.1, 0.08) == pytest.approx(0.1 + np.sqrt(0.04))
