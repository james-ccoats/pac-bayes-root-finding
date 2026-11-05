"""Root-finding for the PAC-Bayes-kl (Maurer) generalization bound."""

from pac_bayes_root_finding.bounds import calculate_e, pinsker_bound
from pac_bayes_root_finding.kl_divergence import kl_divergence, kl_divergence_dq
from pac_bayes_root_finding.root_finding import kl_inv_bisection, kl_inv_newton

__all__ = [
    "calculate_e",
    "pinsker_bound",
    "kl_divergence",
    "kl_divergence_dq",
    "kl_inv_bisection",
    "kl_inv_newton",
]
