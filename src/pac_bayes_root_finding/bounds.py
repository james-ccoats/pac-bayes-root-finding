#PAC-Bayes quantities used by Maurer's bound.
import numpy as np


def calculate_e(kl_qp, n, delta):
    """e = [KL(Q || P) + ln(2 sqrt(n) / delta)] / n.

    kl_qp is KL(Q || P), the divergence between the posterior and prior
    distributions over model weights. Valid for n >= 8.
    """
    t2 = np.log(2 * np.sqrt(n) / delta)
    return (kl_qp + t2) / n


def pinsker_bound(p, e):
    """Pinsker relaxation of the upper kl-inverse: q <= p + sqrt(e / 2)."""
    return p + np.sqrt(e / 2)