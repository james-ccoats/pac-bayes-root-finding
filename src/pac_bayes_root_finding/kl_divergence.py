#Binary KL divergence and its derivative.

#p is the observed error rate; q is a candidate true error rate.


import numpy as np


def kl_divergence(p, q):
    term1 = 0 if p == 0 else p * np.log(p / q)
    term2 = 0 if p == 1 else (1 - p) * np.log((1 - p) / (1 - q))
    return term1 + term2


def kl_divergence_dq(p, q):
    return (q - p) / (q * (1 - q))
    