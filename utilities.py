from scipy.special import loggamma, logsumexp
import numpy as np


def logbinom(n, k): 
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)


def lognormalize(array):
    return array - logsumexp(array)