from scipy.special import loggamma, logsumexp
import numpy as np


def logbinom(n, k): 
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)


def lognormalize(array):
    return array - logsumexp(array)


def randint_with_exclude(n, exclude): 
    '''
    pick a random integer from {0, ..., n-1}
    exclude: list of integers to be exclude, must be subset of {0, ..., n-1} and sorted
    '''
    rnd = np.random.randint(n - len(exclude))
    for i in exclude: 
        if rnd >= i: 
            rnd += 1
        else: 
            break
    return rnd