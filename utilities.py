from scipy.special import loggamma, logsumexp
import numpy as np


def logbinom(n, k): 
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)


def lognormalize(array):
    return array - logsumexp(array)


def randint_with_exclude(n, exclude): 
    '''
    pick a random integer from [0, n-1] but exclude some integers
    exclude: list of integers to be exclude, should be strict subset of [0, n-1] and sorted
    '''
    # TBC: handle the case when len(exclude) >= n
    exclude.sort()
    rnd = np.random.randint(n - len(exclude))
    for i in exclude: 
        if rnd >= i: 
            rnd += 1
        else: 
            break
    return rnd