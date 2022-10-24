from random import randrange
from scipy.special import loggamma, logsumexp
import numpy as np


def logbinom(n, k): 
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)


def lognormalize(array):
    return array - logsumexp(array)


def randint_with_exclude(n, exclude): 
    '''
    pick a random integer from [0, n-1] but exclude some integers
    exclude: list/numpy.array of integers to be excluded
    '''
    exclude.sort()
    rnd = randrange(0, n - len(exclude))
    for ex in exclude: 
        if rnd >= ex: 
            rnd += 1
        else:
            break
    return rnd


def path_len_dist(ct1, ct2): 
    ''' 
    Manhatten distance between the distance matrices of two cells trees 
    The distance matrix does not include internal nodes
    ''' 
    return np.sum(np.abs(ct1.dist_matrix - ct2.dist_matrix))