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


def path_len_dist(tree1, tree2): 
    ''' 
    MSE between the (upper triangular) distance matrices of two cell/mutation trees 
    NB The distance matrix of a cell tree does not include internal nodes
    NB2 The denominator is the number of node pairs (ignoring order), not the matrix size
    ''' 
    dist_mat1, dist_mat2 = tree1.dist_matrix, tree2.dist_matrix
    denominator = (dist_mat1.size - dist_mat1.shape[0]) / 2
    return np.sum((dist_mat1 - dist_mat2)**2) / denominator