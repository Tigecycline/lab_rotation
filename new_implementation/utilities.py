from random import randrange
from scipy.special import loggamma, logsumexp
import numpy as np
import pandas as pd


def read_data(fn_ref, fn_alt, chromosome = None, row_is_cell = False): 
    df_ref = pd.read_csv(fn_ref, index_col = 0)
    df_alt = pd.read_csv(fn_alt, index_col = 0)
    
    df_ref['chromosome'] = [locus.split('_')[0] for locus in df_ref.index]
    df_ref['locus'] = [locus.split('_')[1] for locus in df_ref.index]
    df_ref = df_ref.set_index(['chromosome', 'locus']) # use multi-index

    df_alt['chromosome'] = [locus.split('_')[0] for locus in df_alt.index]
    df_alt['locus'] = [locus.split('_')[1] for locus in df_alt.index]
    df_alt = df_alt.set_index(['chromosome', 'locus'])
    
    if chromosome is not None: 
        df_ref = df_ref.loc[chromosome,:]
        df_alt = df_alt.loc[chromosome,:]
    
    ref = df_ref.to_numpy(dtype = int)
    alt = df_alt.to_numpy(dtype = int)
    #coverage = ref.flatten() + alt.flatten()
    
    if row_is_cell: 
        return ref, alt
    else: 
        return ref.T, alt.T


def logbinom(n, k): 
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)


def lognormalize(array):
    return array - logsumexp(array)


def randint_with_exclude(n, exclude): 
    '''
    pick a random integer from [0, n-1] but exclude some integers

    [Arguments]
        n: specifies the range to be sampled from
        exclude: array of integers to be excluded, all integers must be within range and mutually different
    '''
    exclude.sort()
    rnd = randrange(0, n - len(exclude))
    for ex in exclude: 
        if rnd >= ex: 
            rnd += 1
        else:
            break
    return rnd


def log_n_sbtrees(n_leaves):
    ''' number of full binary trees with n labelled leaves, in log space '''
    assert(n_leaves >= 1)
    if n_leaves == 1:
        return 0
    else:
        return loggamma(2*n_leaves - 2) - (n_leaves - 2) * np.log(2) - loggamma(n_leaves - 1)


def shannon_info(n_cells, n_affected):
    ''' Shannon information of a locus that affects exactly n_affected out of n_cells cells '''
    assert(n_cells > 0 and n_cells >= n_affected)
    
    if n_affected < 1:
        return 0
    else:
        return log_n_sbtrees(n_cells) - log_n_sbtrees(n_affected) - log_n_sbtrees(n_cells - n_affected + 1)