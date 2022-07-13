import numpy as np
import multiprocessing as mp
from scipy.stats import betabinom

from utilities import *




def read_likelihood(n_ref, n_alt, genotype, log_scale = True): 
    f = 0.95
    omega = 100
    if genotype == 'R': 
        alpha = f * omega
        beta = omega - alpha
    elif genotype == 'A': 
        alpha = (1 - f) * omega
        beta = omega - alpha
    elif genotype == 'H': 
        alpha = omega/4 
        beta = omega/4 
    
    if log_scale: 
        return betabinom.logpmf(n_ref, n_ref + n_alt, alpha, beta)
    else: 
        return betabinom.pmf(n_ref, n_ref + n_alt, alpha, beta)


def locus_likelihood(ref, alt, genotypes):
    result = 0.
    for i in range(len(ref)): 
        result += read_likelihood(ref[i], alt[i], genotypes[i])
    return result


def likelihood_k_mut(ref, alt, gt1, gt2): 
    '''
    ref, alt: read counts for a locus
    gt1, gt2: genotypes of interest
    '''
    
    N = ref.size # number of cells
    #assert(alt.size == N)
    
    if gt1 == gt2: 
        return np.sum([read_likelihood(ref[i], alt[i]) for i in range(N)])
    
    # likelihoods[n, k]: log-likelihood that k of the first n cells have gt2 (and n-k of them have gt1), given the first n observations
    # If K < N: likelihoods[n, K] means K or more cells have gt2
    
    likelihoods = np.ones((N+1, N+1)) * (-np.inf) 
    likelihoods[0,0] = 0 # Trivial case when there is 0 cell: number of gt2 cells must be 0
    
    with np.errstate(divide='ignore'): # TODO: other methods to deal with log 0 problem? 
        for n in range(N): 
            likelihoods[n+1, 0] = likelihoods[n, 0] + read_likelihood(ref[n], alt[n], gt1)
            k_over_n = np.array([k/(n+1) for k in range(1,n+2)])
            l1 = np.log(1 - k_over_n) + read_likelihood(ref[n], alt[n], gt1) + likelihoods[n, 1:n+2]
            l2 = np.log(k_over_n) + read_likelihood(ref[n], alt[n], gt2) + likelihoods[n, 0:n+1]
            likelihoods[n+1, 1:n+2] = np.logaddexp(l1, l2)
    
    return likelihoods[N, :]


def posteriors_of_locus(ref, alt, RH_priors, HA_priors): 
    '''
    posteriors for different mutation states: ['R', 'H', 'A', 'RH', 'HA']
    '''
    RH = likelihood_k_mut(ref, alt, 'R', 'H')
    HA = likelihood_k_mut(ref, alt, 'H', 'A')

    joint_probabilities = np.array([RH[0] + RH_priors[0], # R
                                    HA[0] + HA_priors[0], # H
                                    HA[-1] + HA_priors[-1], # A 
                                    logsumexp(RH[1:-1] + RH_priors[1:-1]), # RH
                                    logsumexp(HA[1:-1] + HA_priors[1:-1]), # HA
                                   ])
    posteriors = lognormalize(joint_probabilities)
    return posteriors


def get_k_mut_priors(N, affect_all = True):
    result = np.array([2 * logbinom(N, k) - np.log(2*k-1) - logbinom(2*N, 2*k) for k in range(1,N+1)])
    if affect_all: # take into condiseration the case of all cells being mutated
        return result
    else: # consider only cases where at least two genotypes exist
        return lognormalize(result[:-1])


def get_composition_priors(n_cells, genotype_freq, mutation_rate):
    '''
    genotype_freq: dictionary, prior probabilities of the root having genotype R, H or A
    mutation_rate: proportion of mutated loci
    
    returns priors for RH mixture and AH mixture
    '''
    
    state_freq = {s: None for s in ['R', 'H', 'A', 'RH', 'HR', 'AH', 'HA']}
    # state_freq = pd.DataFrame(data = np.zeros(7), columns = ['frequency'], index = ['RR', 'HH', 'AA', ''])
    state_freq['R'] = genotype_freq['R'] * (1 - mutation_rate)
    state_freq['H'] = genotype_freq['H'] * (1 - mutation_rate)
    state_freq['A'] = genotype_freq['A'] * (1 - mutation_rate)
    state_freq['RH'] = genotype_freq['R'] * mutation_rate 
    state_freq['HR'] = genotype_freq['H'] * mutation_rate/2 
    state_freq['AH'] = genotype_freq['A'] * mutation_rate 
    state_freq['HA'] = genotype_freq['H'] * mutation_rate/2
    # convert into log space
    for s in state_freq: 
        state_freq[s] = np.log(state_freq[s])
    
    priors_k_mut = get_k_mut_priors(n_cells)
    
    result = np.ones(2*n_cells + 1) * (-np.inf) # R --- RH mixture --- H --- HA mixture --- A
    result[0] = state_freq['R']
    result[n_cells] = state_freq['H']
    result[-1] = state_freq['A']
    for k in range(1, n_cells+1): 
        result[k] = np.logaddexp(result[k], state_freq['RH'] + priors_k_mut[k-1])
        result[n_cells - k] = np.logaddexp(result[n_cells - k], state_freq['HR'] + priors_k_mut[k-1])
        result[-k-1] = np.logaddexp(result[-k-1], state_freq['AH'] + priors_k_mut[k-1])
        result[n_cells + k] = np.logaddexp(result[n_cells + k], state_freq['HA'] + priors_k_mut[k-1])
    
    return result
    

def get_posteriors(ref, alt, genotype_freq = {'R': 1/3, 'H': 1/3, 'A': 1/3}, mutation_rate = 0.25, log_scale = False, n_threads = 1): 
    n_loci = ref.shape[0]
    n_cells = ref.shape[1]
    # assert(n_loci == alt.shape[0] and n_cells == alt.shape[1])
    # assert(df_ref.index.size == n_loci)
    
    # get priors for each situation 
    log_priors = get_composition_priors(n_cells, genotype_freq, mutation_rate)
    RH_priors = log_priors[:n_cells+1]
    HA_priors = log_priors[n_cells:]

    # multiprocessing
    pool = mp.Pool(n_threads)
    result = [pool.apply_async(posteriors_of_locus, (ref[i,:], alt[i,:], RH_priors, HA_priors)) for i in range(n_loci)]
    result = np.stack([r.get() for r in result])
    
    if not log_scale:
        result = np.exp(result)
    
    return result
    



    
    
    
    
    
if __name__ == '__main__': 
    import pandas as pd
    
    df_ref = pd.read_csv('./Data/glioblastoma_BT_S2/ref.csv', index_col = 0)
    df_alt = pd.read_csv('./Data/glioblastoma_BT_S2/alt.csv', index_col = 0)
    
    ref = df_ref.to_numpy(dtype = float)
    alt = df_alt.to_numpy(dtype = float)
    
    del df_ref, df_alt
    
    print(get_posteriors(ref[:20], alt[:20], n_threads = 6))
    
