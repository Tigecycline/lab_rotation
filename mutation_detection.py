import numpy as np
import multiprocessing as mp
from scipy.stats import betabinom

from utilities import *




def single_read_likelihood(n_ref, n_alt, genotype, f = 0.95, omega = 100, log_scale = True): 
    ''' 
    likelihood of reference and alternative read counts at a specific cell and locus, given the corresponding genotype
    '''
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

    
def likelihood_matrices(ref, alt, gt1, gt2, correlated = None, n_threads = 4):
    '''
    likelihoods1[i,j]: likelihood of cell i having gt1 at locus j
    likelihoods2[i,j]: likelihood of cell i having gt2 at locus j
    '''
    n_cells = ref.shape[0]
    n_mut = ref.shape[1]
    likelihoods1 = np.empty((n_cells, n_mut)) # likelihood of cell i not mutated at locus j
    likelihoods2 = np.empty((n_cells, n_mut)) # likelihood of cell i mutated at locus j
    
    pool = mp.Pool(n_threads)
    def fill(i,j): 
        likelihoods1[i,j] = single_read_likelihood(ref[i,j], alt[i,j], gt1[j])
        likelihoods2[i,j] = single_read_likelihood(ref[i,j], alt[i,j], gt2[j])
    
    for i in range(n_cells): 
        for j in range(n_mut): 
            pool.apply_async(fill(i,j))
    return likelihoods1, likelihoods2


def k_mut_priors(N, affect_all = True):
    result = np.array([2 * logbinom(N, k) - np.log(2*k-1) - logbinom(2*N, 2*k) for k in range(1,N+1)])
    if affect_all: # take into condiseration the case of all cells being mutated
        return result
    else: # consider only cases where at least two genotypes exist
        return lognormalize(result[:-1])


def k_mut_likelihoods(ref, alt, gt1, gt2): 
    '''
    ref, alt: read counts for a locus
    gt1, gt2: genotypes of interest
    '''
    
    N = ref.size # number of cells
    #assert(alt.size == N)
    
    if gt1 == gt2: 
        return np.sum([single_read_likelihood(ref[i], alt[i], gt1) for i in range(N)])
    
    k_in_first_n = np.ones((N+1, N+1)) * (-np.inf) # [n,k]: likelihood that k among the first n cells are mutated 
    k_in_first_n[0,0] = 0 # Trivial case when there is 0 cell: number of mutated cells must be 0
    
    with np.errstate(divide='ignore'): # TODO: other methods to deal with log 0 problem? 
        for n in range(N): 
            k_in_first_n[n+1, 0] = k_in_first_n[n, 0] + single_read_likelihood(ref[n], alt[n], gt1)
            k_over_n = np.array([k/(n+1) for k in range(1,n+2)])
            l1 = np.log(1 - k_over_n) + single_read_likelihood(ref[n], alt[n], gt1) + k_in_first_n[n, 1:n+2]
            l2 = np.log(k_over_n) + single_read_likelihood(ref[n], alt[n], gt2) + k_in_first_n[n, 0:n+1]
            k_in_first_n[n+1, 1:n+2] = np.logaddexp(l1, l2)
    
    return k_in_first_n[N, :]


def composition_priors(n_cells, genotype_freq, mutation_rate):
    '''
    genotype_freq: prior probabilities of the root having genotype R, H or A
    mutation_rate: proportion of mutated loci
    
    In case of 'R', 'H' and 'A', the result is a single prior value
    In case the locus is mutated, the result is an array of priors for different compositions
    A "composition" refers to a specific number of affected cells
    N.B. for computational convenience, the arrays 'HR' and 'AH' are flipped
    '''
    state_freq = {s: None for s in ['R', 'H', 'A', 'RH', 'HR', 'AH', 'HA']}
    state_freq['R'] = genotype_freq['R'] * (1 - mutation_rate)
    state_freq['H'] = genotype_freq['H'] * (1 - mutation_rate)
    state_freq['A'] = genotype_freq['A'] * (1 - mutation_rate)
    state_freq['RH'] = genotype_freq['R'] * mutation_rate 
    state_freq['HR'] = genotype_freq['H'] * mutation_rate/2 
    state_freq['AH'] = genotype_freq['A'] * mutation_rate 
    state_freq['HA'] = genotype_freq['H'] * mutation_rate/2
    for s in state_freq: 
        state_freq[s] = np.log(state_freq[s]) # convert to log space
    
    k_priors = k_mut_priors(n_cells)

    result = {}
    for state in ['R', 'H', 'A']: 
        result[state] = state_freq[state]
    for state in ['RH', 'HR', 'HA', 'AH']:
        result[state] = state_freq[state] + k_priors
    for state in ['HR', 'AH']:
        result[state] = np.flip(result[state])
    
    return result


def locus_posteriors(ref, alt, priors): 
    '''
    different mutation states: ['R', 'H', 'A', 'RH', 'HR', 'HA', 'AH']
    '''
    RH_likelihoods = k_mut_likelihoods(ref, alt, 'R', 'H')
    HA_likelihoods = k_mut_likelihoods(ref, alt, 'H', 'A')
    assert(RH_likelihoods[-1] == HA_likelihoods[0])

    joint_probabilities = np.array([
        RH_likelihoods[0] + priors['R'],
        HA_likelihoods[0] + priors['H'],
        HA_likelihoods[-1] + priors['A'],
        logsumexp(RH_likelihoods[1:] + priors['RH']),
        logsumexp(RH_likelihoods[:-1] + priors['HR']),
        logsumexp(HA_likelihoods[1:] + priors['HA']),
        logsumexp(HA_likelihoods[:-1] + priors['AH'])
    ])
    posteriors = lognormalize(joint_probabilities)
    return posteriors
    

def mut_type_posteriors(ref, alt, genotype_freq = {'R': 1/4, 'H': 1/2, 'A': 1/4}, mutation_rate = 0.5, n_threads = 4, ignore_dir = True): 
    n_cells, n_loci = ref.shape
    # assert(n_loci == alt.shape[0] and n_cells == alt.shape[1])
    # assert(df_ref.index.size == n_loci)
    
    # get priors for each situation 
    priors = composition_priors(n_cells, genotype_freq, mutation_rate)

    # multiprocessing
    pool = mp.Pool(n_threads)
    result = [pool.apply_async(locus_posteriors, (ref[:,j], alt[:,j], priors)) for j in range(n_loci)]
    result = np.stack([r.get() for r in result])

    result = np.exp(result) # convert back to linear space
    if ignore_dir: # consider mutations with opposite directions as a single type
        result[:,3] = result[:,3] + result[:,4] # merge RH and HR
        result[:,4] = result[:,5] + result[:,6] # merge HA and AH
        result = result[:,:5]
    return result