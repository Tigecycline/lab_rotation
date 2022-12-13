import numpy as np
import multiprocessing as mp
from scipy.stats import betabinom

from utilities import *




def single_read_likelihood(n_ref, n_alt, genotype, f = 0.95, omega = 100, log_scale = True): 
    ''' 
    likelihood of reference and alternative read counts at a specific cell and locus, given the corresponding genotype
    '''
    '''
    For python 3.10 consider this: 
    match genotype:
        case 'R':
            alpha = f * omega
            beta = omega - alpha
        case 'A':
            alpha = (1 - f) * omega
            beta = omega - alpha
        case 'H':
            alpha = omega/4
            beta = omega/4
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

    
def likelihood_matrices(ref, alt, gt1, gt2):
    '''
    likelihoods1[i,j]: likelihood of cell i having gt1 at locus j
    likelihoods2[i,j]: likelihood of cell i having gt2 at locus j
    '''
    n_cells = ref.shape[0]
    n_mut = ref.shape[1]
    likelihoods1 = np.empty((n_cells, n_mut)) # likelihood of cell i not mutated at locus j
    likelihoods2 = np.empty((n_cells, n_mut)) # likelihood of cell i mutated at locus j
    
    for i in range(n_cells): 
        for j in range(n_mut): 
            likelihoods1[i,j] = single_read_likelihood(ref[i,j], alt[i,j], gt1[j])
            likelihoods2[i,j] = single_read_likelihood(ref[i,j], alt[i,j], gt2[j])
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
    
    k_in_first_n = np.full((N+1, N+1), -np.inf) # [n,k]: likelihood that k among the first n cells are mutated 
    k_in_first_n[0,0] = 0 # Trivial case when there is 0 cell: number of mutated cells must be 0
    
    with np.errstate(divide='ignore'): # TODO: other methods to deal with log 0 problem? 
        for n in range(N): 
            k_in_first_n[n+1, 0] = k_in_first_n[n, 0] + single_read_likelihood(ref[n], alt[n], gt1)
            k_over_n = np.array([k/(n+1) for k in range(1,n+2)])
            l1 = np.log(1 - k_over_n) + single_read_likelihood(ref[n], alt[n], gt1) + k_in_first_n[n, 1:n+2]
            l2 = np.log(k_over_n) + single_read_likelihood(ref[n], alt[n], gt2) + k_in_first_n[n, 0:n+1]
            k_in_first_n[n+1, 1:n+2] = np.logaddexp(l1, l2)
    
    return k_in_first_n[N, :]


def composition_priors(n_cells, genotype_freq = {'R': 1/4, 'H': 1/2, 'A': 1/4}, mut_prop = 0.5):
    '''
    genotype_freq: prior probabilities of the root having genotype R, H or A
    mut_prop: prior for the proportion of mutated loci
    
    In case of 'R', 'H' and 'A', the result is a single prior value
    In case the locus is mutated, the result is an array of priors for different compositions
    A "composition" refers to a specific number of affected cells
    N.B. for computational convenience, the arrays 'HR' and 'AH' are flipped
    '''
    state_freq = {s: None for s in ['R', 'H', 'A', 'RH', 'HR', 'AH', 'HA']}
    state_freq['R'] = genotype_freq['R'] * (1 - mut_prop)
    state_freq['H'] = genotype_freq['H'] * (1 - mut_prop)
    state_freq['A'] = genotype_freq['A'] * (1 - mut_prop)
    state_freq['RH'] = genotype_freq['R'] * mut_prop 
    state_freq['HA'] = genotype_freq['H'] * mut_prop/2
    state_freq['HR'] = genotype_freq['H'] * mut_prop/2 
    state_freq['AH'] = genotype_freq['A'] * mut_prop
    
    for s in state_freq: 
        state_freq[s] = np.log(state_freq[s]) # convert to log space
    
    k_priors = k_mut_priors(n_cells)

    result = {}
    for state in ['R', 'H', 'A']: 
        result[state] = state_freq[state]
    for state in ['RH', 'HA', 'HR', 'AH']:
        result[state] = state_freq[state] + k_priors
    for state in ['HR', 'AH']:
        result[state] = np.flip(result[state])
    
    return result


def locus_posteriors(ref, alt, priors, single_cell_mut = False): 
    '''
    single_cell_mut: whether to consider a locus as mutated when the mutation affects one single cell
    different mutation states: ['R', 'H', 'A', 'RH', 'HA', 'HR', 'AH']
    '''
    RH_likelihoods = k_mut_likelihoods(ref, alt, 'R', 'H')
    HA_likelihoods = k_mut_likelihoods(ref, alt, 'H', 'A')
    assert(RH_likelihoods[-1] == HA_likelihoods[0])

    if single_cell_mut:
        joint_probabilities = np.array([
            RH_likelihoods[0] + priors['R'],
            HA_likelihoods[0] + priors['H'],
            HA_likelihoods[-1] + priors['A'],
            logsumexp(RH_likelihoods[1:] + priors['RH']),
            logsumexp(RH_likelihoods[:-1] + priors['HR']),
            logsumexp(HA_likelihoods[1:] + priors['HA']),
            logsumexp(HA_likelihoods[:-1] + priors['AH'])
        ])
    else:
        joint_probabilities = np.array([
            logsumexp([RH_likelihoods[0] + priors['R'], RH_likelihoods[1] + priors['RH'][0]]),
            logsumexp([HA_likelihoods[0] + priors['H'], HA_likelihoods[1] + priors['HA'][0], RH_likelihoods[-2] + priors['HR'][0]]),
            logsumexp([HA_likelihoods[-1] + priors['A'], HA_likelihoods[-2] + priors['AH'][0]]),
            logsumexp(RH_likelihoods[2:] + priors['RH'][1:]),
            logsumexp(HA_likelihoods[2:] + priors['HA'][1:]),
            logsumexp(RH_likelihoods[:-2] + priors['HR'][1:]),
            logsumexp(HA_likelihoods[:-2] + priors['AH'][1:])
        ])
    
    posteriors = lognormalize(joint_probabilities)
    return posteriors


def mut_type_posteriors(ref, alt, genotype_freq = {'R': 1/4, 'H': 1/2, 'A': 1/4}, mut_prop = 0.5, log_space = False, n_threads = 4): 
    n_cells, n_loci = ref.shape
    # assert(n_loci == alt.shape[0] and n_cells == alt.shape[1])
    # assert(df_ref.index.size == n_loci)
    
    # get priors for each situation 
    priors = composition_priors(n_cells, genotype_freq, mut_prop)

    # multiprocessing
    pool = mp.Pool(n_threads)
    result = [pool.apply_async(locus_posteriors, (ref[:,j], alt[:,j], priors)) for j in range(n_loci)]
    result = np.stack([r.get() for r in result])

    if not log_space:
        result = np.exp(result) # convert back to linear space
    return result


def filter_mutations(ref, alt, genotype_freq = {'R': 1/4, 'H': 1/2, 'A': 1/4}, mut_prop = 0.5, method = 'highest_P', t = None, n_exp = None): 
    '''
    Infer genotypes from matrices of reference and alternative alleles
    Then filter ref and alt according to the posteriors
    '''
    assert(ref.shape == alt.shape)
    posteriors = mut_type_posteriors(ref, alt, genotype_freq, mut_prop)
    # merge mutations of opposite directions
    #posteriors[:,3] += posteriors[:,5]
    #posteriors[:,4] += posteriors[:,6]
    #posteriors = posteriors[:,:5]
    
    # TBC: use match instead of if-elif
    if method == 'highest_P': # for each locus, choose the state with highest posterior
        selected = np.where(np.argmax(posteriors, axis = 1) >= 3)[0]
    elif method == 'threshold': # choose loci at which mutated posterior > threshold 
        selected = np.where(np.sum(posteriors[:,3:], axis = 1) > t)[0]
    elif method == 'first_N': # choose loci with the N highest mutated posteriors
        mut_posteriors = np.sum(posteriors[:,3:], axis = 1)
        order = np.argsort(mut_posteriors)
        selected = order[-n_exp:]
    
    ref_filtered, alt_filtered = ref[:,selected], alt[:,selected]
    mut_type = np.argmax(posteriors[selected, 3:], axis = 1) 
    gt1 = np.choose(mut_type, choices = ['R', 'H', 'H', 'A'])
    gt2 = np.choose(mut_type, choices = ['H', 'A', 'R', 'H'])
    
    return ref_filtered, alt_filtered, gt1, gt2