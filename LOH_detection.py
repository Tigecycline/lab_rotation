import numpy as np
from tqdm.notebook import tqdm

from mutation_detection import *
from utilities import *



def corr_likelihood(ref1, alt1, ref2, alt2, homo1, homo2): 
    N = ref1.size
    
    likelihoods = np.ones((N+1, N+1)) * (-np.inf) 
    likelihoods[0,0] = 0 # Trivial case when there is 0 cell: number of gt2 cells must be 0
    
    with np.errstate(divide='ignore'): # CONSIDER: other methods to deal with log 0 problem? 
        for n in range(N): 
            likelihoods[n+1, 0] = likelihoods[n, 0] + read_likelihood(ref1[n], alt1[n], 'H') + read_likelihood(ref1[n], alt1[n], 'H')
            k_over_n = np.array([k/(n+1) for k in range(1,n+2)])
            l1 = np.log(1 - k_over_n) + read_likelihood(ref1[n], alt1[n], 'H') + read_likelihood(ref2[n], alt2[n], 'H') + likelihoods[n, 1:n+2]
            l2 = np.log(k_over_n) + read_likelihood(ref1[n], alt1[n], homo1) + read_likelihood(ref2[n], alt2[n], homo2) + likelihoods[n, 0:n+1]
            likelihoods[n+1, 1:n+2] = np.logaddexp(l1, l2)
            
    return logsumexp(likelihoods[N,1:-1] + get_k_mut_priors(N, affect_all = False))


def get_corr_likelihoods(ref, alt, homos): 
    n_loci = ref.shape[0]
    return np.array([corr_likelihood(ref[i,:], alt[i,:], ref[i+1,:], alt[i+1,:], homos[i], homos[i+1]) for i in tqdm(range(n_loci-1))])
    
    
def get_indep_likelihoods(ref, alt, homos): 
    n_loci = ref.shape[0]
    n_cells = ref.shape[1]
    
    k_mut_priors = np.logaddexp(get_k_mut_priors(n_cells, affect_all = False), np.flip(get_k_mut_priors(n_cells, affect_all = False))) - np.log(2)
    
    likelihoods = [logsumexp(likelihood_k_mut(ref[i,:], alt[i,:], 'H', homos[i])[1:-1] + k_mut_priors) for i in tqdm(range(n_loci))]
    
    return np.array([likelihoods[i] + likelihoods[i+1] for i in range(n_loci-1)])
    

def get_corr_posteriors(ref, alt, homos, corr_prior, log_scale = False): 
    n_loci = ref.shape[0]
    n_cells = ref.shape[1]
    
    corr_likelihoods = get_corr_likelihoods(ref, alt, homos)
    indep_likelihoods = get_indep_likelihoods(ref, alt, homos)
    
    corr_joints = corr_likelihoods + np.log(corr_prior)
    indep_joints = indep_likelihoods + np.log(1 - corr_prior)
    result = corr_joints - np.logaddexp(corr_joints, indep_joints)
    
    if log_scale: 
        return result
    else: 
        return np.exp(result)


def get_loss_probabilities(ref, alt, homos, corr_rate = 1/2): 
    corr_posteriors = get_corr_posteriors(ref, alt, homos, corr_rate)
    left_posteriors = np.concatenate(([0], corr_posteriors))
    right_posteriors = np.concatenate((corr_posteriors, [0]))
    
    return 1 - (1 - left_posteriors) * (1 - right_posteriors)
    # return np.max(np.stack((left_posteriors, right_posteriors)), axis = 0)
    

    
    


if __name__ == '__main__': 
    import pandas as pd
    
    ref = pd.read_csv('./Data/glioblastoma_BT_S2/ref.csv', index_col = 0).to_numpy(dtype = float)
    alt = pd.read_csv('./Data/glioblastoma_BT_S2/alt.csv', index_col = 0).to_numpy(dtype = float)
    
    ref1 = ref[4,:]
    alt1 = alt[4,:]
    ref2 = ref[77,:]
    alt2 = alt[77,:]
    
    homo1 = 'R'
    homo2 = 'R'
    
    print(corr_likelihood(ref1, alt1, ref2, alt2, homo1, homo2))
    print(sep_likelihood(ref1, alt1, ref2, alt2, homo1, homo2))
    