import numpy as np
from tqdm.notebook import tqdm

from mutation_detection import *
from utilities import *




def loss_likelihood(ref1, alt1, ref2, alt2, homo1, homo2): 
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
    
    
def retention_likelihood(ref1, alt1, ref2, alt2, homo1, homo2): 
    N = ref1.size
    priors_single_direction = get_k_mut_priors(N, affect_all = False)
    priors = np.logaddexp(priors_single_direction, np.flip(priors_single_direction)) - np.log(2)
    likelihood1 = logsumexp(likelihood_k_mut(ref1, alt1, 'H', homo1)[1:-1] + priors)
    likelihood2 = logsumexp(likelihood_k_mut(ref2, alt2, 'H', homo2)[1:-1] + priors)
    #print(likelihood1)
    #print(likelihood2)
    
    return likelihood1 + likelihood2
    

def get_loss_posteriors(ref, alt, homos, priors, log_scale = False): 
    n_mutated = len(homos)
    
    result = np.zeros(n_mutated-1)
    
    for i in tqdm(range(n_mutated-1)): 
        loss_joint = loss_likelihood(ref[i,:], alt[i,:], ref[i+1,:], alt[i+1,:], homos[i], homos[i+1]) + priors['loss']
        ret_joint = retention_likelihood(ref[i,:], alt[i,:], ref[i+1,:], alt[i+1,:], homos[i], homos[i+1]) + priors['ret']
        result[i] = loss_joint - np.logaddexp(loss_joint, ret_joint)
    
    if log_scale: 
        return result
    else: 
        return np.exp(result)

    

    
    


if __name__ == '__main__': 
    import pandas as pd
    
    df_ref = pd.read_csv('./Data/glioblastoma_BT_S2/ref.csv', index_col = 0)
    df_alt = pd.read_csv('./Data/glioblastoma_BT_S2/alt.csv', index_col = 0)
    
    ref = df_ref.to_numpy(dtype = float)
    alt = df_alt.to_numpy(dtype = float)
    
    del df_ref
    del df_alt
    
    ref1 = ref[4,:]
    alt1 = alt[4,:]
    ref2 = ref[77,:]
    alt2 = alt[77,:]
    
    homo1 = 'R'
    homo2 = 'R'
    
    print(loss_likelihood(ref1, alt1, ref2, alt2, homo1, homo2))
    print(retention_likelihood(ref1, alt1, ref2, alt2, homo1, homo2))
    