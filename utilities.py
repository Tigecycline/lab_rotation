from scipy.special import loggamma
import numpy as np

def logbinom(n, k): 
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)

def log_sum_of_exp(lst): 
    result = lst[0]
    for i in range(1, len(lst)): 
        result = np.logaddexp(lst[i], result)
    return result