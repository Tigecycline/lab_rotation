from tree import *


def optimize_tree(likelihoods1, likelihoods2, convergence_factor = 5, timeout_factor = 10, start_space = 0): 
    n_cells, n_mut = likelihoods1.shape
    assert(n_cells == likelihoods2.shape[0] and n_mut == likelihoods2.shape[1])
    ct_convergence = n_cells * convergence_factor
    mt_convergence = n_mut * convergence_factor
    ct_timeout = ct_convergence * timeout_factor
    mt_timeout = mt_convergence * timeout_factor
    
    ct = CellTree()
    ct.fit(likelihoods1, likelihoods2)
    
    mt = MutationTree(ct)
    mt.fit(likelihoods1, likelihoods2)
    
    ct_converged = False
    mt_converged = False
    
    current_space = start_space # 0 = cell tree space, 1 = mutation tree space
    likelihood_history = []
    space_history = []
    
    while not ct_converged or not mt_converged: 
        if current_space == 0: 
            new_history = ct.hill_climb(convergence = ct_convergence, timeout = ct_timeout)
            if len(new_history) == 1: 
                ct_converged = True
            else: 
                ct_converged = False
            mt.fit_cell_tree(ct)
            
        elif current_space == 1: 
            new_history = mt.hill_climb(convergence = mt_convergence, timeout = mt_timeout)
            if len(new_history) == 1: 
                mt_converged = True
            else: 
                mt_converged = False
            ct.fit_mutation_tree(mt)
        
        likelihood_history += new_history
        space_history += [current_space] * len(new_history)
        
        # swap to the other space
        if current_space == 0: 
            current_space = 1
        else: 
            current_space = 0
    
    return ct, mt, likelihood_history, space_history