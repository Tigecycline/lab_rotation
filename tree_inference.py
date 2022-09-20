from tree import *


def optimize_tree(likelihoods1, likelihoods2): 
    ct = CellTree()
    ct.fit(likelihoods1, likelihoods2)
    
    mt = MutationTree(ct)
    mt.fit(likelihoods1, likelihoods2)
    
    ct_converged = False
    mt_converged = False
    
    current_space = 0 # 0 = cell tree, 1 = mutation tree
    likelihood_history = []
    space_history = []
    
    while not ct_converged or not mt_converged: 
        if current_space == 0: 
            ct.fit_mutation_tree(mt)
            new_history = ct.hill_climb()
            if len(new_history) == 1: 
                ct_converged = True
            else: 
                ct_converged = False
            
        elif current_space == 1: 
            mt.fit_cell_tree(ct)
            new_history = mt.hill_climb()
            if len(new_history) == 1: 
                mt_converged = True
            else: 
                mt_converged = False
        
        likelihood_history += new_history
        space_history += [current_space] * len(new_history)
        
        # swap to the other space
        if current_space == 0: 
            current_space = 1
        else: 
            current_space = 0
    
    return ct, mt, likelihood_history, space_history