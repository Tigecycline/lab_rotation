from tree import *
from mutation_detection import read_likelihood, get_likelihoods
from utilities import randint_with_exclude



def hill_climb(likelihoods1, likelihoods2, init_tree = None, timeout = 1024, convergence = 64, weights = [0.5, 0.5]): 
    print('Initializing tree...')
    if init_tree is None: 
        tree = CellTree(likelihoods1.shape[0])
    else: 
        tree = init_tree.copy()
    # Nothing to be done for trees with 2 or less leaves, since there is only one possible tree
    assert(tree.n_cells > 2) 
    tree.fit(likelihoods1, likelihoods2)
    likelihood_history = [tree.likelihood]
    
    #print(tree.L1)
    #print(tree.L2)
    #print(tree.LLR)
    
    print('Exploring tree space...')
    n_proposed = 0
    n_steps = 0
    best_likelihood = likelihood_history[0]
    
    while n_steps < timeout and n_proposed < convergence: 
        n_steps += 1
        
        move_type = np.random.choice(2, p = weights)
        if move_type == 0: # prune a subtree & insert elsewhere 
            subroot, target, ex_sibling = propose_prune_insert(tree)
            if target is ex_sibling: 
                continue # inserting above sibling results in same tree 
            tree.prune_insert(subroot, target)
            tree.fit()
            new_likelihood = tree.likelihood
            if new_likelihood > best_likelihood: 
                best_likelihood = new_likelihood
                likelihood_history.append(best_likelihood)
                n_proposed = 0
            else:
                tree.prune_insert(subroot, ex_sibling) # revert the previous move
                n_proposed += 1
        else: # move_type == 1, swap two subtrees
            subroot1, subroot2 = propose_swap(tree)
            if subroot1 in subroot2.siblings:
                continue # swapping with sibling results in same tree 
            tree.subtree_swap(subroot1, subroot2)
            tree.fit()
            new_likelihood = tree.likelihood
            if new_likelihood > best_likelihood: 
                best_likelihood = new_likelihood
                likelihood_history.append(best_likelihood)
                n_proposed = 0
            else:
                tree.subtree_swap(subroot1, subroot2) # revert the previous move
                n_proposed += 1
        
    
    if n_steps == timeout: 
        print('Timeout reached without convergence.')
    else:
        print('Converged after %d steps' % n_steps)
    
    return tree, likelihood_history


def propose_prune_insert(tree): 
        subroot_idx = np.random.choice(tree.n_nodes - 1)
        if subroot_idx >= tree.root.ID: 
            subroot_idx += 1
        subroot = tree.nodes[subroot_idx]

        # insertion to a descendant destroys tree structure
        exclude = [node.ID for node in subroot.DFS]
        # insertion to a sibling doesn't change the tree
        #for sibling in subroot.siblings: 
        #    exlucde.append(sibling.ID)
        # parent of the subtree needs to be pruned as well, since tree is strictly binary
        exclude.append(subroot.parent.ID)
        exclude.sort()

        #suitable_idx = np.delete(np.arange(tree.n_nodes), exclude)
        target = tree.nodes[randint_with_exclude(tree.n_nodes, exclude)]
        for sibling in subroot.siblings: 
            ex_sibling = sibling

        return subroot, target, ex_sibling
    

def propose_swap(tree): 
    non_root_idx = np.delete(np.arange(tree.n_nodes), tree.root.ID) 
    subroot1 = tree.nodes[np.random.choice(non_root_idx)]

    exclude = [node.ID for node in subroot1.DFS]
    exclude += [node.ID for node in subroot1.ancestors]
    #exclude += [node.ID for node in subroot1.siblings]

    subroot2 = tree.nodes[randint_with_exclude(tree.n_nodes, exclude)]
    
    return subroot1, subroot2