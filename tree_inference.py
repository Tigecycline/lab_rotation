from tree import *
from mutation_detection import read_likelihood



def hill_climb(ref, alt, gt1, gt2, init_tree = None, timeout = 256, convergence = 32, weights = [0.5, 0.5]): 
    print('Calculating likelihoods...')
    n_cells = ref.shape[0]
    n_mut = ref.shape[1]
    likelihoods1 = np.empty((n_cells, n_mut))
    likelihoods2 = np.empty((n_cells, n_mut))
    for i in range(n_cells): 
        for j in range(n_mut): 
            likelihoods1[i,j] = read_likelihood(ref[i,j], alt[i,j], gt1[j])
            likelihoods2[i,j] = read_likelihood(ref[i,j], alt[i,j], gt2[j])
    
    print('Initializing tree...')
    if init_tree is None: 
        tree = CellTree(likelihoods1.shape[0])
    else: 
        tree = init_tree.copy()
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
    non_root_idx = np.delete(np.arange(tree.n_nodes), tree.root.ID) 
    subroot = tree.nodes[np.random.choice(non_root_idx)]

    # insertion above descendant destroys tree structure
    exclude = [node.ID for node in subroot.DFS] 
    # insertion above a sibling makes no sense since tree is not changed
    #exclude += [sibling.ID for sibling in subroot.siblings] 
    # parent no longer exists after pruning since tree is strictly binary
    exclude.append(subroot.parent.ID) 

    suitable_idx = np.delete(np.arange(tree.n_nodes), exclude)
    target = tree.nodes[np.random.choice(suitable_idx)]
    for sibling in subroot.siblings: 
        ex_sibling = sibling
    
    return subroot, target, ex_sibling
    

def propose_swap(tree): 
    non_root_idx = np.delete(np.arange(tree.n_nodes), tree.root.ID) 
    subroot1 = tree.nodes[np.random.choice(non_root_idx)]

    exclude = [node.ID for node in subroot1.DFS]
    exclude += [node.ID for node in subroot1.ancestors]
    #exclude += [node.ID for node in subroot1.siblings]
    suitable_idx = np.delete(np.arange(tree.n_nodes), exclude)

    subroot2 = tree.nodes[np.random.choice(suitable_idx)]
    
    return subroot1, subroot2