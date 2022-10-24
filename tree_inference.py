from tree import *
from scipy.spatial.distance import hamming


def shortest_path_dist(ct1, ct2): 
    return hamming(ct1.dist_matrix, ct2.dist_matrix)


class TreeOptimizer: 
    def __init__(self, convergence_factor = 5, timeout_factor = 20, space = 'b'): 
        self.convergence_factor = convergence_factor
        self.timeout_factor = timeout_factor
        self.space = space # c = cell tree space; m = mutation tree space; b = both        
    
    
    def fit(self, likelihoods1, likelihoods2, sig_dig = 10, reversible = False): 
        ''' 
        In case the mutations are reversible (i.e. mutation direction unknown), 
        the two likelihood matrices will be modified during optimization 
        '''
        assert(likelihoods1.shape == likelihoods2.shape)
        self.n_cells, self.n_mut = likelihoods1.shape
        
        self.likelihoods1, self.likelihoods2 = likelihoods1.copy(), likelihoods2.copy()
        self._decimals = sig_dig - int(np.log10(np.abs(np.sum(likelihoods1)))) # round tree likelihoods to this precision
        
        self.f1t2 = [True] * self.n_mut # direction of mutation j, True if from gt1 to gt2, False otherwise
        
        self.ct = CellTree(self.n_cells, self.n_mut)
        self.ct.randomize()
        self.mt = MutationTree(self.n_cells, self.n_mut)
        
        # Todo: come up with better attribute names (or better implementation)
        self.ct_LLR = np.zeros((self.ct.n_nodes, self.n_mut)) # [i,j] log-likelihood ratio of mutation j attached to edge above cell i in the cell tree
        self.ct_LLR[:self.n_cells,:] = self.likelihoods2 - self.likelihoods1 
        self.ct_L1 = np.sum(self.likelihoods1, axis = 0) # [j] log-likelihood that all cells have gt1 at locus j
        self.ct_L2 = np.sum(self.likelihoods2, axis = 0) # [j] log-likelihood that all cells have gt2 at locus j
        
        self.mt_L = np.empty((self.n_cells, self.mt.n_nodes)) # [i,j] log-likelihood of cell i attached to mutation j in the mutation tree
        self.mt_L[:,self.mt.root.ID] = np.sum(self.likelihoods1, axis = 1)
        
        if reversible == False: 
            self.reversible = []
        elif reversible == True: 
            self.reversible = [j for j in range(self.n_mut)]
        else: 
            self.reversible = reversible
        
        self.update_ct()
        self.mt.fit_structure(self.ct)
        self.update_mt()
    
    
    @property
    def ct_joint(self): 
        ''' joint likelihood of the cell tree with attached mutations '''
        # sum seems faster than numpy.sum when passing a list
        result = sum([self.ct_LLR[self.ct.attachments[j], j] for j in range(self.n_mut) if self.ct.attachments[j] >= 0]) + sum(self.ct_L1)
        return round(result, self._decimals)
    
    
    @property
    def mt_joint(self): 
        ''' joint likelihood of the mutation tree with attached cells '''
        result = sum([self.mt_L[i, self.mt.attachments[i]] for i in range(self.n_cells)])
        return round(result, self._decimals)
    
    
    def flip_direction(self, j): 
        self.f1t2[j] = not self.f1t2[j]
        temp = self.likelihoods1[:,j].copy()
        self.likelihoods1[:,j] = self.likelihoods2[:,j]
        self.likelihoods2[:,j] = temp
        self.ct_LLR[:,j] = - self.ct_LLR[:,j]
        self.ct_L1[j], self.ct_L2[j] = self.ct_L2[j], self.ct_L1[j]
    
    
    def update_ct_likelihoods(self): 
        ''' Recalculate ct_LLR according to current structure of cell tree '''
        for node in self.ct.root.reverse_DFS: 
            if node.isleaf: # nothing to be done for leaves
                continue
            children_ID = [child.ID for child in node.children]
            self.ct_LLR[node.ID,:] = np.sum(self.ct_LLR[children_ID,:], axis = 0) 
    
    
    def update_mt_likelihoods(self): 
        ''' Recalculate mt_L according to current structure of mutation tree '''
        DFS = self.mt.root.DFS
        next(DFS) # skip the root, since likelihoods at root are known
        for node in DFS: 
            self.mt_L[:,node.ID] = self.mt_L[:,node.parent.ID] + self.ct_LLR[:self.n_cells, node.ID]
    
    
    def ct_attach_mutations(self): 
        self.ct.attachments = np.argmax(self.ct_LLR, axis = 0) # when not flipped
        
        # for reversible mutations, check if the inverse mutation has better likelihood
        for j in self.reversible: 
            alt_attachment = np.argmin(self.ct_LLR[:,j]) # when flipped
            if self.ct_L1[j] + self.ct_LLR[self.ct.attachments[j], j] < self.ct_L2[j] - self.ct_LLR[alt_attachment, j]: 
                self.flip_direction(j)
                self.ct.attachments[j] = alt_attachment
        
        # if best attachment still worse than L1, we want the mutation to attach nowhere
        for j in range(self.n_mut): 
            if self.ct_LLR[self.ct.attachments[j], j] < 0: 
                # any negative integer represents "outside of the tree", here we us -1
                self.ct.attachments[j] = -1 
    
    
    def mt_attach_cells(self): 
        self.mt.attachments = np.argmax(self.mt_L, axis = 1)
    
    
    def update_ct(self): 
        self.update_ct_likelihoods()
        self.ct_attach_mutations()
    
    
    def update_mt(self): 
        self.update_mt_likelihoods()
        self.mt_attach_cells()
    
    
    def ct_hill_climb(self, convergence, timeout = np.inf, weights = [0.5, 0.5], print_result = True): 
        n_proposed = 0 # number of failed moves
        n_steps = 0
        likelihood_history = [self.ct_joint]
        best_likelihood = likelihood_history[0]
        
        while n_steps < timeout and n_proposed < convergence: 
            n_steps += 1
            move_type = np.random.choice(2, p = weights)
            
            if move_type == 0: # prune a subtree & attach to another node 
                self.ct.random_prune_insert() 
            elif move_type == 1: # swap two subtrees
                self.ct.random_subtree_swap()
            
            self.update_ct()
            new_likelihood = self.ct_joint
            if new_likelihood > best_likelihood: 
                best_likelihood = new_likelihood
                likelihood_history.append(best_likelihood)
                n_proposed = 0
            else: # if likelihood not better, revert the move
                self.ct.undo_move()
                n_proposed += 1
        
        # When a proposal fails, only the tree structure is restored
        # The LLR and mutation attachments are still that of the (failed) proposed tree
        # This is not a problem during hill-climbing since LLR and mutation attachements 
        # are recalculated for the next proposal 
        # However, recalculation is necessary after hill-climbing is over
        self.update_ct()
        
        if print_result: 
            if n_steps == timeout: 
                status = 'timeout'
            else: 
                status = 'convergence'
            print('[Cell Tree Space] %s after %d steps and %d move(s).' % (status, n_steps, len(likelihood_history) - 1))
        
        return likelihood_history
        
        
    def mt_hill_climb(self, convergence, timeout = np.inf, weights = [0.5, 0.5], print_result = True): 
        n_proposed = 0 # number of failed moves
        n_steps = 0
        likelihood_history = [self.mt_joint]
        best_likelihood = likelihood_history[0]
        
        while n_steps < timeout and n_proposed < convergence: 
            n_steps += 1
            move_type = np.random.choice(2, p = weights)
            
            if move_type == 0: # prune a subtree & attach to another node 
                self.mt.random_prune_attach() 
            elif move_type == 1: # swap two nodes
                self.mt.random_node_swap()
            
            self.update_mt()
            new_likelihood = self.mt_joint
            if new_likelihood > best_likelihood: 
                best_likelihood = new_likelihood
                likelihood_history.append(best_likelihood)
                n_proposed = 0
            else: # if likelihood not better, revert the move
                self.mt.undo_move()
                n_proposed += 1
        
        # recalculate likelihoods and cell attachments
        # for the same reason as in self.update_ct_likelihoods
        self.update_mt()
        
        if print_result: 
            if n_steps == timeout: 
                status = 'timeout'
            else: 
                status = 'convergence'
            print('[Mutation Tree Space] %s after %d steps and %d move(s).' % (status, n_steps, len(likelihood_history) - 1))
        
        return likelihood_history
    
    
    def optimize(self, start_space = 0, print_space_result = True, strategy = 'hill climb'): 
        ct_convergence = self.n_cells * self.convergence_factor
        mt_convergence = self.n_mut * self.convergence_factor * 2
        ct_timeout = ct_convergence * self.timeout_factor
        mt_timeout = mt_convergence * self.timeout_factor
        
        n_spaces = 2
        converged = [False] * n_spaces
        
        current_space = start_space 
        likelihood_history = []
        space_history = []
        
        while not all(converged): 
            # 0 = cell tree space, 1 = mutation tree space
            if current_space == 0: 
                new_history = self.ct_hill_climb(convergence = ct_convergence, timeout = ct_timeout, print_result = print_space_result)
                self.mt.fit_structure(self.ct)
                self.mt_L[:,self.mt.root.ID] = np.sum(self.likelihoods1, axis = 1)
                self.update_mt()

            elif current_space == 1: 
                new_history = self.mt_hill_climb(convergence = mt_convergence, timeout = mt_timeout, print_result = print_space_result)
                self.ct.fit_structure(self.mt)
                self.update_ct()
            
            if len(new_history) == 1: 
                converged[current_space] = True
            else: 
                converged[current_space] = False
                
            likelihood_history += new_history
            space_history += [current_space] * len(new_history)

            # swap to the other space
            current_space = (current_space + 1) % n_spaces

        return likelihood_history, space_history