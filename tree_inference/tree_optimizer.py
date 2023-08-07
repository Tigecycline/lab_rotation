from .tree import *
from .mutation_detection import *
#from .LOH_detection import *




class TreeOptimizer:
    def __init__(self, convergence_factor = 5, timeout_factor = 20): 
        self.convergence_factor = convergence_factor
        self.timeout_factor = timeout_factor
    
    
    def fit(self, likelihoods1, likelihoods2, sig_dig = 10, reversible = False): 
        '''
        Gets ready to run optimization using provided likelihoods.
        Before calling this function, it is assumed that for each locus, the two genotypes gt1 and gt2 are known.
        [Arguments]
            likelihoods1: 2D array in which entry [i,j] represents the likelihood of cell i having gt1 at locus j
            likelihoods2: 2D array in which entry [i,j] represents the likelihood of cell i having gt2 at locus j
            sig_dig: number of significant digits to use when calculating joint probability
            reversible: either an array that indicates whether each of the mutations is reversible (i.e. direction unknown),
                or a boolean value that applies to all loci.
                When a mutation is not reversible, the direction is assumed to be from gt1 to gt2.
        '''
        assert(likelihoods1.shape == likelihoods2.shape)
        self.n_cells, self.n_mut = likelihoods1.shape
        if self.n_cells < 3 or self.n_mut < 2:
            print('[TreeOptimizer.fit] ERROR: cell tree / mutation tree too small, nothing to explore')
            return
        
        self.likelihoods1, self.likelihoods2 = likelihoods1.copy(), likelihoods2.copy()
        self._decimals = sig_dig - int(np.log10(np.abs(np.sum(likelihoods1)))) # round tree likelihoods to this precision
        # Need to round because numpy.sum can give slightly different results when summing a matrix along different axis
        # See the "Notes" part in documentation: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # If not rounded, the likelihood might increase when converting between the two tree spaces
        # Sometimes this traps the optimization in an infinite search
        
        self.f1t2 = [True] * self.n_mut # direction of mutation j, True if from gt1 to gt2, False otherwise
        
        self.ct = CellTree(self.n_cells)
        self.ct.randomize()
        self.mt = MutationTree(self.n_mut)
        
        # TBC: come up with better attribute names (or better implementation)
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
        else: # list / array
            self.reversible = reversible
        
        # adapt mt to ct, not necessary for optimization
        self.update_ct()
        self.mt_L[:,self.mt.root.ID] = np.sum(self.likelihoods1, axis = 1) # need this because update of ct can change root likelihood
        self.mt.fit_structure(self.ct)
        self.update_mt()
    
    
    @property
    def ct_joint(self): 
        ''' joint likelihood of the cell tree, with attached mutations '''
        # sum seems faster than numpy.sum when passing a list
        result = sum([self.ct_LLR[self.ct.attachments[j], j] for j in range(self.n_mut) if self.ct.attachments[j] >= 0]) + sum(self.ct_L1)
        return round(result, self._decimals)
    
    
    @property
    def mt_joint(self): 
        ''' joint likelihood of the mutation tree, with attached cells '''
        result = sum([self.mt_L[i, self.mt.attachments[i]] for i in range(self.n_cells)])
        return round(result, self._decimals)
    
    
    @property
    def ct_mean_likelihood(self):
        return self.ct_joint / self.likelihoods1.size
    
    
    @property
    def mt_mean_likelihood(self):
        return self.mt_joint / self.likelihoods1.size
    
    
    @property
    def mean_history(self):
        return self.likelihood_history / self.likelihoods1.size
    
    
    def flip_direction(self, j):
        ''' Reverses the direction of a mutation j by exchanging corresponding columns in likelihoods1 and likelihoods2 '''
        self.f1t2[j] = not self.f1t2[j]
        temp = self.likelihoods1[:,j].copy()
        self.likelihoods1[:,j] = self.likelihoods2[:,j]
        self.likelihoods2[:,j] = temp
        self.ct_LLR[:,j] = - self.ct_LLR[:,j]
        self.ct_L1[j], self.ct_L2[j] = self.ct_L2[j], self.ct_L1[j]
    
    
    def update_ct_likelihoods(self):
        ''' Recalculates ct_LLR according to current structure of cell tree '''
        for node in self.ct.root.reverse_DFS:
            if node.isleaf: # nothing to be done for leaves
                continue
            children_ID = [child.ID for child in node.children]
            self.ct_LLR[node.ID,:] = np.sum(self.ct_LLR[children_ID,:], axis = 0) 
    
    
    def update_mt_likelihoods(self):
        ''' Recalculates mt_L according to current structure of mutation tree '''
        mt_DFS = self.mt.root.DFS
        next(mt_DFS) # skip the root, since likelihoods at root are known
        for node in mt_DFS:
            self.mt_L[:,node.ID] = self.mt_L[:,node.parent.ID] + self.ct_LLR[:self.n_cells, node.ID]
    
    
    def ct_attach_mutations(self):
        '''  '''
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
    
    
    def ct_hill_climb(self, convergence, timeout = np.inf, weights = [0.5, 0.5], print_info = True): 
        ''' Optimzes the cell lineage tree using hill climbing approach '''
        n_proposed = 0 # number of failed moves
        n_steps = 0
        likelihood_history = [self.ct_joint]
        best_likelihood = likelihood_history[0]
        
        while n_steps < timeout and n_proposed < convergence: 
            n_steps += 1
            if print_info:
                print('[TreeOptimizer.ct_hill_climb] step %i/%i' % (n_steps, timeout), end='\r')
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
        
        if print_info: 
            if n_steps == timeout: 
                status = 'timeout'
            else: 
                status = 'convergence'
            print('[Cell Tree Space] %s after %d steps and %d move(s).' % (status, n_steps, len(likelihood_history) - 1))
        
        return likelihood_history
        
        
    def mt_hill_climb(self, convergence, timeout = np.inf, weights = [0.5, 0.5], print_info = True):
        ''' Optimzes the mutation tree using hill climbing approach '''
        n_proposed = 0 # number of failed moves
        n_steps = 0
        likelihood_history = [self.mt_joint]
        best_likelihood = likelihood_history[0]
        
        while n_steps < timeout and n_proposed < convergence: 
            n_steps += 1
            if print_info:
                print('[TreeOptimizer.mt_hill_climb] step %i/%i' % (n_steps, timeout), end='\r')
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
        
        if print_info: 
            if n_steps == timeout: 
                status = 'timeout'
            else: 
                status = 'convergence'
            print('[Mutation Tree Space] %s after %d steps and %d move(s).' % (status, n_steps, len(likelihood_history) - 1))
        
        return likelihood_history
    
    
    def optimize(self, print_info = True, strategy = 'hill climb', spaces = None, n_space_max = None): 
        '''
        Optimizes the tree in all spaces.
        [Arguments]
            strategy: 'hill climb' is the only available option now 
                it accepts moves that (strictly) increase the joint likelihood and rejects everything else 
            spaces: spaces that will be searched and the order of the search
                'c' = cell tree space, 'm' = mutation tree space
                default is ['c','m'], i.e. start with cell tree and search both spaces
            n_space_max: maximal allowed number of space swaps.
        '''
        ct_convergence = self.n_cells * self.convergence_factor
        mt_convergence = self.n_mut * self.convergence_factor * 2
        ct_timeout = ct_convergence * self.timeout_factor
        mt_timeout = mt_convergence * self.timeout_factor
        
        if spaces is None: 
            spaces = ['c','m']
        n_spaces = len(spaces)
        converged = [False] * n_spaces # stop searching when all spaces have converged
        
        self.likelihood_history = [-np.inf]
        self.space_history = []
        space_idx = 0
        
        if n_space_max is None:
            n_space_max = min(self.n_cells, self.n_mut) * n_spaces
        space_counter = 0
        while not all(converged): 
            space_counter += 1
            if space_counter > n_space_max: # monitor and cut infinite loops
                print('[TreeOptimizer.optimize] WARNING: maximal number (%i) of spaces reached' % n_space_max)
                break
            # 0 = cell tree space, 1 = mutation tree space
            current_space = spaces[space_idx]
            if current_space == 'c': 
                new_history = self.ct_hill_climb(convergence = ct_convergence, timeout = ct_timeout, print_info = print_info)
                self.mt.fit_structure(self.ct)
                self.mt_L[:,self.mt.root.ID] = np.sum(self.likelihoods1, axis = 1)
                self.update_mt()

            elif current_space == 'm': 
                new_history = self.mt_hill_climb(convergence = mt_convergence, timeout = mt_timeout, print_info = print_info)
                self.ct.fit_structure(self.mt)
                self.update_ct()
                
            else: 
                print('[TreeOptimizer.optimize] ERROR: invalid space name')
                return
            
            if new_history[-1] == self.likelihood_history[-1]: 
                converged[space_idx] = True
            elif new_history[-1] < self.likelihood_history[-1]:
                print('[TreeOptimizer.optimize] WARNING: likelihood decreased, current space %s, space count %i' % (current_space, n_spaces))
                break
            else:
                converged[space_idx] = False
            
            self.likelihood_history += new_history
            self.space_history += [current_space] * len(new_history)

            # move to the next space
            space_idx = (space_idx + 1) % n_spaces




# TODO?: move to utilities
def mean_likelihood(ct, likelihoods1, likelihoods2):
    ''' Returns the mean_likelihood of a knwon cell tree '''
    optz = TreeOptimizer()
    optz.fit(likelihoods2, likelihoods1, reversible = True)
    optz.ct = ct
    optz.ct.n_mut = optz.n_mut
    optz.update_ct()
    return optz.ct_mean_likelihood