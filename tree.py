import numpy as np
import graphviz
import copy
from random import shuffle

from tree_node import *
from mutation_detection import read_likelihood, get_likelihoods
from utilities import randint_with_exclude




class CellTree: 
    def __init__(self, n_cells = 1, n_mut = None, randomize = True): 
        self.n_cells = n_cells 
        self.n_mut = n_mut
        self.nodes = [TreeNode(i) for i in range(2 * self.n_cells - 1)]
        self.root = None
        
        self.mutations = [[] for i in range(self.n_nodes)]
        self.L1 = None
        self.L2 = None
        self.LLR = None # log-likelihood ratio of each node
        self.reversible = None
        
        if randomize: 
            self.randomize()
        
    
    @property
    def n_nodes(self): 
        ''' number of nodes '''
        return len(self.nodes)
    
    
    @property
    def likelihood(self): 
        ''' likelihood of the entire tree including mutations '''
        result = 0
        for i in range(self.n_nodes): 
            for mut in self.mutations[i]: 
                result += self.L1[mut] + self.LLR[i, mut]
        return result
    
    
    def clear_mutations(self): 
        self.mutations = [[] for i in range(self.n_nodes)]
    
    
    def copy(self): 
        return copy.deepcopy(self)
    
    
    def copy_structure(self, other): 
        self.nodes = copy.deepcopy(other.nodes)
        self.root = self.nodes[other.root.ID]
        
    
    def random_subtree(self, selected, internal_idx): 
        '''
        Create a random binary tree on selected nodes 
        selected: indices of the selected nodes ("leaves" of the random subtree)
        internal_idx: nodes from this index are used as internal nodes of the subtree
        '''
        # TBC: use a copy of selected instead of changing it directly
        #selected = selected.copy()
        for i in range(len(selected) - 1): 
            children_idx = np.random.choice(selected, size = 2, replace = False)
            # TBC: clear existing children of the parent 
            self.nodes[children_idx[0]].assign_parent(self.nodes[internal_idx])
            self.nodes[children_idx[1]].assign_parent(self.nodes[internal_idx])
            selected.remove(children_idx[0])
            selected.remove(children_idx[1])
            selected.append(internal_idx)
            internal_idx += 1
        
        return internal_idx
    
    
    def randomize(self): 
        '''
        Shuffle the entire tree
        '''
        self.random_subtree([i for i in range(self.n_cells)], self.n_cells)
        self.root = self.nodes[-1]
    
    
    def fit_mutation_tree(self, mutation_tree): 
        mrca = np.ones(mutation_tree.n_nodes, dtype = int) * -1 # most recent common ancestor of cells below a mutation node; -1 means no cell has been found below this mutation
        
        current_idx = self.n_cells
        for mnode in mutation_tree.root.reverse_DFS: 
            need_adoption = [mrca[child.ID] for child in mnode.children if mrca[child.ID] >= 0]
            need_adoption += np.where(mutation_tree.attachments == mnode.ID)[0].tolist()
            # nothing to do if no cell found below
            if len(need_adoption) == 1: # 1 cell found, no internal node added
                mrca[mnode.ID] = need_adoption[0] 
            elif len(need_adoption) > 1: # more than one cell found, add new internal node(s)
                current_idx = self.random_subtree(need_adoption, current_idx)
                mrca[mnode.ID] = current_idx - 1
            
        self.root = self.nodes[-1]
    
    
    def fit_likelihoods(self, likelihoods1, likelihoods2, reversible = None): 
        n_cells, n_mut = likelihoods1.shape
        assert(n_cells == likelihoods2.shape[0] and n_mut == likelihoods2.shape[1])
        if n_cells != self.n_cells or n_mut != self.n_mut: 
            self.__init__(n_cells, n_mut)
        
        self.L1 = np.sum(likelihoods1, axis = 0) # log-likelihood all cells have gt1
        self.L2 = np.sum(likelihoods2, axis = 0) # log-likelihood all cells have gt2
        self.LLR = np.zeros((self.n_nodes, self.n_mut))
        self.LLR[:self.n_cells,:] = likelihoods2 - likelihoods1
        #if reversible is None: 
            #self.reversible = [False] * self.n_mut # if reversibility is not provided, consider all mutations irreversible
        #else: 
            #self.reversible = reversible
    
    
    def refresh_internal_LLR(self):
        '''
        Recalculate the LLR values for internal nodes in a subtree
        node: MRCA of the subtree, if set to root, recalculate LLR for the entire tree
        '''
        for node in self.root.reverse_DFS: 
            if not node.isleaf: # nothing to be done for leaves 
                children_ID = [child.ID for child in node.children]
                self.LLR[node.ID,:] = np.sum(self.LLR[children_ID,:], axis = 0) 
    
    
    def assign_mutations(self): 
        self.mutations = [[] for i in range(self.n_nodes)]
        best_nodes = np.argmax(self.LLR, axis = 0)
        for i in range(self.n_mut): 
            self.mutations[best_nodes[i]].append(i)
    
    
    def fit(self, likelihoods1 = None, likelihoods2 = None, reversible = None, mutation_tree = None): 
        if mutation_tree is not None: 
            self.fit_mutation_tree(mutation_tree)
        if likelihoods1 is not None and likelihoods2 is not None: 
            self.fit_likelihoods(likelihoods1, likelihoods2, reversible)
        self.refresh_internal_LLR()
        self.assign_mutations()
    
    
    def propose_prune_insert(self): 
        subroot = self.nodes[randint_with_exclude(self.n_nodes, [self.root.ID])]

        # insertion to a descendant destroys tree structure
        exclude = [node.ID for node in subroot.DFS]
        # insertion to a sibling doesn't change the tree
        #for sibling in subroot.siblings: 
        #    exlucde.append(sibling.ID)
        # parent of the subtree needs to be pruned as well, since tree is strictly binary
        exclude.append(subroot.parent.ID)

        #suitable_idx = np.delete(np.arange(self.n_nodes), exclude)
        target = self.nodes[randint_with_exclude(self.n_nodes, exclude)]
        for sibling in subroot.siblings: 
            ex_sibling = sibling

        return subroot, target, ex_sibling
    
    
    def prune_insert(self, subroot, target): 
        '''
        Prune a subtree and insert it somewhere else
        subroot: root of the subtree that gets pruned
        target: self.nodes[target_idx] # subtree is inserted to the edge above target
        '''
        # TBC: any more elegant implementation? 
        # the parent of subroot is reused as the new node created when inserting to an edge
        anchor = subroot.parent
        
        for sibling in subroot.siblings: 
            if anchor.isroot: # if pruned at root, sibling becomes new root
                anchor.remove_child(sibling)
                sibling.parent = None
                self.root = sibling 
            else: 
                sibling.assign_parent(anchor.parent)
        
        if target.isroot: # if inserted to original root, anchor becomes new root
            anchor.parent.remove_child(anchor)
            anchor.parent = None
            self.root = anchor
        else: 
            anchor.assign_parent(target.parent) 
        target.assign_parent(anchor)
    
    
    def propose_swap(self): 
        subroot1 = self.nodes[randint_with_exclude(self.n_nodes, [self.root.ID])]

        exclude = [node.ID for node in subroot1.DFS]
        exclude += [node.ID for node in subroot1.ancestors]
        #print(exclude)
        #exclude += [node.ID for node in subroot1.siblings]

        subroot2 = self.nodes[randint_with_exclude(self.n_nodes, exclude)]

        return subroot1, subroot2
    
    
    def subtree_swap(self, subroot1, subroot2): 
        old_parent = subroot1.parent
        subroot1.assign_parent(subroot2.parent)
        subroot2.assign_parent(old_parent)
    
    
    def hill_climb(self, timeout = 1024, convergence = 64, weights = [0.5, 0.5]): 
        n_proposed = 0 # number of failed moves
        n_steps = 0
        likelihood_history = [self.likelihood]
        best_likelihood = likelihood_history[0]

        while n_steps < timeout and n_proposed < convergence: 
            old_dg = self.to_graphviz()
            move_type = np.random.choice(2, p = weights)
            
            if move_type == 0: # prune a subtree & insert elsewhere 
                subroot, target, ex_sibling = self.propose_prune_insert()
                if target is ex_sibling: 
                    continue # inserting above sibling results in same tree 
                n_steps += 1
                self.prune_insert(subroot, target)
                self.fit()
                new_likelihood = self.likelihood
                if new_likelihood > best_likelihood: 
                    best_likelihood = new_likelihood
                    likelihood_history.append(best_likelihood)
                    n_proposed = 0
                else: # if likelihood not better, revert the move
                    self.prune_insert(subroot, ex_sibling) 
                    n_proposed += 1 
            
            else: # move_type == 1, swap two subtrees
                subroot1, subroot2 = self.propose_swap()
                if subroot1 in subroot2.siblings:
                    continue # swapping with sibling results in same tree 
                n_steps += 1
                self.subtree_swap(subroot1, subroot2)
                self.fit()
                new_likelihood = self.likelihood
                if new_likelihood > best_likelihood: 
                    best_likelihood = new_likelihood
                    likelihood_history.append(best_likelihood)
                    n_proposed = 0
                else: # if likelihood not better, revert the move
                    self.subtree_swap(subroot1, subroot2) 
                    n_proposed += 1
        
        # N.B. When a proposal fails, only the tree structure is reverted
        # The LLR and mutation attachments are still that of the proposed tree
        # This doesn't affect the hill-climbing since LLR and mutation attachements 
        # are re-calculated for each new proposal 
        # However, a re-calculation is necessary after hill-climbing is over
        self.fit()
        
        if n_steps == timeout: 
            print('[Cell Tree] Timeout reached.')
        else:
            print('[Cell Tree] Converged after %d steps' % n_steps)

        return likelihood_history
    
    
    def to_graphviz(self, filename = None, show_mutations = True): 
        dg = graphviz.Digraph(filename = filename)
        for node in self.nodes: 
            if node.isleaf: 
                dg.node(str(node.ID), shape = 'circle')
            else: 
                dg.node(str(node.ID), shape = 'circle', style = 'filled', color = 'gray')
            
            if show_mutations and self.mutations[node.ID]: 
                # TBC: test show_mutations outside the loop to improve efficiency
                label = 'm' + ', '.join([str(m) for m in self.mutations[node.ID]])
            else: 
                label = ''
            
            if node.isroot: 
                dg.node('dummy', label = '', shape = 'point')
                dg.edge('dummy', str(node.ID), label = label)
            else: 
                dg.edge(str(node.parent.ID), str(node.ID), label = label)
        
        return dg




class MutationTree:     
    def __init__(self, cell_tree): 
        self.fit_cell_tree(cell_tree)
        
        self.likelihoods = None
        self.attachments = None
    
    
    @property
    def n_nodes(self): 
        return self.n_mut + 1
    
    
    @property
    def root(self): 
        return self.nodes[-1]
    
    
    @property
    def likelihood(self): 
        return np.sum([self.likelihoods[i, self.attachments[i]] for i in range(self.n_cells)])
    
    
    def fit_cell_tree(self, cell_tree): 
        self.n_mut = cell_tree.n_mut
        self.n_cells = cell_tree.n_cells
        
        self.nodes = [TreeNode(i) for i in range(self.n_nodes)]
        
        mrm = np.empty(cell_tree.n_nodes, dtype = int) # mrm for "most recent mutation"
        mrm[cell_tree.root.ID] = self.root.ID # start with "root mutation", which represents wildtype
        cell_tree.root.parent = cell_tree.root # temporary change so that no need to constantly check cnode.isroot
        
        for cnode in cell_tree.root.DFS:  # cnode for "cell node"
            mut_idx = cell_tree.mutations[cnode.ID].copy() 
            parent_mut = mrm[cnode.parent.ID]
            if mut_idx: 
                shuffle(mut_idx)
                self.nodes[mut_idx[0]].assign_parent(self.nodes[parent_mut])
                for idx1, idx2 in zip(mut_idx, mut_idx[1:]): 
                    self.nodes[idx2].assign_parent(self.nodes[idx1])
                mrm[cnode.ID] = mut_idx[-1]
            else: 
                mrm[cnode.ID] = mrm[cnode.parent.ID]
                
        cell_tree.root.parent = None # revert the temporary change
    
    
    def fit_likelihoods(self, likelihoods1, likelihoods2): 
        self.n_cells = likelihoods1.shape[0]
        self.n_mut = likelihoods1.shape[1]
        self.LLR = likelihoods2 - likelihoods1 # log-likelihood ratio for single mutations
        self.likelihoods = np.empty((self.n_cells, self.n_nodes)) # likelihood of cell i attached to node j
        self.likelihoods[:,-1] = np.sum(likelihoods1, axis = 1)
        
    
    def refresh_likelihoods(self): 
        for node in self.root.DFS_without_self: 
            self.likelihoods[:,node.ID] = self.likelihoods[:,node.parent.ID] + self.LLR[:, node.ID]
    
    
    def attach_cells(self): 
        self.attachments = np.argmax(self.likelihoods, axis = 1)
        # TBC: use other data structures, e.g. a list of lists, or a boolean matrix
    
    
    def fit(self, likelihoods1 = None, likelihoods2 = None): 
        if likelihoods1 is not None and likelihoods2 is not None: 
            self.fit_likelihoods(likelihoods1, likelihoods2)
        self.refresh_likelihoods()
        self.attach_cells()
    
    
    def propose_prune_attach(self): 
        # If root has one child and it is pruned, there will be no suitable node to attach
        if len(self.root.children) == 1: 
            subroot_idx = randint_with_exclude(self.n_mut, [self.root.children[0].ID])
        else: 
            subroot_idx = np.random.randint(self.n_mut)
        subroot = self.nodes[subroot_idx]
        ex_parent = subroot.parent # needed to revert the move
        
        # attaching to a descendant violates tree structure
        exclude = [node.ID for node in subroot.DFS]
        # attaching to original parent does not change the tree
        exclude.append(ex_parent.ID)
        target = self.nodes[randint_with_exclude(self.n_nodes, exclude)]
        
        return subroot, target, ex_parent
    
    
    def prune_attach(self, subroot, target): 
        subroot.assign_parent(target)
    
    
    def propose_node_swap(self): 
        node1 = self.nodes[np.random.randint(self.n_mut)]
        
        # the two nodes must be different
        exclude = [node1.ID]
        # swapping with a sibling does not change the tree
        exclude += [sibling.ID for sibling in node1.siblings]
        
        node2 = self.nodes[randint_with_exclude(self.n_mut, exclude)]
        
        return node1, node2
    
    
    def node_swap(self, node1, node2): 
        self.nodes[node1.ID], self.nodes[node2.ID] = self.nodes[node2.ID], self.nodes[node1.ID]
        node1.ID, node2.ID = node2.ID, node1.ID
        
        
    #def propose_subtree_swap(self): 
    #    if len(self.root.children) == 1: 
    #        subroot_idx = randint_with_exclude(self.n_mut, [self.root.children[0].ID])
    #    else: 
    #        subroot_idx = np.random.randint(self.n_mut)
    #    subroot1 = self.nodes[subroot_idx]
        
    
    #def subtree_swap(self, subroot1, subroot2): 
    #    parent1 = subroot1.parent
    #    parent2 = subroot2.parent
    #    subroot1.assign_parent(parent2)
    #    subroot2.assign_parent(parent1)
    
    
    def hill_climb(self, timeout = 1024, convergence = 64, weights = [0.5, 0.5]): 
        n_proposed = 0 # number of failed moves
        n_steps = 0
        likelihood_history = [self.likelihood]
        best_likelihood = likelihood_history[0]
        
        while n_steps < timeout and n_proposed < convergence: 
            n_steps += 1
            move_type = np.random.choice(2, p = weights)
            
            if move_type == 0: # prune a subtree & attach to another node 
                subroot, target, ex_parent = self.propose_prune_attach()
                self.prune_attach(subroot, target)
                self.fit()
                new_likelihood = self.likelihood
                if new_likelihood > best_likelihood: 
                    best_likelihood = new_likelihood
                    likelihood_history.append(best_likelihood)
                    n_proposed = 0
                else: # if likelihood not better, revert the move
                    self.prune_attach(subroot, ex_parent) 
                    n_proposed += 1 
                    
            else: # move_type == 1, swap two nodes
                node1, node2 = self.propose_node_swap()
                self.node_swap(node1, node2)
                self.fit()
                new_likelihood = self.likelihood
                if new_likelihood > best_likelihood: 
                    best_likelihood = new_likelihood
                    likelihood_history.append(best_likelihood)
                    n_proposed = 0
                else: # if likelihood not better, revert the move
                    self.node_swap(node1, node2) 
                    n_proposed += 1
        
        # N.B. When a proposal fails, only the tree structure is reverted
        # The likelihoods and cell attachments are still that of the proposed tree
        # This doesn't affect the hill-climbing since likelihoods and cell attachements 
        # are re-calculated for each new proposal 
        # However, a re-calculation is necessary after hill-climbing is over
        self.fit()
        
        if n_steps == timeout: 
            print('[Mutation Tree] Timeout reached.')
        else:
            print('[Mutation Tree] Converged after %d steps' % n_steps)

        return likelihood_history
    
    
    def to_graphviz(self, filename = None): 
        dg = graphviz.Digraph(filename = filename)
        
        dg.node(str(self.root.ID), label = 'wt', shape = 'rectangle')
        for node in self.nodes[:-1]: 
            dg.node(str(node.ID), shape = 'rectangle')
            dg.edge(str(node.parent.ID), str(node.ID))
        
        if self.attachments is not None: 
            for i in range(self.n_cells): 
                name = 'c' + str(i)
                dg.node(name, shape = 'plaintext')
                # TBC: use undirected edge for cell attachment
                #dg.edge(str(self.attachments[i]), name, dir = 'none')
                dg.edge(str(self.attachments[i]), name)
        
        return dg
    
    
    
