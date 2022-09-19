import numpy as np
import graphviz
import copy
from random import shuffle

from tree_node import *




class CellTree: 
    def __init__(self, n_cells, n_mut = None, mutation_tree = None): 
        self.n_cells = n_cells 
        self.n_mut = n_mut
        self.nodes = [TreeNode(i) for i in range(2 * self.n_cells - 1)]
        self.root = None
        
        self.mutations = [[] for i in range(self.n_nodes)]
        self.L1 = None
        self.L2 = None
        self.LLR = None # log-likelihood ratio of each node
        self.reversible = None
        
        if mutation_tree is not None: 
            self.fit_mutation_tree(mutation_tree)
        else: 
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
        assert(self.n_cells == likelihoods1.shape[0] and self.n_cells == likelihoods2.shape[0])
        
        if self.n_mut is None: 
            self.n_mut = likelihoods1.shape[1]
            assert(self.n_mut == likelihoods2.shape[1])
        else: 
            assert(self.n_mut == likelihoods1.shape[1] and self.n_mut == likelihoods2.shape[1])
        
        self.L1 = np.sum(likelihoods1, axis = 0) # log-likelihood all cells have gt1
        self.L2 = np.sum(likelihoods2, axis = 0) # log-likelihood all cells have gt2
        self.LLR = np.zeros((self.n_nodes, self.n_mut))
        self.LLR[:self.n_cells,:] = likelihoods2 - likelihoods1
        if reversible is None: 
            self.reversible = [False] * self.n_mut # if reversibility is not provided, consider all mutations irreversible
        else: 
            self.reversible = reversible
    
    
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
    
    
    def subtree_swap(self, subroot1, subroot2): 
        old_parent = subroot1.parent
        subroot1.assign_parent(subroot2.parent)
        subroot2.assign_parent(old_parent)
    
    
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
    
    
    def fit_cell_tree(self, cell_tree): 
        self.n_mut = cell_tree.n_mut
        self.n_cells = cell_tree.n_cells
        
        self.nodes = [TreeNode(i) for i in range(self.n_nodes)]
        
        most_recent = np.empty(cell_tree.n_nodes, dtype = int) # index of the youngest mutation that a cell node contains
        most_recent[cell_tree.root.ID] = self.root.ID # start with "root mutation", which represents wildtype
        cell_tree.root.parent = cell_tree.root # temporary change so that no need to constantly check cnode.isroot
        
        for cnode in cell_tree.root.DFS:  # cnode for "cell node"
            mut_idx = cell_tree.mutations[cnode.ID].copy() 
            parent_mut = most_recent[cnode.parent.ID]
            if mut_idx: 
                shuffle(mut_idx)
                self.nodes[mut_idx[0]].assign_parent(self.nodes[parent_mut])
                for idx1, idx2 in zip(mut_idx, mut_idx[1:]): 
                    self.nodes[idx2].assign_parent(self.nodes[idx1])
                most_recent[cnode.ID] = mut_idx[-1]
            else: 
                most_recent[cnode.ID] = most_recent[cnode.parent.ID]
                
        cell_tree.root.parent = None # revert the temporary change
    
    
    def fit_likelihoods(self, likelihoods1, likelihoods2): 
        LLR = likelihoods2 - likelihoods1 # log-likelihood ratio for single mutations
        self.likelihoods = np.empty((self.n_cells, self.n_nodes)) # likelihood of cell i attached to node j
        self.likelihoods[:,-1] = np.sum(likelihoods1, axis = 1)
        for node in self.root.DFS_without_self: 
            self.likelihoods[:,node.ID] = self.likelihoods[:,node.parent.ID] + LLR[:, node.ID]
    
    
    def attach_cells(self): 
        self.attachments = np.argmax(self.likelihoods, axis = 1)
        # TBC: use other data structures, e.g. a list of lists, or a boolean matrix
    
    
    def prune_attach(self, subroot, target): 
        subroot.assign_parent(target)
    
    
    def node_swap(self, node1, node2): 
        self.nodes[node1.ID], self.nodes[node2.ID] = self.nodes[node2.ID], self.nodes[node1.ID]
        node1.ID, node2.ID = node2.ID, node1.ID
        
    
    def subtree_swap(self, subroot1, subroot2): 
        parent1 = subroot1.parent
        parent2 = subroot2.parent
        subroot1.assign_parent(parent2)
        subroot2.assign_parent(parent1)
    
    
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
    
    
    
