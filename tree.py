import numpy as np
import graphviz
import copy

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
    
    
    def copy(self): 
        return copy.deepcopy(self)
    
    
    def copy_structure(self, other): 
        self.nodes = copy.deepcopy(other.nodes)
        self.root = self.nodes[other.root.ID]
        
    
    def random_subtree(self, nodes_idx, internal_idx): 
        '''
        Create a random binary tree on selected nodes 
        nodes_idx: indices of the selected nodes 
        internal_idx: nodes from this index are used as internal nodes of the subtree
        '''
        remaining = nodes_idx.copy() # indices of nodes without parent
        for i in range(len(nodes_idx) - 1): 
            children_idx = np.random.choice(remaining, size = 2, replace = False)
            # TBC: clear existing children of the parent 
            self.nodes[children_idx[0]].assign_parent(self.nodes[internal_idx])
            self.nodes[children_idx[1]].assign_parent(self.nodes[internal_idx])
            remaining.remove(children_idx[0])
            remaining.remove(children_idx[1])
            remaining.append(internal_idx)
            internal_idx += 1
    
    
    def randomize(self): 
        '''
        Shuffle the entire tree
        '''
        self.random_subtree([i for i in range(self.n_cells)], self.n_cells)
        self.root = self.nodes[-1]
    
    
    def fit_mutation_tree(self, mutation_tree): 
        '''
        Works only when the mutation tree is properly sorted, i.e. children nodes are found after their parents (which should be the case if the mutation tree is generated using the default constructor)
        '''
        mutation_nodes = mutation_tree.nodes.copy()
        leaves = [np.where(mutation_tree.attachments == i)[0].tolist() for i in range(self.L)] # cells attached to each mutation node
        
        current_idx = self.n_cells
        while mutation_nodes: 
            node = mutation_nodes.pop() # unprocessed node with highest index, all its children should be processed already
            n_leaves = len(leaves[node.ID])
            if n_leaves == 0: 
                pass
            elif n_leaves == 1: 
                leaves[node.parent.ID].append(leaves[node.ID][0]) # pass the child to its grandparent
            else: 
                self.create_random_subtree(leaves[node.ID], current_idx)
                if not node.isroot: 
                    subtree_root = current_idx + n_leaves - 2
                    leaves[node.parent.ID].append(subtree_root)
                    current_idx = subtree_root + 1 
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
    
    
    def refresh_internal_LLR(self, node = None):
        '''
        Recalculate the LLR values for internal nodes in a subtree
        node: MRCA of the subtree, if set to root, recalculate LLR for the entire tree
        '''
        if node is None: 
            node = self.root
        if not node.isleaf: # nothing to be done for leaves 
            self.LLR[node.ID,:] = 0 # erase old LLR 
            for child in node.children: 
                self.refresh_internal_LLR(child) 
                self.LLR[node.ID,:] += self.LLR[child.ID,:] 
    
    
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
        for sibling in subroot.siblings: 
            sibling.assign_parent(subroot.grand_parent)
            if sibling.parent is None: 
                self.root = sibling
        subroot.parent.assign_parent(target.parent) 
        if target.parent is None: 
            self.root = subroot.parent
        target.assign_parent(subroot.parent)
    
    
    def subtree_swap(self, subroot1, subroot2): 
        old_parent = subroot1.parent
        subroot1.assign_parent(subroot2.parent)
        subroot2.assign_parent(old_parent)
    
    
    def to_graphviz(self, filename = None, show = False): 
        dg = graphviz.Digraph(filename = filename)
        for node in self.nodes: 
            if node.isleaf: 
                dg.node(str(node.ID), shape = 'circle')
            else: 
                dg.node(str(node.ID), shape = 'circle', style = 'filled', color = 'gray')
            
            label = ', '.join(['m' + str(m) for m in self.mutations[node.ID]])
            if node.isroot: 
                dg.node('dummy', label = '', shape = 'point')
                dg.edge('dummy', str(node.ID), label = label)
            else: 
                dg.edge(str(node.parent.ID), str(node.ID), label = label)
        
        if show: 
            dg.view()
        
        return dg




class MutationTree:     
    def __init__(self, n_cells, cell_tree = None): 
        self.n_cells = n_cells
        self.n_mut = None
        
        if cell_tree is None: # if cell tree not given, put all mutations at one node
            self.nodes = [TreeNode(0), TreeNode(1)]
            self.nodes[1].assign_parent(self.nodes[0])
            self.nodes[1].mutations = [i for i in range(self.M)] 
        else: 
            self.fit_cell_tree(cell_tree)
            self.root.sort()
        
        self.LLR = None
        self.node_likelihoods = None
        self.attachments = None
        
        self.fit_likelihoods(likelihoods1, likelihoods2)
        self.attach_cells()
    
    
    @property
    def n_nodes(self): 
        return len(self.nodes)
    
    
    @property
    def root(self): 
        return self.nodes[0]
    
    
    def __eq__(self, other): 
        return self.root == other.root
    
    
    def fit_cell_tree(self, cell_tree): 
        self.nodes = [TreeNode(0)] + cell_tree.root.create_NED_tree()
        self.nodes[1].assign_parent(self.root)
        for i in range(self.N): 
            self.nodes[i].ID = i
    
    
    def fit_likelihoods(self, likelihoods1, likelihoods2): 
        self.LLR = likelihoods2 - likelihoods1 # log-likelihood ratio for single mutations
        self.likelihoods = np.zeros()
        self.likelihoods[:,0] = np.sum(likelihoods1, axis = 1) # root has no mutation
        for node in self.root.DFS: 
            self.likelihoods[:,node.ID] = self.likelihoods[:,node.parent.ID] + np.sum(sm_LLR[:,node.mutations], axis = 1)
       
    
    def refresh_cell_LLR(self): 
        self.likelihoods = np.zeros((self.L, self.N)) # likelihood of cell i attached to node j
    
    
    def attach_cells(self): 
        self.attachments = np.argmax(self.likelihoods, axis = 1)
        # TBC: use other data structures, e.g. a list of lists, or a boolean matrix
    
    
    def attachment_info(self): 
        print(self.attachments)
        
    
    def to_graphviz(self, filename = None, show = False): 
        dg = graphviz.Digraph(filename = filename, engine = 'neato')
        for node in self.nodes: 
            label = ', '.join(['m' + str(m) for m in node.mutations])
            dg.node(str(node.ID), label = label, shape = 'rectangle')
            for child in node.children:
                dg.edge(str(node.ID), str(child.ID))
        for i in range(self.L): 
            cell_name = 'cell' + str(i)
            dg.node(cell_name, label = str(i), shape = 'plaintext')
            parent = self.nodes[self.attachments[i]]
            dg.edge(str(parent.ID), cell_name)
        if show: 
            dg.view()
        return dg
    
    
    




        
        
        
        
        
        

if __name__ == '__main__': 
    #import pandas as pd
    
    #ref = pd.read_csv('./Data/glioblastoma_BT_S2/ref.csv', index_col = 0).to_numpy(dtype = float)
    #alt = pd.read_csv('./Data/glioblastoma_BT_S2/alt.csv', index_col = 0).to_numpy(dtype = float)
    '''
    likelihoods1 = np.log(np.array([[1/2, 1/2], 
                                    [1/4, 1/2], 
                                    [1/4, 1/4]]))

    likelihoods2 = np.log(np.array([[1/4, 1/4], 
                                    [1/2, 1/4], 
                                    [1/2, 1/2]]))
    
    gt1 = ['H', 'H']
    gt2 = ['R', 'R']
    
    ct = CellTree(likelihoods1, likelihoods2, gt1, gt2)
    ct.node_info()
    
    print('**********************************')
    mt = MutationTree(likelihoods1, likelihoods2, ct)b
    mt.node_info()
    mt.attachment_info()
    print()
    
    print('**********************************')
    ct.read_mutation_tree(mt)
    ct.node_info()
    '''
    
    mt = MutationTree()
    
    likelihoods1 = np.log(np.array([[1/2, 1/2], 
                                    [1/4, 1/2], 
                                    [1/4, 1/4]]))

    likelihoods2 = np.log(np.array([[1/4, 1/4], 
                                    [1/2, 1/4], 
                                    [1/2, 1/2]]))
    
    gt1 = ['H', 'H']
    gt2 = ['R', 'R']
    
    ct = CellTree(likelihoods1, likelihoods2, gt1, gt2)
    
    ct.nodes = [TreeNode(i) for i in range(16)]
    ct.L = 8
    ct.read_mutation_tree(mt)
    
    ct.node_info()