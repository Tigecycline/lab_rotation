import numpy as np
import graphviz
from tree_node import TreeNode




class CellTree: 
    def __init__(self, likelihoods1, likelihoods2, gt1, gt2, reversible = None): 
        self.L = likelihoods1.shape[0] # number of leaves (observed cells)
        self.M = likelihoods1.shape[1] # number of mutations
        self.nodes = [TreeNode(i) for i in range(self.N)] # TBC: use dictionary instead
        self.gt1 = gt1 
        self.gt2 = gt2 
        self.L1 = np.sum(likelihoods1, axis = 0) # log-likelihood all cells have gt1
        self.L2 = np.sum(likelihoods2, axis = 0) # log-likelihood all cells have gt2
        self.LLR = np.zeros((self.N, self.M)) # log-likelihood ratio of each node
        self.LLR[:self.L,:] = likelihoods2 - likelihoods1
        
        if reversible is None: # if reversibility not given, consider all mutations as irreversible
            self.reversible = [False] * self.M 
        else: 
            self.reversible = reversible 
        
        self.refresh_edges()
        self.refresh_LLR(self.root)
        self.refresh_mutations()
    
    
    @property
    def N(self): 
        ''' number of nodes '''
        return 2*self.L
    
    
    @property
    def root(self): 
        return self.nodes[-1]
    
    
    def create_random_subtree(self, nodes_idx, new_idx): 
        '''
        Create a random binary tree on selected nodes 
        nodes_idx: indices of the selected nodes 
        new_idx: nodes from this index are used as internal nodes of the subtree
        '''
        remaining = nodes_idx # indices of nodes without parent
        parent_idx = new_idx
        for i in range(len(nodes_idx) - 1): 
            children_idx = np.random.choice(remaining, size = 2, replace = False)
            # TBC: clear existing children of the parent 
            self.nodes[children_idx[0]].assign_parent(self.nodes[parent_idx])
            self.nodes[children_idx[1]].assign_parent(self.nodes[parent_idx])
            remaining.remove(children_idx[0])
            remaining.remove(children_idx[1])
            remaining.append(parent_idx)
            parent_idx += 1
    
    
    def read_mutation_tree(self, mutation_tree): 
        '''
        Works only when the mutation tree is properly sorted, i.e. children nodes are found after their parents (which should be the case if the mutation tree is generated using the default constructor)
        '''
        mutation_nodes = mutation_tree.nodes.copy()
        leaves = [np.where(mutation_tree.attachments == i)[0].tolist() for i in range(self.L)] # attached cells & other internal nodes
        print(leaves)
        
        idx = self.L
        while mutation_nodes: 
            node = mutation_nodes.pop() # unprocessed node with highest index, all its children should be processed already
            print('Processing node', node.name)
            print('The leaves are:', leaves[node.name])
            n_leaves = len(leaves[node.name])
            print(node)
            if n_leaves == 0: 
                pass
            elif n_leaves == 1: 
                leaves[node.parent.name].append(leaves[node.name][0]) # pass the child to its grandparent
            else: 
                self.create_random_subtree(leaves[node.name], idx)
                if not node.isroot: 
                    subtree_root = idx + n_leaves - 2
                    leaves[node.parent.name].append(subtree_root)
                    idx = subtree_root + 1 
        self.nodes[-2].assign_parent(self.root)
    
    
    def refresh_edges(self, mutation_tree = None): 
        if mutation_tree is None: # random tree if no mutation tree is provided
            self.create_random_subtree([i for i in range(self.L)], self.L)
            self.nodes[-2].assign_parent(self.root)
        else: 
            self.read_mutation_tree(mutation_tree)
    
    
    def refresh_LLR(self, node): 
        if not node.isleaf: # nothing to be done for leaves
            self.LLR[node.name,:] = 0 # erase old LLR 
            for child in node.children: 
                self.refresh_LLR(child)
                self.LLR[node.name,:] += self.LLR[child.name,:]
    
    
    def refresh_mutations(self): 
        for node in self.nodes: 
            node.clear_mutations()
        best_nodes = np.argmax(self.LLR, axis = 0)
        for i in range(self.M): 
            self.nodes[best_nodes[i]].add_mutation(i)
    
    
    def move_subtree(self, mrca_idx, target_idx): 
        # Todo: test this function
        assert(mrca_idx < self.N - 1 and target_idx < self.N - 1)
        mrca = self.nodes[mrca_idx]
        target = self.nodes[target_idx]
        assert(mrca is not self.root and mrca is not target)
        assert(not target.descends_from(mrca))
        
        
    
    
    def node_info(self): 
        for node in self.nodes: 
            print(node)
    
    
    def to_graphviz(self, filename = None, show = False): 
        dg = graphviz.Digraph(filename = filename)
        for node in self.nodes: 
            if node.isleaf: 
                dg.node(str(node.name), shape = 'circle')
            else: 
                dg.node(str(node.name), label = '', shape = 'circle')
            for child in node.children: 
                label = ', '.join(['m' + str(m) for m in child.mutations])
                dg.edge(str(node.name), str(child.name), label = label)
        if show: 
            dg.view()
        return dg




class MutationTree:     
    def __init__(self, likelihoods1 = None, likelihoods2 = None, cell_tree = None): 
        # \begin {for test}
        if likelihoods1 is None: 
            self.L = 8
            
            self.nodes = [TreeNode(i) for i in range(7)]
            self.nodes[1].assign_parent(self.nodes[0])
            self.nodes[2].assign_parent(self.nodes[0])
            self.nodes[3].assign_parent(self.nodes[0])
            self.nodes[4].assign_parent(self.nodes[1])
            self.nodes[5].assign_parent(self.nodes[2])
            self.nodes[6].assign_parent(self.nodes[2])
            
            self.nodes[1].mutations = [0,1]
            self.nodes[2].mutations = [2]
            self.nodes[3].mutations = [3]
            self.nodes[4].mutations = [4]
            self.nodes[5].mutations = [5]
            self.nodes[6].mutations = [6]
            
            self.attachments = np.array([1,1,0,5,5,5,3,2], dtype = int)
            return
            
        # \end {for test}
        self.L = likelihoods1.shape[0] # number of leaves (observed cells)
        self.M = likelihoods1.shape[1] # number of mutations
        
        if cell_tree is None: 
            self.nodes = [TreeNode(0), TreeNode(1)]
            self.nodes[1].mutations = [i for i in range(self.M)] # all mutations at one node
        else: 
            self.read_cell_tree(cell_tree)
            self.sort()
        
        self.refresh_likelihoods(likelihoods1, likelihoods2)
        self.attach_cells()
    
    
    @property
    def N(self): 
        return len(self.nodes)
    
    
    @property
    def root(self): 
        return self.nodes[0]
    
    
    def __eq__(self, other): 
        return self.root == other.root
    
    
    def sort(self): 
        self.root.sort()
    
    
    def read_cell_tree(self, cell_tree): 
        self.nodes = cell_tree.root.create_NED_tree()
        for i in range(self.N): 
            self.nodes[i].name = i
    
    
    def refresh_likelihoods(self, likelihoods1, likelihoods2): 
        sm_LLR = likelihoods2 - likelihoods1 # log-likelihood ratio for single mutations
        self.likelihoods = np.zeros((self.L, self.N)) # likelihood of cell i attached to node j
        self.likelihoods[:,0] = np.sum(likelihoods1, axis = 1) # root has no mutation
        # The construction of the tree ensures that, in self.nodes, children are always found after their parents. So we can traverse the nodes using the order in self.nodes. 
        for node in self.nodes[1:]: 
            self.likelihoods[:,node.name] = self.likelihoods[:,node.parent.name] + np.sum(sm_LLR[:,node.mutations], axis = 1)
        
    
    def attach_cells(self): 
        self.attachments = np.argmax(self.likelihoods, axis = 1)
        # TBC: use other data structures, e.g. a list of lists, or a boolean matrix
    
    
    def node_info(self): 
        for node in self.nodes: 
            print(node)
    
    
    def attachment_info(self): 
        print(self.attachments)
        
    
    def to_graphviz(self, filename = None, show = False): 
        dg = graphviz.Digraph(filename = filename, engine = 'neato')
        for node in self.nodes: 
            label = ', '.join(['m' + str(m) for m in node.mutations])
            dg.node(str(node.name), label = label, shape = 'rectangle')
            for child in node.children:
                dg.edge(str(node.name), str(child.name))
        for i in range(self.L): 
            cell_name = 'cell' + str(i)
            dg.node(cell_name, label = str(i), shape = 'plaintext')
            parent = self.nodes[self.attachments[i]]
            dg.edge(str(parent.name), cell_name)
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