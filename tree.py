import numpy as np
from tree_node import TreeNode




class CellTree: 
    def __init__(self, likelihoods1, likelihoods2, gt1, gt2, reversible = None): 
        self.L = likelihoods1.shape[0] # number of leaves (observed cells)
        self.M = likelihoods1.shape[1] # number of mutations
        self.nodes = [TreeNode(i) for i in range(self.N)]
        self.gt1 = gt1 
        self.gt2 = gt2 
        self.L1 = np.sum(likelihoods1, axis = 0) # log-likelihood all cells have gt1
        self.L2 = np.sum(likelihoods2, axis = 0) # log-likelihood all cells have gt2
        self.LLR = np.zeros((self.N, self.M)) # log-likelihood ratio of each node
        self.LLR[:self.L,:] = likelihoods2 - likelihoods1
        
        if reversible is None: # if reversibility not given, consider all mutations as irreversible
            self.reversible = [False for i in range(self.M)] 
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
    
    
    def refresh_edges(self, mutation_tree = None): 
        if mutation_tree is None: # random tree if no mutation tree is provided
            remaining = [i for i in range(self.L)] # indices of nodes without parent
            for i in range(self.L, self.N - 1): 
                idx = np.random.choice(remaining, size = 2, replace = False)
                self.nodes[i].add_child(self.nodes[idx[0]]) # or leave it as numpy array?
                self.nodes[i].add_child(self.nodes[idx[1]]) 
                remaining.remove(idx[0])
                remaining.remove(idx[1])
                remaining.append(i)
            self.root.add_child(self.nodes[-2])
        else: 
            pass # TO BE IMPLEMENTED
    
    
    def refresh_LLR(self, node): 
        if not node.isleaf: # nothing to be done for leaves
            self.LLR[node.name,:] = 0 # erase old LLR (if exists)
            for child in node.get_children(): # DF traverse of all descendants
                self.refresh_LLR(child)
                self.LLR[node.name,:] += self.LLR[child.name,:]
    
    
    def refresh_mutations(self): 
        for node in self.nodes: 
            node.clear_mutations()
        best_nodes = np.argmax(self.LLR, axis = 0)
        for i in range(self.M): 
            self.nodes[best_nodes[i]].add_mutation(i)
            
    
    def read_mutation_tree(self, mutation_tree): 
        pass
    
    
    def node_info(self): 
        for node in self.nodes: 
            print(node)
    
    
    def structure_info(self): 
        for node in self.nodes: 
            print(node.name, '-->', [child.name for child in node.get_children()])




class MutationTree:     
    def __init__(self, likelihoods1, likelihoods2, cell_tree = None): 
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
            self.likelihoods[:,node.name] = self.likelihoods[:,node.get_parent().name] + np.sum(sm_LLR[:,node.mutations], axis = 1)
        
    
    def attach_cells(self): 
        self.attachments = np.argmax(self.likelihoods, axis = 1)
        # TBC: use other data structures, e.g. a list of lists, or a boolean matrix
    
    
    def node_info(self): 
        for node in self.nodes: 
            print(node)
    
    
    def attachment_info(self): 
        print(self.attachments)
    
    
    
    




        
        
        
        
        
        

if __name__ == '__main__': 
    #import pandas as pd
    
    #ref = pd.read_csv('./Data/glioblastoma_BT_S2/ref.csv', index_col = 0).to_numpy(dtype = float)
    #alt = pd.read_csv('./Data/glioblastoma_BT_S2/alt.csv', index_col = 0).to_numpy(dtype = float)
    
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
    mt = MutationTree(likelihoods1, likelihoods2, ct)
    mt.node_info()
    mt.attachment_info()
    print()
    