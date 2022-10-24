import numpy as np
import graphviz
import copy
from random import shuffle

from tree_node import TreeNode
from utilities import randint_with_exclude




class CellTree: 
    def __init__(self, n_cells = 1, n_mut = None): 
        self.n_cells = n_cells 
        self.n_mut = n_mut
        self.mutation_tree = None
        self.nodes = [TreeNode(i) for i in range(2 * self.n_cells - 1)]
        self.root = None
        
        self.attachments = None
        
        self.last_move = None # 0 = prune & insert, 1 = subtree swap
        self.undo_args = None # information of the move required to undo it
        
    
    @property
    def n_nodes(self): 
        ''' number of nodes '''
        return len(self.nodes)
    
    
    @property
    def mutations(self): 
        result = [[] for i in range(self.n_nodes)]
        if self.attachments is not None: 
            for j in range(self.n_mut): 
                if self.attachments[j] >= 0: # negative values mean "outside of the tree"
                    result[self.attachments[j]].append(j)
        return result
    
    
    @property 
    def dist_matrix(self): 
        # TBC: find more efficient algorithm 
        result = - np.ones((self.n_cells, self.n_nodes), dtype = int)
        np.fill_diagonal(result, 0)
        for node in self.root.reverse_DFS: 
            if node.isleaf: 
                continue
            child1, child2 = node.children
            for leaf1 in child1.leaves: 
                for leaf2 in child2.leaves: 
                    result[leaf1.ID, node.ID] = result[leaf1.ID, child1.ID] + 1
                    result[leaf2.ID, node.ID] = result[leaf2.ID, child2.ID] + 1
                    dist = result[leaf1.ID, child1.ID] + result[leaf2.ID, child2.ID] + 2
                    if leaf1.ID < leaf2.ID: 
                        result[leaf1.ID, leaf2.ID] = dist
                    else: 
                        result[leaf2.ID, leaf1.ID] = dist
        return result[:,:self.n_cells]
    
    
    def copy(self): 
        return copy.deepcopy(self)
        
    
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
    
    
    def fit_structure(self, mutation_tree): 
        mrca = np.ones(mutation_tree.n_nodes, dtype = int) * -1 # most recent common ancestor of cells below a mutation node; -1 means no cell has been found below this mutation
        
        current_idx = self.n_cells
        for mnode in mutation_tree.root.reverse_DFS: 
            need_adoption = [mrca[child.ID] for child in mnode.children if mrca[child.ID] >= 0]
            need_adoption += np.where(mutation_tree.attachments == mnode.ID)[0].tolist()
            # if no cell found below mnode, nothing to do
            if len(need_adoption) == 1: # 1 cell found, no internal node added
                mrca[mnode.ID] = need_adoption[0] 
            elif len(need_adoption) > 1: # more than one cell found, add new internal node(s)
                current_idx = self.random_subtree(need_adoption, current_idx)
                mrca[mnode.ID] = current_idx - 1
        
        self.root = self.nodes[-1]
        # N.B. For most of the cell nodes, their old parent (before fitting to a mutation tree)
        # is overwritten by the assign_parent function, but the new root is not. 
        # This is not a problem when new root is the same as the original root, but 
        # otherwise, the parent of the new root is not overwritten and needs to be set to None
        self.root.assign_parent(None)
    
    
    def propose_prune_insert(self): 
        subrootID = randint_with_exclude(self.n_nodes, [self.root.ID])
        # subroot: root of the pruned subtree
        subroot = self.nodes[subrootID]

        # insertion to a descendant destroys tree structure
        exclude = [node.ID for node in subroot.DFS]
        # insertion to a sibling doesn't change the tree
        #for sibling in subroot.siblings: 
        #    exlucde.append(sibling.ID)
        # parent of the subtree needs to be pruned as well, since tree is strictly binary
        exclude.append(subroot.parent.ID)

        #suitable_idx = np.delete(np.arange(self.n_nodes), exclude)
        targetID = randint_with_exclude(self.n_nodes, exclude)

        return subrootID, targetID
    
    
    def prune_insert(self, subrootID, targetID): 
        '''
        Prune a subtree and insert it somewhere else
        subroot: root of the subtree that gets pruned
        target: self.nodes[target_idx] # subtree is inserted to the edge above target
        '''
        self.last_move = 0
        
        # the parent of subroot is reused as the new node created when inserting to an edge
        subroot, target = self.nodes[subrootID], self.nodes[targetID]
        anchor = subroot.parent
        
        for sibling in subroot.siblings: 
            self.undo_args = [subrootID, sibling.ID] # insert to ex-sibling to undo the move
            sibling.assign_parent(anchor.parent) # sibling becomes child of grandparent
            if sibling.isroot: # if pruned at root, sibling becomes new root
                self.root = sibling 
        
        anchor.assign_parent(target.parent) 
        if anchor.isroot: # if inserted to root, anchor becomes new root
            self.root = anchor
            
        target.assign_parent(anchor)
    
    
    def random_prune_insert(self): 
        subrootID, targetID = self.propose_prune_insert()
        self.prune_insert(subrootID, targetID)
    
    
    def propose_swap(self): 
        subroot1ID = randint_with_exclude(self.n_nodes, [self.root.ID])
        subroot1 = self.nodes[subroot1ID]
        
        exclude = [node.ID for node in subroot1.DFS]
        exclude += [node.ID for node in subroot1.ancestors]
        #print(exclude)
        #exclude += [node.ID for node in subroot1.siblings]

        subroot2ID = randint_with_exclude(self.n_nodes, exclude)

        return subroot1ID, subroot2ID
    
    
    def subtree_swap(self, subroot1ID, subroot2ID): 
        self.last_move = 1
        self.undo_args = [subroot1ID, subroot2ID]
        subroot1, subroot2 = self.nodes[subroot1ID], self.nodes[subroot2ID]
        old_parent = subroot1.parent
        subroot1.assign_parent(subroot2.parent)
        subroot2.assign_parent(old_parent)
    
    
    def random_subtree_swap(self): 
        subroot1ID, subroot2ID = self.propose_swap()
        self.subtree_swap(subroot1ID, subroot2ID)
    
    
    def undo_move(self): 
        if self.last_move == 0: 
            self.prune_insert(self.undo_args[0], self.undo_args[1])
        elif self.last_move == 1: 
            self.subtree_swap(self.undo_args[0], self.undo_args[1])
        else: 
            print('[CellTree.undo_move] ERROR: last_move not found.')
        self.last_move = None
    
    
    def to_graphviz(self, filename = None): 
        dg = graphviz.Digraph(filename = filename)
        
        mutations = self.mutations
        for node in self.nodes: 
            if node.isleaf: 
                dg.node(str(node.ID), shape = 'circle')
            else: 
                dg.node(str(node.ID), shape = 'circle', style = 'filled', color = 'gray')
            
            if mutations[node.ID]: 
                edge_label = 'm' + ','.join([str(j) for j in mutations[node.ID]])
            else: 
                edge_label = ''
            
            if node.isroot: 
                dg.node('dummy', label = '', shape = 'point')
                dg.edge('dummy', str(node.ID), label = edge_label)
            else: 
                dg.edge(str(node.parent.ID), str(node.ID), label = edge_label)
        
        return dg




class MutationTree: 
    def __init__(self, n_cells = 0, n_mut = 0, cell_tree = None): 
        if cell_tree is None: 
            self.n_cells = n_cells
            self.n_mut = n_mut
        else: 
            self.n_cells = cell_tree.n_cells
            self.n_mut = cell_tree.n_mut
        
        self.nodes = [TreeNode(i) for i in range(self.n_mut + 1)]
        if cell_tree is not None: 
            self.fit_structure(cell_tree)
        
        self.node_joints = np.empty((self.n_cells, self.n_nodes)) # joint likelihood of cell i when attached to node j
        self.attachments = None
        
        self.last_move = None # 0 = prune & attach, 1 = node swap
        self.undo_args = None
    
    
    @property
    def n_nodes(self): 
        return len(self.nodes)
    
    
    @property
    def root(self): 
        return self.nodes[-1]
    
    
    @property
    def cells(self): 
        result = [[] for i in range(self.n_nodes)]
        if self.attachments is not None: 
            for i in range(self.n_cells): 
                result[self.attachments[i]].append(i)
        return result
    
    
    def random_structure(self): 
        pass # to be implemented
    
    
    def fit_structure(self, cell_tree): 
        for node in self.nodes: 
            node.assign_parent(None) # clear the current structure
        mrm = np.empty(cell_tree.n_nodes, dtype = int) # mrm for "most recent mutation"
        mrm[cell_tree.root.ID] = self.root.ID # start with "root mutation", which represents wildtype
        mutations = cell_tree.mutations
        cell_tree.root.parent = cell_tree.root # temporary change so that no need to constantly check cnode.isroot
        
        for cnode in cell_tree.root.DFS:  # cnode for "cell node"
            mut_idx = mutations[cnode.ID] # mutations attached to the edge above cnode
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
        
        for node in self.nodes[:-1]:
            # mutations that are not assigned in the cell tree (i.e. CellTree.attachment[j] out of range)
            if node.parent is None: 
                node.assign_parent(self.root) 
    
    
    def fit(self, likelihoods1 = None, likelihoods2 = None): 
        self.refresh_node_joints()
        self.attach_cells()
    
    
    def propose_prune_attach(self): 
        # If root has one child and it is pruned, there will be no suitable node to attach
        if len(self.root.children) == 1: 
            exclude = [self.root.children[0].ID]
            subrootID = randint_with_exclude(self.n_mut, exclude)
        else: 
            subrootID = np.random.randint(self.n_mut)
        subroot = self.nodes[subrootID]
        
        # attaching to a descendant violates tree structure
        exclude = [node.ID for node in subroot.DFS]
        # attaching to original parent does not change the tree
        exclude.append(subroot.parent.ID)
        
        targetID = randint_with_exclude(self.n_nodes, exclude)
        
        return subrootID, targetID
    
    
    def prune_attach(self, subrootID, targetID):
        self.last_move = 0
        subroot, target = self.nodes[subrootID], self.nodes[targetID]
        self.undo_args = [subrootID, subroot.parent.ID]
        subroot.assign_parent(target)
    
    
    def random_prune_attach(self): 
        subrootID, targetID = self.propose_prune_attach()
        self.prune_attach(subrootID, targetID)
    
    
    def propose_node_swap(self): 
        node1ID = np.random.randint(self.n_mut)
        node1 = self.nodes[node1ID]
        
        # the two nodes must be different
        exclude = [node1ID]
        # swapping with a sibling does not change the tree
        #exclude += [sibling.ID for sibling in node1.siblings]
        
        node2ID = randint_with_exclude(self.n_mut, exclude)
        
        return node1ID, node2ID
    
    
    def node_swap(self, node1ID, node2ID): 
        self.last_move = 1
        self.undo_args = [node1ID, node2ID]
        node1, node2 = self.nodes[node1ID], self.nodes[node2ID]
        
        self.nodes[node1ID], self.nodes[node2ID] = node2, node1
        node1.ID, node2.ID = node2.ID, node1.ID
    
    
    def random_node_swap(self): 
        node1ID, node2ID = self.propose_node_swap()
        self.node_swap(node1ID, node2ID)
    
    
    def undo_move(self): 
        if self.last_move == 0: 
            self.prune_attach(self.undo_args[0], self.undo_args[1])
        elif self.last_move == 1:
            self.node_swap(self.undo_args[0], self.undo_args[1])
        else: 
            print('[MutationTree.undo_move] ERROR: last_move not found.')
        self.last_move = None
        
        
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
    
    
    def to_graphviz(self, filename = None): 
        dg = graphviz.Digraph(filename = filename)
        
        dg.node(str(self.root.ID), label = 'root', shape = 'rectangle')
        for node in self.nodes[:-1]: 
            dg.node(str(node.ID), shape = 'rectangle', style = 'filled', color = 'gray')
            dg.edge(str(node.parent.ID), str(node.ID))
        
        if self.attachments is not None: 
            for i in range(self.n_cells): 
                name = 'c' + str(i)
                dg.node(name, shape = 'circle')
                # TBC: use undirected edge for cell attachment
                #dg.edge(str(self.attachments[i]), name, dir = 'none')
                dg.edge(str(self.attachments[i]), name)
        
        return dg
    
    
    
