import numpy as np
import graphviz
import copy
from random import shuffle

from .tree_node import TreeNode
from .utilities import randint_with_exclude




class CellTree: 
    def __init__(self, n_cells): 
        self.n_cells = n_cells 
        self.nodes = [TreeNode(i) for i in range(2 * self.n_cells - 1)]
        self.root = None
        
        self.attachments = [] # locations of the mutations
        # If mutation j is attached to some edge, then attachments[j] is the ID of the child node of that edge. 
        # Otherwise, attachments[j] is -1. 
        
        self.last_move = None # 0 = prune & insert, 1 = subtree swap
        self.undo_args = None # information of the move required to undo it
        
    
    @property
    def n_nodes(self): 
        ''' number of nodes '''
        return len(self.nodes)
    

    @property
    def n_mut(self):
        ''' number of attached mutations '''
        return len(self.attachments)

    
    @property
    def mutations(self): 
        result = [[] for i in range(self.n_nodes)]
        if self.attachments is not None: 
            for j in range(self.n_mut): 
                if self.attachments[j] >= 0: # negative values mean "outside of the tree"
                    result[self.attachments[j]].append(j)
        return result
    

    @property
    def parent_vec(self):
        '''
        An array in which the i-th entry is the ID of the parent of node i
        If node i has no parent (i.e. is root), the entry is -1
        '''
        result = np.empty(self.n_nodes, dtype=int)
        for node in self.nodes:
            result[node.ID] = -1 if node.isroot else node.parent.ID
        return result
    
    
    @parent_vec.setter
    def parent_vec(self, parent_vec):
        '''
        Rearranges the tree according to provided parent vector
        Root is indicated by -1
        '''
        assert(len(parent_vec) == self.n_nodes)
        for node in self.nodes:
            parent_id = parent_vec[node.ID]
            if self.n_cells <= parent_id < self.n_nodes:
                node.assign_parent(self.nodes[parent_id])
            else:
                node.assign_parent(None)
                self.root = node
    

    @property
    def linkage_matrix(self):
        '''
        Returns a matrix that can be used to draw dendrogram with other packages
        It is in the same form as the one given by scipy.cluster.hierarchy.linkage
        Except for the leaves, cluster index is generally not the same as the ID of the node that represents the cluster
        '''
        result = np.empty((self.n_cells - 1, 4))
        heights = np.zeros(self.n_nodes) # height of the cluster node, equal to distance between the two children clusters
        n_leaves = np.ones(self.n_nodes) # number of cells in a cluster
        clusters = np.empty(self.n_nodes, dtype=int) # maps node ID to cluster
        clusters[:self.n_cells] = np.arange(self.n_cells, dtype=int) # the first N clusters are single cells

        i = 0 # counter for the merge operation
        for node in self.root.reverse_DFS:
            if not node.isleaf:
                clusters[node.ID] = self.n_cells + i
                cluster1, cluster2 = clusters[node.children[0].ID], clusters[node.children[1].ID] # the two clusters to be merged
                result[i,:2] = cluster1, cluster2
                heights[self.n_cells + i] = max(heights[cluster1], heights[cluster2]) + 1
                n_leaves[self.n_cells + i] = n_leaves[cluster1] + n_leaves[cluster2]
                i += 1
        result[:,2] = heights[self.n_cells:]
        result[:,3] = n_leaves[self.n_cells:]

        return result

    
    @property 
    def dist_matrix(self): 
        ''' Distance matrix of all leaves '''
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
        Creates a random binary tree using selected leaves and internal nodes
        [Arguments]
            selected: indices of the selected nodes ("leaves" of the random subtree)
            internal_idx: nodes from this index are used as internal nodes of the subtree
        '''
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
        Shuffles the entire tree
        '''
        self.random_subtree([i for i in range(self.n_cells)], self.n_cells)
        self.root = self.nodes[-1]
    
    
    def fit_structure(self, mutation_tree):
        '''
        Rearranges the tree according to provided mutation tree
        NB The cell lineage tree corresponding to a mutation tree is usually NOT unique
        If multiple structures are available, a random one is picked
        '''
        assert(self.n_cells == mutation_tree.n_cells)

        mrca = np.ones(mutation_tree.n_nodes, dtype = int) * -1 # most recent common ancestor of cells below a mutation node; -1 means no cell has been found below this mutation
        
        current_idx = self.n_cells
        for mnode in mutation_tree.root.reverse_DFS: 
            need_adoption = [mrca[child.ID] for child in mnode.children if mrca[child.ID] >= 0]
            need_adoption += np.where(mutation_tree.attachments == mnode.ID)[0].tolist()
            # if no cell found below mnode, nothing to do
            if len(need_adoption) == 1: # one cell below, no internal node added
                mrca[mnode.ID] = need_adoption[0] 
            elif len(need_adoption) > 1: # more than one cell below, add new internal node(s)
                current_idx = self.random_subtree(need_adoption, current_idx)
                mrca[mnode.ID] = current_idx - 1
        
        self.root = self.nodes[-1]
        # N.B. For most of the cell nodes, their old parent (before fitting to a mutation tree)
        # is overwritten by the assign_parent function, but the new root is not. 
        # This is not a problem when new root is the same as the original root, but 
        # otherwise, the parent of the new root is not overwritten and needs to be set to None
        self.root.assign_parent(None)
    
    
    def propose_prune_insert(self):
        ''' Returns random valid arguments for the prune_insert method '''
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
        ''' Returns a graphviz Digraph object corresponding to the tree '''
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
    def __init__(self, n_mut): 
        self.n_mut = n_mut
        
        self.nodes = [TreeNode(i) for i in range(self.n_mut + 1)]
        
        self.attachments = None
        
        self.last_move = None # 0 = prune & attach, 1 = node swap
        self.undo_args = None
    
    
    @property
    def n_nodes(self): 
        return len(self.nodes)
    

    @property
    def n_cells(self):
        if self.attachments is None:
            return 0
        else:
            return len(self.attachments)
    
    
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
    
    
    @property
    def dist_matrix(self): 
        pass # to be implemented


    def copy(self):
        return copy.deepcopy(self)
    
    
    def random_structure(self): 
        pass # to be implemented
    
    
    def fit_structure(self, cell_tree, random_order = False): 
        '''
        random_order: whether the order of mutations on the same edge is randomized
        '''
        assert(self.n_mut == cell_tree.n_mut)

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
                if random_order: 
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
        # prune the subtree with root subrootID, attach it to the edge targetID
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
    
    
    
