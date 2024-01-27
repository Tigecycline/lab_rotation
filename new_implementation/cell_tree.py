import numpy as np
import graphviz
import warnings

from .prunetree_base import PruneTree




class CellTree(PruneTree):
    def __init__(self, n_cells=3, n_mut=0, reversible_mut=True):
        '''
        [Arguments]
            reversible_mut: a boolean value indicating whether the opposite direction should be considered.
                If the mutations are not reversible, the direction is assumed to be from gt1 to gt2.
        '''
        if n_cells < 3:
            warnings.warn('Cell tree too small, nothing to explore', RuntimeWarning)

        super().__init__(2*n_cells - 1)
        self.n_cells = n_cells
        self.n_mut = n_mut
        self.reversible = reversible_mut

        # initialize with completely random structure
        self.rand_subtree()


    @property
    def n_mut(self):
        return self.mut_loc.size
    

    @n_mut.setter
    def n_mut(self, n_mut):
        self.mut_loc = np.ones(n_mut, dtype=int) * -1
        self.flipped = np.zeros(n_mut, dtype=bool)


    def rand_subtree(self, leaves=None, internals=None):
        '''
        Construct a random subtree. If no subtree is specified, randomize the entire tree.
        '''
        # determine the leaf and internal vertices
        if leaves is None or internals is None:
            leaves = [i for i in range(self.n_cells)]
            internals = [i for i in range(self.n_cells, self.n_vtx)]
        else:
            if len(leaves) != len(internals) + 1:
                raise ValueError('There must be exactly one more leaf than internals.')
        
        # repeatedly assign two children to an internal vertex
        for parent in internals:
            children = np.random.choice(leaves, size=2, replace=False) # choose two random children
            for child in children:
                self.assign_parent(child, parent)
                leaves.remove(child) # the children cannot be assigned to other parents
            leaves.append(parent) # the parent can now serve as a child
        
        self.reroot(internals[-1])
    
    
    def rand_mut_loc(self):
        self.mut_loc = np.random.randint(self.n_vtx, size=self.n_mut)
    

    def fit_llh(self, llh_1, llh_2, sig_digits=10):
        '''
        Gets ready to run optimization using provided log-likelihoods.
        The two genotypes involved in the mutation, gt1 and gt2, are not required for optimization.

        [Arguments]
            llh1: 2D array in which entry [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh2: 2D array in which entry [i,j] is the log-likelihood of cell i having gt2 at locus j
        '''
        assert(llh_1.shape == llh_2.shape)

        # adjust tree size if needed
        if llh_1.shape[0] != self.n_cells:
            self.__init__(*llh_1.shape, self.reversible)
            warnings.warn('Reinitialized cell tree since the number of cells does not match the row count of the llh matrix.')
        elif llh_1.shape[1] != self.n_mut:
            self.n_mut = llh_1.shape[1]

        # data to be used directly in optimization
        self.llr = np.empty((self.n_vtx, self.n_mut))
        self.llr[:self.n_cells,:] = llh_2 - llh_1

        # joint likelihood of each locus when all cells have genotype 1 or 2
        self.loc_joint_1 = llh_1.sum(axis=0)
        self.loc_joint_2 = llh_2.sum(axis=0)

        # determine a rounding precision for joint likelihood calculation
        #mean_abs = np.sum(np.abs(llh_1 + llh_2)) / 2 # mean abs value when attaching mutations randomly
        #self.n_decimals = int(sig_digits - np.log10(mean_abs))
        # Need to round because numpy.sum can give slightly different results when summing a matrix along different axis
        # See the "Notes" section in this page: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # If not rounded, the joint likelihood might increase when converting between the two tree spaces
        # Sometimes this traps the optimization in an infinite loop

        # assign mutations to optimal locations
        self.update_all()
    

    def fit_mutation_tree(self, mt):
        assert(self.n_cells == mt.n_cells)

        # most recent common ancestor of cells below a mutation node
        mrca = np.ones(mt.n_vtx, dtype=int) * -1 # -1 means no cell has been found below this mutation
        
        next_internal = self.n_cells
        for mvtx in mt.rdfs(mt.main_root): # mvtx for "mutation vertex"
            leaves = [mrca[child] for child in mt.children(mvtx) if mrca[child] != -1]
            leaves += np.where(mt.cell_loc == mvtx)[0].tolist()
            if len(leaves) == 0: # no cell below, nothing to do
                continue
            elif len(leaves) == 1: # one cell below, no internal node added
                mrca[mvtx] = leaves[0]
            elif len(leaves) > 1: # more than one cell below, add new internal node(s)
                internals = [i for i in range(next_internal, next_internal + len(leaves) - 1)]
                self.rand_subtree(leaves, internals)
                mrca[mvtx] = internals[-1]
                next_internal += len(internals)


    def update_llr(self):
        for rt in self.roots:
            for vtx in self.rdfs(rt):
                if self.isleaf(vtx): # nothing to be done for leaves
                    continue
                # LLR at internal vertex is the sum of LLR of both children
                self.llr[vtx,:] = self.llr[self.children(vtx),:].sum(axis=0)


    def update_mut_loc(self):
        # if mutation directions are unknown, test both directions
        loc_joint = np.empty(self.n_mut)
        if self.reversible:
            for j in range(self.n_mut):
                loc_neg, loc_pos = self.llr[:,j].argmin(), self.llr[:,j].argmax() # best locations
                llh_pos = self.loc_joint_1[j] + self.llr[loc_pos, j]
                llh_neg = self.loc_joint_2[j] - self.llr[loc_neg, j]
                if llh_pos < llh_neg:
                    self.mut_loc[j] = loc_neg
                    self.flipped[j] = True
                    loc_joint[j] = llh_neg
                else:
                    self.mut_loc[j] = loc_pos
                    self.flipped[j] = False
                    loc_joint[j] = llh_pos
        else:
            self.mut_loc = self.llr.argmax(axis=0)
            loc_joint = np.array([self.llr[self.mut_loc[j], j] for j in range(self.n_mut)]) + self.loc_joint_1

        self.joint = loc_joint.sum()


    def update_all(self):
        self.update_llr()
        self.update_mut_loc()
    

    def binary_prune(self, subroot):
        '''
        Prune a subtree while keeping the main tree strictly binary
        This is achieved by removing the subtree together with its "anchor"
            which is the direct parent of the pruned subtree
        Meanwhile, the original sibling of the pruned subtree is re-assigned to its grandparent
        As a result, the pruned subtree has the "anchor" as its root and is not binary
        '''
        # for purpose of efficiency, this check can be commented out after thorough testsing
        self.splice(next(self.siblings(subroot)))
    

    def greedy_insert(self):
        def search_insertion_loc(target):
            # calculate LLR at anchor when anchor is inserted above target
            self.llr[anchor,:] = self.llr[subroot,:] + self.llr[target,:]
            # highest achievable joint log-likelihood with this insertion
            self.update_mut_loc()

            best_target = target
            best_joint = self.joint

            if not self.isleaf(target):
                # for any descendant of target, the LLR at target is the original one plus that of subroot
                original = self.llr[target,:].copy()
                self.llr[target,:] += self.llr[subroot,:]
                # recursively search all descendants
                for child in self.children(target):
                    child_best_target, child_best_joint = search_insertion_loc(child)
                    if child_best_joint > best_joint:
                        best_target = child_best_target
                        best_joint = child_best_joint
                # restore the original LLR at target after searching all descendants
                self.llr[target,:] = original
            
            return best_target, best_joint

        for anchor in self.pruned_roots():
            subroot = self.children(anchor)[0]
            best_target, best_joint = search_insertion_loc(self.main_root)
            self.insert(anchor, best_target)
            self.update_all()
        
        #if best_joint != self.joint:
        #    warnings.warn(f'Predicted joint {best_joint} is not consistent with calcualted joint {self.joint}.')
    
                                                                                                                           
    def exhaustive_optimize(self, leaf_only=False):
        self.update_all()

        sr_candidates = range(self.n_cells) if leaf_only else range(self.n_vtx)

        for sr in sr_candidates:
            if sr == self.main_root:
                continue
            self.binary_prune(sr)
            self.update_llr()
            self.greedy_insert()


    def to_graphviz(self, filename=None, engine='dot', leaf_shape='circle', internal_shape='circle'):
        ''' Returns a graphviz Digraph object corresponding to the tree '''
        dgraph = graphviz.Digraph(filename=filename, engine=engine)

        mutations = [[] for i in range(self.n_vtx)]
        for mut, loc in enumerate(self.mut_loc):
            mutations[loc].append(mut)
        
        for vtx in range(self.n_vtx):
            # node label is the integer that represents it
            node_label = str(vtx)
            # find mutations that should be placed above current node
            edge_label = ','.join([str(j) for j in mutations[vtx]])

            # treat leaf (observed) and internal nodes differently
            if self.isleaf(vtx):
                dgraph.node(node_label, shape=leaf_shape)
            else:
                dgraph.node(node_label, shape=internal_shape, style='filled', color='gray')
            
            # create edge to parent node, or to void node in case of root
            if self.isroot(vtx):
                # create a void node with an edge to the root
                dgraph.node(f'void_{vtx}', label='', shape='point')
                dgraph.edge(f'void_{vtx}', node_label, label=edge_label)
            else:
                dgraph.edge(str(self.parent(vtx)), node_label, label=edge_label)
        
        return dgraph