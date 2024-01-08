import numpy as np
import graphviz
import warnings

from .forest import Forest
from .utilities import randint_with_exclude




class CellTree(Forest):
    def __init__(self, n_cells=3, n_mut=0):
        if n_cells < 3:
            warnings.warn('Cell tree too small, nothing to explore', RuntimeWarning)

        super().__init__(2*n_cells - 1)
        self.n_cells = n_cells
        self.n_mut = n_mut
        
        # keep track of main tree during pruning & inserting
        self.main_root = None

        # initialize with completely random structure
        self.rand_structure()


    @property
    def n_mut(self):
        return self.mut_loc.size
    

    @n_mut.setter
    def n_mut(self, n_mut):
        self.mut_loc = np.ones(n_mut, dtype=int) * -1
        self.directions = np.ones(n_mut, dtype=int) * -1
    

    @property
    def anchors(self):
        return [rt for rt in self.roots if rt != self.main_root]
    

    @property
    def joint(self):
        ''' joint log-likelihood of the cell tree, with attached mutations '''
        result = sum([self.llr[self.mut_loc[j], j] for j in range(self.n_mut)]) + sum(self.loc_joint_1)
        # Need to round because numpy.sum can give slightly different results when summing a matrix along different axis
        # See the "Notes" section in this page: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # If not rounded, the joint likelihood might increase when converting between the two tree spaces
        # Sometimes this traps the optimization in an infinite loop
        return round(result, self.n_decimals)


    def rand_structure(self, subroot=None):
        '''
        Randomize a given subtree. If no subtree is provided, randomize the entire tree.
        '''
        # determine the leaf and internal vertices
        if subroot is None:
            anchor = -1
            leaves = [i for i in range(self.n_cells)]
            internals = [i for i in range(self.n_cells, self.n_vtx)]
            self.main_root = internals[-1]
        else:
            anchor = self.parent(subroot)
            leaves = []
            internals = []
            for vtx in self.dfs(subroot):
                # caveat: what if vtx is both leaf and root?
                if self.isleaf(vtx):
                    leaves.append(vtx)
                else:
                    internals.append(vtx)

        # repeatedly assign two children to an internal vertex
        for parent in internals:
            children = np.random.choice(leaves, size = 2, replace = False) # choose two random children
            for child in children:
                self.assign_parent(child, parent)
                leaves.remove(child) # the children cannot be assigned to other parents
            leaves.append(parent) # the parent can now serve as a child
        
        self.assign_parent(leaves[-1], anchor)
    

    def rand_mut_loc(self):
        self.mut_loc = np.random.randint(self.n_vtx, size=self.n_mut)


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
    

    def fit_llh(self, llh_1, llh_2, reversible=True, sig_digits=10):
        '''
        Gets ready to run optimization using provided log-likelihoods.
        The two genotypes involved in the mutation, gt1 and gt2, are not required for optimization.

        [Arguments]
            llh1: 2D array in which entry [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh2: 2D array in which entry [i,j] is the log-likelihood of cell i having gt2 at locus j
            reversible: either an array containing mutations whose directions are reversible (i.e. not known a priori),
                or a boolean value that applies to all loci.
                If a mutation is not reversible, the direction is assumed to be from gt1 to gt2.
        '''
        assert(llh_1.shape == llh_2.shape)

        # adjust tree size if needed
        if llh_1.shape[0] != self.n_cells:
            self.__init__(*llh_1.shape)
            warnings.warn('Reinitialized cell tree since the number of cells does not match the row count of the llh matrix.')
        if llh_1.shape[1] != self.n_mut:
            self.n_mut = llh_1.shape[1]

        # data to be used directly in optimization
        self.llr = np.empty((self.n_vtx, self.n_mut))
        self.llr[:self.n_cells,:] = llh_2 - llh_1
        self.loc_joint_1 = np.sum(llh_1, axis=0)
        self.loc_joint_2 = np.sum(llh_2, axis=0)

        # prepare to deal with reversibility
        self.flipped = np.zeros(self.n_mut, dtype=bool)

        if reversible == False:
            self.reversible = []
        elif reversible == True:
            self.reversible = [j for j in range(self.n_mut)]
        else:
            self.reversible = reversible

        # determine a rounding precision
        self.n_decimals = int(sig_digits - np.log10(np.abs(np.sum(llh_1)))) # round tree likelihoods to this precision

        # assign mutations to optimal locations
        self.update_all()
    

    def flip_direction(self, mut):
        self.flipped[mut] = not self.flipped[mut]
        self.llr[:,mut] = - self.llr[:,mut]
        self.loc_joint_1[mut], self.loc_joint_2[mut] = self.loc_joint_2[mut], self.loc_joint_1[mut]
    

    def update_llr(self):
        for rt in self.roots:
            for vtx in self.rdfs(rt):
                if self.isleaf(vtx): # nothing to be done for leaves
                    continue
                # LLR at internal vertex is the sum of LLR of both children
                self.llr[vtx,:] = np.sum(self.llr[self.children(vtx),:], axis=0)


    def update_mut_loc(self):
        self.mut_loc = np.argmax(self.llr, axis=0)
        
        # for reversible mutations, check if the inverse mutation has better log-likelihood
        for j in self.reversible:
            alt_loc = np.argmin(self.llr[:,j]) # when flipped
            if self.loc_joint_1[j] + self.llr[self.mut_loc[j], j] < self.loc_joint_2[j] - self.llr[alt_loc, j]:
                self.flip_direction(j)
                self.mut_loc[j] = alt_loc
        
        # if best attachment still worse than L1, we want the mutation to attach to nowhere
        #for j in range(self.n_mut):
        #    if self.ct_LLR[self.ct.attachments[j], j] < 0:
        #        # any negative integer represents "outside of the tree", here we use -1
        #        self.ct.attachments[j] = -1
    

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
        if self.isroot(subroot):
            raise ValueError('Cannot prune a root and still maintain a binary structure.')
        
        anchor = self.parent(subroot)

        for sib in self.siblings(subroot):
            self.assign_parent(sib, self.parent(anchor))
            if self.isroot(anchor):
                self.main_root = sib
        
        self.prune(anchor)
    

    def binary_insert(self, anchor, target):
        '''
        Insert a pruned subtree while keeping the main tree strictly binary
        '''
        if not self.isroot(anchor) or anchor == self.main_root or anchor == -1:
            raise ValueError(f'{anchor} is not anchor of a pruned subtree.')
        
        if self.isroot(target):
            self.main_root = anchor
        self.assign_parent(anchor, self.parent(target))
        self.assign_parent(target, anchor)
    

    def random_prune(self, n_prunes):
        if len(self.roots) != 1:
            raise RuntimeError('Random pruning works only when there is a single tree.')

        for i in range(n_prunes):
            exclude = self.roots + [self.children(anchor)[0] for anchor in self.anchors]
            new_sr = randint_with_exclude(self.n_vtx, exclude)
            self.binary_prune(new_sr)

        self.update_all()
    

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
                self.llr[target,:] += self.llr[subroot,:]
                # recursively search all descendants
                for child in self.children(target):
                    child_best_target, child_best_joint = search_insertion_loc(child)
                    if child_best_joint > best_joint:
                        best_target = child_best_target
                        best_joint = child_best_joint
                # restore the original LLR at target after searching all descendants
                self.llr[target,:] -= self.llr[subroot,:]
            
            return best_target, best_joint

        for anchor in self.anchors:
            subroot = self.children(anchor)[0]
            best_target, best_joint = search_insertion_loc(self.main_root)
            self.binary_insert(anchor, best_target)
            self.update_llr()
        
        self.update_mut_loc()
        assert(best_joint == self.joint)


    def random_optimize(self, n_steps=20):
        for i in range(n_steps):
            old_joint = self.joint
            old_parent_vec = self.parent_vec
            old_main_root = self.main_root
            self.random_prune(2)
            self.greedy_insert()
            if self.joint < old_joint:
                self.parent_vec = old_parent_vec
                self.main_root = old_main_root
                self.update_all()
    

    def exhaustive_optimize(self, max_loops=None, leaf_only=False):
        self.update_all()
        current_joint = self.joint

        sr_candidates = range(self.n_cells) if leaf_only else range(self.n_vtx)

        if max_loops is None:
            max_loops = self.n_vtx
        loop_count = 0
        while loop_count < max_loops:
            for sr in sr_candidates:
                if sr == self.main_root:
                    continue
                self.binary_prune(sr)
                self.update_llr()
                self.greedy_insert()
            new_joint = self.joint
            if new_joint == current_joint:
                break
            elif new_joint < current_joint:
                warnings.warn(
                    f'Likelihood decreased from {current_joint} to {new_joint}, ' + 
                    'something might have gone wrong when searching for the best insertion location.'
                )
            else:
                current_joint = new_joint
                loop_count += 1
        
        print(f'Completed {loop_count} search loops.')
