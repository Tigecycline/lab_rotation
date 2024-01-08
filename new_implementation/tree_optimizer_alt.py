import warnings
from itertools import permutations
import numpy as np

from .cell_tree import CellTree
#from .mutation_filter import *
from .utilities import randint_with_exclude




class TreeOptimizer2:
    def __init__(self, convergence_factor=5, timeout_factor=20, sig_digits=10, strategy='hill climb', spaces=None):
        '''
        [Arguments]
            spaces: spaces that will be searched and the order of the search
                    'c' = cell tree space, 'm' = mutation tree space
                    default is ['c','m'], i.e. start with cell tree and search both spaces
            sig_dig: number of significant digits to use when calculating joint probability
        '''
        self.convergence_factor = convergence_factor
        self.timeout_factor = timeout_factor
        self.sig_digits = sig_digits
        self.strategy = strategy
        self.spaces = ['c', 'm'] if spaces is None else spaces
    

    @property
    def ct_joint(self):
        ''' joint likelihood of the cell tree, with attached mutations '''
        result = sum([self.ct_llr[self.ct.mut_loc[j], j] for j in range(self.n_mut)]) + sum(self.locus_joint_1)
        return round(result, self.n_decimals)
    

    def flip_direction(self, j):
        ''' Reverses the direction of a mutation j by exchanging corresponding columns in likelihoods1 and likelihoods2 '''
        self.direction[j] = - self.direction[j]
        #temp = self.llh1[:,j].copy()
        #self.llh1[:,j] = self.llh2[:,j]
        #self.llh2[:,j] = temp
        self.ct_llr[:,j] = - self.ct_llr[:,j]
        self.locus_joint_1[j], self.locus_joint_2[j] = self.locus_joint_2[j], self.locus_joint_1[j]
    

    def update_ct_llr(self):
        for root in self.ct.roots:
            for vtx in self.ct.rdfs(root):
                if self.ct.isleaf(vtx): # nothing to be done for leaves
                    continue
                # LLR at internal vertex is the sum of LLR of both children
                self.ct_llr[vtx,:] = np.sum(self.ct_llr[self.ct.children(vtx),:], axis=0)


    def ct_attach_mut(self):
        self.ct.mut_loc = np.argmax(self.ct_llr, axis=0)
        
        # for reversible mutations, check if the inverse mutation has better log-likelihood
        for j in self.reversible:
            alt_loc = np.argmin(self.ct_llr[:,j]) # when flipped
            if self.locus_joint_1[j] + self.ct_llr[self.ct.mut_loc[j], j] < self.locus_joint_2[j] - self.ct_llr[alt_loc, j]:
                self.flip_direction(j)
                self.ct.mut_loc[j] = alt_loc
        
        # if best attachment still worse than L1, we want the mutation to attach to nowhere
        #for j in range(self.n_mut):
        #    if self.ct_LLR[self.ct.attachments[j], j] < 0:
        #        # any negative integer represents "outside of the tree", here we use -1
        #        self.ct.attachments[j] = -1
    

    def update_ct(self):
        self.update_ct_llr()
        self.ct_attach_mut()
    

    def fit(self, llh1, llh2, reversible=True):
        '''
        Gets ready to run optimization using provided likelihoods.
        Before calling this function, it is assumed that for each locus, the two genotypes gt1 and gt2 are known.
        [Arguments]
            llh1: 2D array in which entry [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh2: 2D array in which entry [i,j] is the log-likelihood of cell i having gt2 at locus j
            reversible: either an array that indicates whether each of the mutations is reversible (i.e. direction unknown),
                or a boolean value that applies to all loci.
                When a mutation is not reversible, the direction is assumed to be from gt1 to gt2.
        '''

        self.n_cells, self.n_mut = llh1.shape
        if self.n_cells < 3 or self.n_mut < 2:
            warnings.warn('[TreeOptimizer.fit] cell tree / mutation tree too small, nothing to explore', RuntimeWarning)
        
        #self.llh1, self.llh2 = llh1.copy(), llh2.copy()
        self.n_decimals = self.sig_digits - int(np.log10(np.abs(np.sum(llh1)))) # round tree likelihoods to this precision
        # Need to round because numpy.sum can give slightly different results when summing a matrix along different axis
        # See the "Notes" section in this page: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # If not rounded, the likelihood might increase when converting between the two tree spaces
        # Sometimes this traps the optimization in an infinite search

        self.direction = [1] * self.n_mut

        self.ct = CellTree(self.n_cells, self.n_mut)

        self.ct_llr = np.empty((self.ct.n_vtx, self.n_mut))
        self.ct_llr[:self.n_cells,:] = llh2 - llh1
        self.locus_joint_1 = np.sum(llh1, axis=0)
        self.locus_joint_2 = np.sum(llh2, axis=0)

        if reversible == False:
            self.reversible = []
        elif reversible == True:
            self.reversible = [j for j in range(self.n_mut)]
        else:
            self.reversible = reversible
        
        self.update_ct()


    def ct_greedy_split_merge(self, subroots):
        '''
        Prune subtrees and re-insert them to best possible locations
        [Arguments]
            subroots: iterable containing subroots to be pruned
        '''

        #for loc1, loc2 in permutations(prune_locs, 2):
        #    if self.ct.parent[loc1] == loc2:
        #        raise warnings.warn('Pruning parent and child simultaneously.')
        
        for sr in subroots:
            self.ct.binary_prune(sr)
        
        self.update_ct_llr()
        
        def search_insertion_loc(target):
            # calculate LLR at anchor when anchor is inserted above target
            self.ct_llr[anchor,:] = self.ct_llr[sr,:] + self.ct_llr[target,:]
            # highest achievable joint log-likelihood with this insertion
            self.ct_attach_mut()

            best_target = target
            best_joint = self.ct_joint

            if not self.ct.isleaf(target):
                # for any descendant of target, the LLR at target is the original one plus that of subroot
                self.ct_llr[target,:] += self.ct_llr[sr,:]
                # recursively search all descendants
                for child in self.ct.children(target):
                    child_best_target, child_best_joint = search_insertion_loc(child)
                    if child_best_joint > best_joint:
                        best_target = child_best_target
                        best_joint = child_best_joint
                # restore original the LLR at target after searching all descendants
                self.ct_llr[target,:] -= self.ct_llr[sr,:]
            
            return best_target, best_joint

        # insert the pruned subtrees one by one
        for sr in subroots:
            anchor = self.ct.parent(sr)
            best_target, best_joint = search_insertion_loc(self.ct.main_root)
            self.ct.binary_insert(anchor, best_target)
            self.update_ct()
        
        assert(best_joint == self.ct_joint)
    

    def propose_ct_prune_loc(self, n, leaf_only=False):
        if leaf_only:
            subroots = np.random.choice(self.ct.n_cells, size=n, replace=False)
        else:
            subroots = []
            for i in range(n):
                exclude = self.ct.roots + [self.ct.parent(sr) for sr in subroots]
                new_sr = randint_with_exclude(self.ct.n_vtx, exclude)
                subroots.append(new_sr)
                exclude += [new_sr, self.ct.parent(new_sr)] + self.ct.children(new_sr)
        
        return subroots
    

    def optimize(self, n_prunes=2, n_steps=20, print_likelihod=True):
        for i in range(n_steps):
            subroots = self.propose_ct_prune_loc(n_prunes)
            self.ct_greedy_split_merge(subroots)
            if print_likelihod:
                print(self.ct_joint)




def best_ct_llh(ct, llh_1, llh_2):
    ''' Returns the highest possible log-likelihood of a known cell tree '''
    optz = CellTree()
    optz.fit(llh_2, llh_1, reversible=True)
    optz.ct = ct
    #optz.ct.n_mut = optz.n_mut
    optz.update_ct()
    return optz.ct_joint


def test_greedy_split_merge():
    optimizer = TreeOptimizer2()
    llh1 = np.array([[-2, -1, -2], [-2, -1, -2], [-1, -1, -2], [-1, -2, -1], [-1, -2, -1]])
    llh2 = np.array([[-1, -2, -1], [-1, -2, -1], [-2, -2, -1], [-2, -1, -2], [-2, -1, -2]])
    optimizer.fit(llh1, llh2, reversible=False)
    optimizer.ct.parent_vec = [6, 5, 5, 7, 7, 6, 8, 8, -1]
    optimizer.update_ct()
    assert(optimizer.ct_joint == -16.0)

    optimizer.ct_greedy_split_merge([1])
    assert(optimizer.ct_joint == -15.0)
    assert(all(optimizer.ct.parent_vec == [5, 5, 6, 7, 7, 6, 8, 8, -1]))

