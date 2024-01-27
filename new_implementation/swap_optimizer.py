import warnings
from itertools import permutations
import numpy as np

from .cell_tree import CellTree
from .mutation_tree import MutationTree
#from .mutation_filter import *
from .utilities import randint_with_exclude




class SwapOptimizer:
    def __init__(self, sig_digits=10):
        '''
        [Arguments]
            spaces: spaces that will be searched and the order of the search
                    'c' = cell tree space, 'm' = mutation tree space
                    default is ['c','m'], i.e. start with cell tree and search both spaces
            sig_dig: number of significant digits to use when calculating joint probability
        '''
        self.sig_digits = sig_digits
    

    @property
    def current_joint(self):
        return round(self.ct.joint, self.n_decimals)


    def fit_llh(self, llh_1, llh_2):
        self.ct = CellTree(llh_1.shape[0], llh_1.shape[1])
        self.ct.fit_llh(llh_1, llh_2)
        
        self.mt = MutationTree(llh_1.shape[1], llh_1.shape[0])
        self.mt.fit_llh(llh_1, llh_2)

        # determine a rounding precision for joint likelihood calculation
        mean_abs = np.sum(np.abs(llh_1 + llh_2)) / 2 # mean abs value when attaching mutations randomly
        self.n_decimals = int(self.sig_digits - np.log10(mean_abs))
        # Need to round because numpy.sum can give slightly different results when summing a matrix along different axis
        # See the "Notes" section in this page: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # If not rounded, the joint likelihood might increase when converting between the two tree spaces
        # Sometimes this traps the optimization in an infinite loop
    

    def optimize(self, max_loops=100):
        converged = [False, False]
        current_space = 0 # 0 for cell lineage tree, 1 for mutation tree

        loop_count = 0
        while not all(converged):
            if loop_count >= max_loops:
                print('Maximal loop number exceeded.')
                break
            loop_count += 1
            start_joint = self.current_joint
            if current_space == 0:
                print('Optimizing cell lineage tree ...')
                self.ct.exhaustive_optimize()
                self.mt.fit_cell_tree(self.ct)
                self.mt.update_all()
            else: # i.e. current_space == 1:
                print('Optimizing mutation tree ...')
                self.mt.exhaustive_optimize()
                self.ct.fit_mutation_tree(self.mt)
                self.ct.update_all()
            
            if start_joint < self.current_joint:
                converged[current_space] = False
            elif start_joint == self.current_joint:
                converged[current_space] = True
            else: # start_joint > self.current_joint
                raise RuntimeError('Observed decrease in joint likelihood.')
            
            current_space = 1 - current_space
        
        print('Both spaces converged.')

        




def best_ct_llh(ct, llh_1, llh_2):
    ''' Returns the highest possible log-likelihood of a known cell tree '''
    optz = CellTree()
    optz.fit(llh_2, llh_1, reversible=True)
    optz.ct = ct
    #optz.ct.n_mut = optz.n_mut
    optz.update_ct()
    return optz.ct_joint


def test_greedy_split_merge():
    optimizer = SwapOptimizer()
    llh1 = np.array([[-2, -1, -2], [-2, -1, -2], [-1, -1, -2], [-1, -2, -1], [-1, -2, -1]])
    llh2 = np.array([[-1, -2, -1], [-1, -2, -1], [-2, -2, -1], [-2, -1, -2], [-2, -1, -2]])
    optimizer.fit(llh1, llh2, reversible=False)
    optimizer.ct.parent_vec = [6, 5, 5, 7, 7, 6, 8, 8, -1]
    optimizer.update_ct()
    assert(optimizer.ct_joint == -16.0)

    optimizer.ct_greedy_split_merge([1])
    assert(optimizer.ct_joint == -15.0)
    assert(all(optimizer.ct.parent_vec == [5, 5, 6, 7, 7, 6, 8, 8, -1]))

