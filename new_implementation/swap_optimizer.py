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
        # Need to round because the sum of floating point numbers can slightly vary depending on the exact process
        # For example, numpy.sum gives different results when summing over different axes
        # See the "Notes" section in this page: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # If not rounded, the joint likelihood might decrease when converting between the two tree spaces
        # Sometimes this traps the optimization in an infinite loop
    

    def optimize(self, max_loops=100):
        converged = [False, False]
        current_space = 0 # 0 for cell lineage tree, 1 for mutation tree

        loop_count = 0
        while not all(converged):
            if loop_count >= max_loops:
                warnings.warn('Maximal loop number exceeded.')
                break
            loop_count += 1
            start_joint = self.current_joint
            if current_space == 0:
                #print('Optimizing cell lineage tree ...')
                self.ct.exhaustive_optimize()
                self.mt.fit_cell_tree(self.ct)
                self.mt.update_all()
            else: # i.e. current_space == 1:
                #print('Optimizing mutation tree ...')
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
        
        #print('Both spaces converged.')