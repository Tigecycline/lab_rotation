import warnings

import numpy as np
import graphviz

from .cell_tree import CellTree




def ct_from_insert_vec(insert_vec):
    ''' Construct a cell tree using an insertion vector '''
    n_cells = insert_vec.size + 1
    pvec = np.empty(2*n_cells-1, dtype=int)
    
    pvec[0] = -1
    for leaf_idx, target_idx in enumerate(insert_vec):
        leaf = leaf_idx + 1 # i-th value in insert_vec is the insertion target for leaf i+1
        # when inserting leaf i+1, there are 2i-1 choices, i.e. i leaves and i-1 internals
        target = target_idx - leaf + n_cells if target_idx > leaf_idx else target_idx
        anchor = n_cells + leaf_idx
        pvec[anchor] = pvec[target]
        pvec[leaf] = anchor
        pvec[target] = anchor

    ct = CellTree(n_cells)
    ct.parent_vec = pvec
    return ct


def all_insert_vec(length):
    '''
    Generator to traverse valid insertion vectors that can be used to construct a cell tree
    The insertion vector describes the process of constructing the cell tree by inserting cells one by one
    After inserting each cell, the tree grows by two vertices, so the next insertion has two more options
    '''
    def generate_arrays(int_array, idx):
        if idx == length:
            yield int_array
        else:
            for value in range(2*idx+1):
                int_array[idx] = value
                yield from generate_arrays(int_array, idx+1)
    
    array = np.empty(length, dtype=int)
    yield from generate_arrays(array, 0)


def exhaustive_best_tree(llh_1, llh_2):
    ''' Find the tree structure with best best likelihood by exhaustive searching '''
    n_cells = llh_1.shape[0]
    if n_cells >= 10:
        warnings.warn('With 10 or more cells, the tree space might be overly large for an exhasutive search.')

    best_joint = -np.inf
    for insert_vec in all_insert_vec(n_cells-1):
        ct = ct_from_insert_vec(insert_vec)
        ct.fit_llh(llh_1, llh_2)
        ct.update_all()
        joint = ct.joint
        if joint > best_joint:
            best_joint = joint
            best_ct = ct
    
    return best_ct


def best_operation_dgraph(llh_1, llh_2):
    n_cells = llh_1.shape[0]
    if n_cells >= 10:
        warnings.warn('With 10 or more cells, the tree space might be overly large for an exhasutive search.')
    
    dgraph = graphviz.Digraph(engine='neato')
    for insert_vec in all_insert_vec(n_cells-1):
        pass
