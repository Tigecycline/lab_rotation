import numpy as np

from .cell_tree import CellTree




def leaf_dist_mat(ct, unrooted=False):
    ''' Distance matrix for all leaves in a cell lineage tree '''
    result = - np.ones((ct.n_cells, ct.n_vtx), dtype = int)
    np.fill_diagonal(result, 0)
    for vtx in ct.rdfs(ct.main_root):
        if ct.isleaf(vtx):
            continue
        dist_growth = 1 if unrooted and vtx == ct.main_root else 2
        child1, child2 = ct.children(vtx)
        for leaf1 in ct.leaves(child1):
            for leaf2 in ct.leaves(child2):
                result[leaf1, vtx] = result[leaf1, child1] + 1
                result[leaf2, vtx] = result[leaf2, child2] + 1
                dist = result[leaf1, child1] + result[leaf2, child2] + dist_growth
                result[leaf1, leaf2] = dist
                result[leaf2, leaf1] = dist
    
    return result[:,:ct.n_cells]


def path_len_dist(ct1, ct2, unrooted=False):
    '''
    MSE between the distance matrices of two cell/mutation trees 
    NB The MSE is with respect to the upper triangle (excluding the diagonal), not the entire matrix
    '''
    dist_mat1, dist_mat2 = leaf_dist_mat(ct1, unrooted), leaf_dist_mat(ct2, unrooted)
    denominator = (dist_mat1.size - dist_mat1.shape[0]) / 2
    return np.sum((dist_mat1 - dist_mat2)**2) / 2 / denominator


def best_llh(ct, llh_1, llh_2):
    test_ct = CellTree(*llh_1.shape)
    test_ct.parent_vec = ct.parent_vec
    test_ct.fit_llh(llh_1, llh_2)
    test_ct.update_all()

    return test_ct.joint
