import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm

sys.path.append('../')
from scite_rna.tree_functions import path_len_dist
from scite_rna.mutation_filter import MutationFilter
from scite_rna.cell_tree import CellTree
from scite_rna.mutation_tree import MutationTree
from scite_rna.swap_optimizer import SwapOptimizer




def get_dist_data(path, n_tests=100):
    pair_dist = np.empty((n_tests, 3))
    likelihood_diff = np.empty((n_tests, 3))

    for i in tqdm(range(n_tests)):
        # construct tree objects
        true_parent_vec = np.loadtxt(os.path.join(path, f'parent_vec_{i}.txt'), dtype=int)
        dendro_parent_vec = np.loadtxt(os.path.join(path, f'dendro_parent_vec_{i}.txt'), dtype=int)
        sciterna_parent_vec = np.loadtxt(os.path.join(path, f'sciterna_parent_vec_{i}.txt'), dtype=int)
        
        n_cells = int((len(true_parent_vec) + 1) / 2)
        ct_true = CellTree(n_cells)
        ct_dendro = CellTree(n_cells)
        ct_sciterna = CellTree(n_cells)
        ct_random = CellTree(n_cells)

        ct_true.use_parent_vec(true_parent_vec)
        ct_dendro.use_parent_vec(dendro_parent_vec)
        ct_sciterna.use_parent_vec(sciterna_parent_vec)
        ct_random.rand_subtree()

        # calculate distances
        pair_dist[i,0] = path_len_dist(ct_true, ct_sciterna)
        pair_dist[i,1] = path_len_dist(ct_true, ct_dendro)
        pair_dist[i,2] = path_len_dist(ct_true, ct_random)
    
    return pair_dist


def get_likelihood_data(path, n_tests=100):
    likelihood_diff = np.empty((n_tests, 3))
    mf = MutationFilter()

    for i in tqdm(range(n_tests)):
        # get parent vectors
        true_parent_vec = np.loadtxt(os.path.join(path, f'parent_vec_{i}.txt'), dtype=int)
        dendro_parent_vec = np.loadtxt(os.path.join(path, f'dendro_parent_vec_{i}.txt'), dtype=int)
        sciterna_parent_vec = np.loadtxt(os.path.join(path, f'sciterna_parent_vec_{i}.txt'), dtype=int)

        # construct likelihood matrices
        ref = np.loadtxt(os.path.join(path, f'ref_{i}.txt'))
        alt = np.loadtxt(os.path.join(path, f'alt_{i}.txt'))
        selected = np.loadtxt(os.path.join(path, f'selected_loci_{i}.txt'), dtype=int)
        gt1, gt2 = np.loadtxt(os.path.join(path, f'inferred_mut_types_{i}.txt'), dtype=str)
        llh_1, llh_2 = mf.get_llh_mat(ref[:,selected], alt[:,selected], gt1, gt2)
        
        # prepare for joint calculation
        n_cells = int((len(true_parent_vec) + 1) / 2)
        ct = CellTree(n_cells)
        ct.fit_llh(llh_1, llh_2)

        # calculate differences in joint likelihood
        ct.use_parent_vec(true_parent_vec)
        ct.update_all()
        true_joint = ct.joint

        ct.use_parent_vec(sciterna_parent_vec)
        ct.update_all()
        likelihood_diff[i,0] = ct.joint - true_joint
        ct.use_parent_vec(dendro_parent_vec)
        ct.update_all()
        likelihood_diff[i,1] = ct.joint  - true_joint
        ct.rand_subtree()
        ct.update_all()
        likelihood_diff[i,2] = ct.joint - true_joint

    return likelihood_diff / (ct.n_cells * ct.n_mut)


def make_boxplot(ax, data, colors=None, positions=None):
    bplot = ax.boxplot(data, patch_artist = True, positions = positions)
    if colors is None:
        colors = 'lightblue'
    if type(colors) == str:
        colors = [colors] * data.shape[0]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_facecolor('lightgray')
    ax.yaxis.grid(color = 'white')
    ax.set_xticklabels(['true vs. scite-rna', 'true vs. dendro', 'true vs. random'])
    return bplot


def make_freq_histogram(ax, data, xlim, colors=None, labels=None):
    if colors is None:
        colors = ['red', 'yellow', 'blue']
    for i in range(3):
        ax.hist(data[:,i], bins=50, range=xlim, weights=np.ones(data.shape[0]) / len(data), histtype='bar', alpha=0.75, color=colors[i])

    if labels is not None:
        ax.legend(labels=labels)

    ax.set_axisbelow(True)
    ax.set_facecolor('lightgray')
    ax.yaxis.grid(color = 'white')


#def make_hist_grid():
#    fig, axes = plt.subplots()

    





if __name__ == '__main__':
    n_tests = 100
    n_cells_list = [50, 50, 100, 100]
    n_mut_list = [200, 400, 200, 400]
    
    for n_cells, n_mut in zip(n_cells_list, n_mut_list):
        fname = f'./figures/pair_dist_{n_cells}c{n_mut}m.txt'
        if not os.path.exists(fname):
            print(f'Pairwise distance {n_cells} cells {n_mut} mutations')
            path = f'./comparison_data/{n_cells}c{n_mut}m'
            pair_dist = get_dist_data(path)
            np.savetxt(fname, pair_dist)
        fname = f'./figures/likelihood_diff_{n_cells}c{n_mut}m.txt'
        if not os.path.exists(fname):
            print(f'Likelihood difference {n_cells} cells {n_mut} mutations')
            path = f'./comparison_data/{n_cells}c{n_mut}m'
            pair_dist = get_likelihood_data(path)
            np.savetxt(fname, pair_dist)

    # make plots
    labels = ['SCITE-RNA', 'DENDRO', 'random']
    colors = ['red', 'green', 'blue']

    # plot distance to real tree
    xlim = (0, 50)
    fig, axes = plt.subplots(2, 2, figsize=(8,8), sharex=True, sharey=True)

    pair_dist = np.loadtxt('./figures/pair_dist_50c200m.txt')
    make_freq_histogram(axes[0,0], pair_dist, xlim, colors=colors, labels=labels)
    pair_dist = np.loadtxt('./figures/pair_dist_50c400m.txt')
    make_freq_histogram(axes[0,1], pair_dist, xlim, colors=colors, labels=labels)
    pair_dist = np.loadtxt('./figures/pair_dist_100c200m.txt')
    make_freq_histogram(axes[1,0], pair_dist, xlim, colors=colors, labels=labels)
    pair_dist = np.loadtxt('./figures/pair_dist_100c400m.txt')
    make_freq_histogram(axes[1,1], pair_dist, xlim, colors=colors, labels=labels)

    pad = 5
    axes[0,1].annotate('50 cells', xy=(1, 0.5), xytext=(pad, 0),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='left', va='center')
    axes[1,1].annotate('100 cells', xy=(1, 0.5), xytext=(pad, 0),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='left', va='center')
    axes[0,0].annotate('200 mutations', xy=(0.5, 1), xytext=(0, pad),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    axes[0,1].annotate('400 mutations', xy=(0.5, 1), xytext=(0, pad),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')

    fig.supxlabel('path length distance to real tree')
    fig.supylabel('frequency')
    fig.tight_layout()
    fig.savefig('./figures/distances_hist.pdf')

    # plot log-likelihood compared to real tree
    xlim = (-0.4, 0.1)
    fig, axes = plt.subplots(2, 2, figsize=(8,8), sharex=True, sharey=True)

    llh_diff = np.loadtxt('./figures/likelihood_diff_50c200m.txt')
    make_freq_histogram(axes[0,0], llh_diff, xlim, colors=colors, labels=labels)
    llh_diff = np.loadtxt('./figures/likelihood_diff_50c400m.txt')
    make_freq_histogram(axes[0,1], llh_diff, xlim, colors=colors, labels=labels)
    llh_diff = np.loadtxt('./figures/likelihood_diff_100c200m.txt')
    make_freq_histogram(axes[1,0], llh_diff, xlim, colors=colors, labels=labels)
    llh_diff = np.loadtxt('./figures/likelihood_diff_100c400m.txt')
    make_freq_histogram(axes[1,1], llh_diff, xlim, colors=colors, labels=labels)

    pad = 5
    axes[0,1].annotate('50 cells', xy=(1, 0.5), xytext=(pad, 0),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='left', va='center')
    axes[1,1].annotate('100 cells', xy=(1, 0.5), xytext=(pad, 0),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='left', va='center')
    axes[0,0].annotate('200 mutations', xy=(0.5, 1), xytext=(0, pad),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    axes[0,1].annotate('400 mutations', xy=(0.5, 1), xytext=(0, pad),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')

    fig.supxlabel('log-likelihood per cell and locus compared to real tree')
    fig.supylabel('frequency')
    fig.tight_layout()
    fig.savefig(f'./figures/likelihood_diff_hist.pdf')