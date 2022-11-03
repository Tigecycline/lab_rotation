from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tree_inference import *
from data_generator import DataGenerator
from utilities import path_len_dist




def test_space_swap(n_cells, n_mut, mut_prop = 0.5, n_tests = 5, fnames = None): 
    n_mut_true = round(n_mut * mut_prop)
    print('Test of space swap with %i cells and %i mutations, of which %i are true mutations' % (n_cells, n_mut, n_mut_true))
    if fnames is None: 
        # data_name: 'L' = likelihood, 'D' = distance to real tree, 'T' = runtime
        # c = number of cells, m = number of true mutations, f = number of fake mutations
        fnames = ['space_swap_%s_%ic_%im_%if' % (data_name, n_cells, n_mut_true, n_mut - n_mut_true) for data_name in ['L','D','T']]
    
    spaces = [['c'], ['c', 'm'], ['m'], ['m', 'c']]
    n_settings = len(spaces)
    
    final_likelihoods = np.empty((n_tests, n_settings))
    final_dist = np.empty((n_tests, n_settings))
    runtime = np.empty((n_tests, n_settings))
    
    dg = DataGenerator(n_cells, 2 * n_mut)
    for i in range(n_tests): 
        # generate new tree & data for each test
        dg.random_tree() 
        dg.random_mutations(mut_prop = mut_prop)
        ref_raw, alt_raw = dg.generate_reads()
        
        # infer mutation types & get likelihood matrices
        ref, alt, gt1, gt2 = filter_mutations(ref_raw, alt_raw)
        likelihoods1, likelihoods2 = likelihood_matrices(ref, alt, gt1, gt2)
        
        # likelihood of true tree
        true_mean = mean_likelihood(dg.tree, likelihoods1, likelihoods2)
        
        
        for j in range(n_settings): 
            print('Running test %i/%i, setting %i/%i' % (i+1, n_tests, j+1, 4), end = '\r')
            optz = TreeOptimizer()
            optz.fit(likelihoods1, likelihoods2, reversible = True)
            start = time()
            optz.optimize(spaces = spaces[j], print_result = False)
            
            runtime[i,j] = time() - start
            final_likelihoods[i,j] = optz.ct_mean_likelihood - true_mean
            final_dist[i,j] = path_len_dist(optz.ct, dg.tree)
            
    
    print('All tests finished, saving results...')
    pd.DataFrame(data = final_likelihoods).to_csv('./test_results/' + fnames[0] + '.csv')
    pd.DataFrame(data = final_dist).to_csv('./test_results/s' + fnames[1] + '.csv')
    pd.DataFrame(data = runtime).to_csv('./test_results/' + fnames[2] + '.csv')
    #make_boxplot(final_likelihoods, 
    #             spaces, 
    #             'mean loglikelihood of inferred tree \n compared to real tree', 
    #             fnames[0] + '.png'
    #            )
    #make_boxplot(final_dist, 
    #             spaces, 
    #             'MSE of distance matrix', 
    #             fnames[1] + '.png'
    #            )
    #make_boxplot(runtime, 
    #             spaces, 
    #             'runtime (s)', 
    #             fnames[2] + '.png'
    #            )
    print('Done')

    
def make_boxplot(ax, data): 
    ax.set_facecolor('lightgray')
    bplot = ax.boxplot(data, patch_artist = True)
    for patch in bplot['boxes']: 
        patch.set_facecolor('lightblue')
    ax.yaxis.grid(color = 'white')
    return bplot





if __name__ == '__main__': 
    #test_space_swap(n_cells = 100, n_mut = 100, n_tests = 100)
    #test_space_swap(n_cells = 100, n_mut = 200, n_tests = 100)
    #test_space_swap(n_cells = 100, n_mut = 400, n_tests = 100)
    
    data = np.empty((3,3), dtype = pd.DataFrame)
    data[0,0] = pd.read_csv('./test_results/space_swap_L_100c_50m_50f.csv', index_col = 0).to_numpy()
    data[0,1] = pd.read_csv('./test_results/space_swap_L_100c_100m_100f.csv', index_col = 0).to_numpy()
    data[0,2] = pd.read_csv('./test_results/space_swap_L_100c_200m_200f.csv', index_col = 0).to_numpy()
    data[1,0] = pd.read_csv('./test_results/space_swap_D_100c_50m_50f.csv', index_col = 0).to_numpy()
    data[1,1] = pd.read_csv('./test_results/space_swap_D_100c_100m_100f.csv', index_col = 0).to_numpy()
    data[1,2] = pd.read_csv('./test_results/space_swap_D_100c_200m_200f.csv', index_col = 0).to_numpy()
    data[2,0] = pd.read_csv('./test_results/space_swap_T_100c_50m_50f.csv', index_col = 0).to_numpy()
    data[2,1] = pd.read_csv('./test_results/space_swap_T_100c_100m_100f.csv', index_col = 0).to_numpy()
    data[2,2] = pd.read_csv('./test_results/space_swap_T_100c_200m_200f.csv', index_col = 0).to_numpy()
    
    fig, axes = plt.subplots(3, 3, sharex = 'all', sharey = 'row', figsize = (12,10))
    titles = ['100 cells\n 50 true mutations\n 50 fake mutations',
              '100 cells\n 100 true mutations\n 100 fake mutations',
              '100 cells\n 200 true mutations\n 200 fake mutations'
             ]
    xlabels = ['C', 'CM', 'M', 'MC']
    ylabels = ['mean loglikelihood of inferred tree \n compared to real tree', 
               'MSE of distance matrix', 
               'runtime (s)'
              ]
    
    for i in range(3): 
        axes[i,0].set_ylabel(ylabels[i], fontsize = 11)
        for j in range(3): 
            bplot = make_boxplot(axes[i,j], data[i,j])
            axes[0,j].set_title(titles[j], fontsize = 11)
            axes[2,j].set_yscale('log') 
            axes[2,j].set_xticks(np.arange(1,5), labels = xlabels)
    
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig('./figures/space_swap.png')