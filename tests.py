from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from tree_inference import *
from data_generator import DataGenerator
from utilities import path_len_dist




def test_space_swap(n_cells, n_mut, n_runs = 5): 
    spaces = [['c'], ['c', 'm'], ['m'], ['m', 'c']]
    
    n_settings = len(spaces)
    
    final_likelihoods = np.empty((n_runs, n_settings))
    final_dist = np.empty((n_runs, n_settings))
    runtime = np.empty((n_runs, n_settings))
    
    dg = DataGenerator(n_cells, 2 * n_mut, mut_prop = 0.5)
    for i in range(n_runs): 
        # generate new tree & data for each test
        dg.random_tree() 
        dg.random_mutations()
        ref_raw, alt_raw = dg.generate_reads()
        
        # infer mutation types & get likelihood matrices
        ref, alt, gt1, gt2 = filter_mutations(ref_raw, alt_raw)
        likelihoods1, likelihoods2 = likelihood_matrices(ref, alt, gt1, gt2)
        
        # likelihood of true tree
        true_mean = mean_likelihood(dg.tree, likelihoods1, likelihoods2)
        
        
        for j in range(4): 
            print('Running test %i/%i, setting %i/%i' % (i+1, n_runs, j+1, 4), end = '\r')
            optz = TreeOptimizer()
            optz.fit(likelihoods2, likelihoods1, reversible = True)
            start = time()
            optz.optimize(spaces = spaces[j], print_result = False)
            
            runtime[i,j] = time() - start
            final_likelihoods[i,j] = optz.ct_mean_likelihood - true_mean
            final_dist[i,j] = path_len_dist(optz.ct, dg.tree)
            
    
    print('All tests finished, making boxplots...')
    pd.DataFrame(data = final_likelihoods).to_csv('./figures/space_swap_1.csv')
    pd.DataFrame(data = final_dist).to_csv('./figures/space_swap_2.csv')
    pd.DataFrame(data = runtime).to_csv('./figures/space_swap_3.csv')
    
    
    def make_boxplot(data, ylabel, fname): 
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_facecolor('lightgray')
        bplot = ax.boxplot(data, patch_artist = True, labels = spaces)
        for patch in bplot['boxes']: 
            patch.set_facecolor('lightblue')
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(color = 'white')
        fig.tight_layout()
        fig.savefig('./figures/' + fname)
    
    make_boxplot(final_likelihoods, 
                 'mean loglikelihood of inferred tree \n compared to real tree', 
                 'space_swap_1.png'
                )
    make_boxplot(final_dist, 
                 'MSE of distance matrix', 
                 'space_swap_2.png'
                )
    make_boxplot(runtime, 
                 'runtime (s)', 
                 'space_swap_3.png'
                )
    
    print('Done')

    






if __name__ == '__main__': 
    test_space_swap(n_cells = 100, n_mut = 200, n_runs = 100)