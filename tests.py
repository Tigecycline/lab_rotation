from time import time
import numpy as np

from tree_inference import *
from data_generator import DataGenerator
from utilities import path_len_dist




def test_space_swap(n_cells, n_loci, mut_prop = 0.5, n_tests = 5, fnames = None): 
    n_mut = round(n_loci * mut_prop)
    print('Space swap test with %i cells and %i loci, of which %i are mutated:' % (n_cells, n_loci, n_mut))
    
    spaces = [['c'], ['c', 'm'], ['m'], ['m', 'c']]
    n_settings = len(spaces)
    
    final_likelihoods = np.empty((n_tests, n_settings))
    final_dist = np.empty((n_tests, n_settings))
    runtime = np.empty((n_tests, n_settings))
    
    dg = DataGenerator(n_cells, 2 * n_loci)
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
    if fnames is None: 
        # data_name: 'L' = likelihood, 'D' = distance to real tree, 'T' = runtime
        # c = number of cells, m = number of true mutations, f = number of fake mutations
        fnames = ['space_swap_%s_%ic_%im_%if' % (data_name, n_cells, n_mut, n_loci - n_mut) for data_name in ['L','D','T']]
    np.save('./test_results/' + fnames[0] + '.npy', final_likelihoods)
    np.save('./test_results/' + fnames[1] + '.npy', final_dist)
    np.save('./test_results/' + fnames[2] + '.npy', runtime)
    print('Done')


def test_mutation_detection(n_cells, n_loci, mut_prop, n_tests = 5, sampler = None, fnames = None): 
    sampler_name = 'def' if sampler is None else 'oth' # def = default sampler, oth  = other samplers
    print('Mutation detection test with %i cells and the %s coverage sampler:' % (n_cells, sampler_name))

    tp = np.zeros((n_tests, n_loci + 2))
    fp = np.zeros((n_tests, n_loci + 2))
    tp_alt = np.empty(n_tests)
    fp_alt = np.empty(n_tests)
    sorted_posteriors = np.empty((n_tests, n_loci))

    n_pos = round(n_loci * mut_prop)
    n_neg = n_loci - n_pos

    dg = DataGenerator(n_cells, n_loci, coverage_sampler = sampler)
    for i in range(n_tests): 
        print('Running test %i/%i' % (i+1, n_tests), end = '\r')
        # generate random tree
        dg.random_tree()
        dg.random_mutations(mut_prop = mut_prop)
        ref, alt = dg.generate_reads()
        # calculate posteriors
        posteriors = mut_type_posteriors(ref, alt)
        mutated_posterior = np.sum(posteriors[:,3:], axis = 1)
        order = np.argsort(mutated_posterior)
        sorted_posteriors[i,:] = mutated_posterior[order]
        # all loci are positive if threshold is zero
        tp[i,0] = n_pos
        fp[i,0] = n_neg
        
        pos_real = dg.gt1 != dg.gt2
        for j in range(n_loci): 
            # if the locus with j-th smallest posterior is mutated
            # then number of true positives reduces by 1 at this threshold
            # otherwise the number of false positives reduces by 1
            if pos_real[order[j]]: 
                tp[i,j+1] = tp[i,j] - 1
                fp[i,j+1] = fp[i,j]
            else: 
                tp[i,j+1] = tp[i,j]
                fp[i,j+1] = fp[i,j] - 1
        
        pos_alt = np.argmax(posteriors, axis = 1) >= 3
        tp_alt[i] = np.sum(np.logical_and(pos_real, pos_alt))
        fp_alt[i] = np.sum(pos_alt) - tp_alt[i]
    
    print('All tests finished, saving results...')
    if fnames is None: 
        data_names = ['sortedP', 'TPR', 'FPR', 'altTPR', 'altFPR']
        fnames = ['mut_detection_%s_%ic_%im_%if_%s' % (dn, n_cells, n_pos, n_neg, sampler_name) for dn in data_names]
    np.save('./test_results/' + fnames[0], sorted_posteriors) 
    np.save('./test_results/' + fnames[1], tp / n_pos) # true positive rates
    np.save('./test_results/' + fnames[2], fp / n_neg) # false positive rates
    np.save('./test_results/' + fnames[3], tp_alt / n_pos) # true positive rate, alternative method
    np.save('./test_results/' + fnames[4], fp_alt / n_neg) # false positive rate, alternative method
    print('Done')


def test_reversibility(n_cells, n_loci, mut_prop = 0.5, n_tests = 5): 
    n_mut = round(n_loci * mut_prop)
    print('Reversibility test with %i cells and %i loci, of which %i are mutated:' % (n_cells, n_loci, n_mut))
    dg = DataGenerator(n_cells, n_loci)

    n_settings = 4
    final_likelihoods = np.empty((n_tests,n_settings))
    final_dist = np.empty((n_tests,n_settings))
    runtime = np.empty((n_tests,n_settings))

    for i in range(n_tests):
        print('Running test %i/%i' % (i+1, n_tests), end = '\r')
        dg.random_tree()
        dg.random_mutations(mut_prop = mut_prop)
        ref_raw, alt_raw = dg.generate_reads()

        ref, alt, gt1, gt2 = filter_mutations(ref_raw, alt_raw)

        mutated = dg.gt1 != dg.gt2
        gt1_true, gt2_true = dg.gt1[mutated], dg.gt2[mutated]
        ref_true, alt_true = ref_raw[mutated,:], alt_raw[mutated,:]





        
        





if __name__ == '__main__':
    #test_space_swap(n_cells = 100, n_loci = 100, n_tests = 100)
    #test_space_swap(n_cells = 100, n_loci = 200, n_tests = 100)
    #test_space_swap(n_cells = 100, n_loci = 400, n_tests = 100)

    #ref, alt = read_data('./Data/glioblastoma_BT_S2/ref.csv', './Data/glioblastoma_BT_S2/alt.csv')
    #alt_sampler = coverage_sampler(ref, alt)
    #test_mutation_detection(25, 200, 0.5, n_tests = 100)
    #test_mutation_detection(100, 200, 0.5, n_tests = 100)
    #test_mutation_detection(400, 200, 0.5, n_tests = 100)
    #test_mutation_detection(25, 200, 0.5, n_tests = 100, sampler = alt_sampler)
    #test_mutation_detection(100, 200, 0.5, n_tests = 100, sampler = alt_sampler)
    #test_mutation_detection(400, 200, 0.5, n_tests = 100, sampler = alt_sampler)
    pass