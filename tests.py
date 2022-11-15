from time import time
import numpy as np
from pathlib import Path
import multiprocessing as mp

from tree_inference import *
from data_generator import DataGenerator
from mutation_detection import *
from utilities import *




def test_mutation_detection(n_cells, n_loci, mut_prop, n_tests = 5, sampler = None, fnames = None): 
    sampler_name = 'def' if sampler is None else 'oth' # def = default sampler, oth  = other samplers
    n_mut = round(n_loci * mut_prop)
    print('Mutation detection test with %i cells and the %s coverage sampler:' % (n_cells, sampler_name))

    tpr = np.empty((n_tests, n_loci + 1))
    fpr = np.empty((n_tests, n_loci + 1))
    tpr_alt = np.empty(n_tests)
    fpr_alt = np.empty(n_tests)
    sorted_posteriors = np.empty((n_tests, n_loci))

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
        
        tp = np.empty(n_loci + 1)
        fp = np.empty(n_loci + 1)
        # real positives: loci that are mutated and the mutation affects at least two cells
        pos_real = np.logical_and(dg.gt1 != dg.gt2, dg.tree.attachments >= dg.n_cells) 
        # all loci are positive if threshold is zero
        n_pos_real = np.sum(pos_real)
        n_neg_real = n_loci - n_pos_real
        tp[0] = n_pos_real
        fp[0] = n_neg_real

        for j in range(n_loci): 
            # if the locus with j-th smallest posterior is mutated
            # then number of true positives reduces by 1 at this threshold
            # otherwise the number of false positives reduces by 1
            if pos_real[order[j]]: 
                tp[j+1] = tp[j] - 1
                fp[j+1] = fp[j]
            else: 
                tp[j+1] = tp[j]
                fp[j+1] = fp[j] - 1
        tpr[i,:] = tp / n_pos_real
        fpr[i,:] = fp / n_neg_real

        pos_alt = np.argmax(posteriors, axis = 1) >= 3
        tp_alt = np.sum(np.logical_and(pos_real, pos_alt))
        fp_alt = np.sum(pos_alt) - tp_alt
        tpr_alt[i] = tp_alt / n_pos_real
        fpr_alt[i] = fp_alt / n_neg_real
    
    print('All tests finished, saving results...')
    if fnames is None: 
        data_names = ['sortedP', 'TPR', 'FPR', 'altTPR', 'altFPR']
        fnames = ['mut_detection_%s_%ic_%im_%if_%s' % (dn, n_cells, n_mut, n_loci - n_mut, sampler_name) for dn in data_names]
    np.save('./test_results/' + fnames[0], sorted_posteriors) 
    np.save('./test_results/' + fnames[1], tpr) # true positive rates
    np.save('./test_results/' + fnames[2], fpr) # false positive rates
    np.save('./test_results/' + fnames[3], tpr_alt) # true positive rate, alternative method
    np.save('./test_results/' + fnames[4], fpr_alt) # false positive rate, alternative method
    print('Done')


def test_space_swap(n_cells, n_loci, mut_prop = 0.5, n_tests = 5, outdir = None): 
    n_mut = round(n_loci * mut_prop)
    print('Space swap test with %i cells and %i loci, of which %i are mutated:' % (n_cells, n_loci, n_mut))
    
    spaces = [['c'], ['m'], ['c', 'm'], ['m', 'c']]
    n_settings = len(spaces)
    
    likelihoods = np.empty((n_tests, n_settings))
    dist = np.empty((n_tests, n_settings))
    runtime = np.empty((n_tests, n_settings))
    
    dg = DataGenerator(n_cells, n_loci, coverage_sampler = coverage_sampler())
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
            print('Running test %i/%i, setting %i/%i' % (i+1, n_tests, j+1, n_settings), end = '\r')
            optz = TreeOptimizer()
            optz.fit(likelihoods1, likelihoods2, reversible = True)
            start = time()
            optz.optimize(spaces = spaces[j], print_info = False)
            
            runtime[i,j] = time() - start
            likelihoods[i,j] = optz.ct_mean_likelihood - true_mean
            dist[i,j] = path_len_dist(optz.ct, dg.tree)
    
    print('All tests finished, saving results...')
    if outdir is None:
        outdir = './test_results/spaceswap_%ic_%im_%if/' % (n_cells, n_mut, n_loci - n_mut)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    np.save(outdir + 'likelihoods.npy', likelihoods)
    np.save(outdir + 'dist.npy', dist)
    np.save(outdir + 'runtime.npy', runtime)
    print('Done')


def test_thresholds(n_cells, n_loci, mut_prop = 0.5, sampler = None, n_tests = 5, outdir = None):
    n_mut = round(n_loci * mut_prop)
    print('Threshold test with %i cells and %i loci, of which %i are mutated:' % (n_cells, n_loci, n_mut))

    thresholds = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    n_settings = len(thresholds)
    dist_rev = np.empty((n_tests,n_settings))
    dist_irr = np.empty((n_tests,n_settings))
    n_selected = np.empty((n_tests,n_settings))
    runtime_rev = np.empty((n_tests,n_settings))
    runtime_irr = np.empty((n_tests,n_settings))

    def test(i): 
        dg = DataGenerator(n_cells, n_loci, coverage_sampler = sampler)
        dg.random_tree()
        dg.random_mutations(mut_prop = mut_prop)

        ref_raw, alt_raw = dg.generate_reads()
        posteriors = mut_type_posteriors(ref_raw, alt_raw)
        mut_posteriors = np.sum(posteriors[:,3:], axis = 1)
        
        mut_type = np.argmax(posteriors[:,3:], axis = 1) 
        gt1 = np.choose(mut_type, choices = ['R', 'H', 'H', 'A'])
        gt2 = np.choose(mut_type, choices = ['H', 'A', 'R', 'H'])
        likelihoods1, likelihoods2 = likelihood_matrices(ref_raw, alt_raw, gt1, gt2)
        
        for j in range(n_settings):
            selected = np.where(mut_posteriors > thresholds[j])[0]
            n_selected[i,j] = selected.size
            
            #print('Running test %i/%i, setting %i/%i, rev' % (i+1, n_tests, j+1, n_settings), end = '\r')
            optz = TreeOptimizer()
            optz.fit(likelihoods1[:,selected], likelihoods2[:,selected], reversible = True)
            start = time()
            optz.optimize(print_info = False)
            
            runtime_rev[i,j] = time() - start
            dist_rev[i,j] = path_len_dist(optz.ct, dg.tree)

            #print('Running test %i/%i, setting %i/%i, irr' % (i+1, n_tests, j+1, n_settings), end = '\r')
            optz = TreeOptimizer()
            optz.fit(likelihoods1[:,selected], likelihoods2[:,selected], reversible = False)
            start = time()
            optz.optimize(print_info = False)
            
            runtime_irr[i,j] = time() - start
            dist_irr[i,j] = path_len_dist(optz.ct, dg.tree)
        print('Test %i finished.')
    
    pool = mp.Pool(4)
    for i in range(n_tests):
        pool.apply_async(test(i))

    print('All tests finished, saving results...')
    if outdir is None:
        outdir = './test_results/thresholds_%ic_%im_%if/' % (n_cells, n_mut, n_loci - n_mut)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    np.save(outdir + 'dist_rev.npy', dist_rev)
    np.save(outdir + 'dist_irr.npy', dist_irr)
    np.save(outdir + 'n_selected.npy', n_selected)
    np.save(outdir + 'runtime_rev.npy', runtime_rev)
    np.save(outdir + 'runtime_irr.npy', runtime_irr)
    print('Done')


def test_reversibility(n_cells, n_mut, n_tests = 5, outdir = None): 
    print('Reversibility test with %i cells and %i mutations:' % (n_cells, n_mut))
    dg = DataGenerator(n_cells, n_mut, coverage_sampler = coverage_sampler())

    prop_wrong = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_wrong = (np.array(prop_wrong) * n_mut).astype(int)

    n_settings = len(prop_wrong)
    likelihoods_rev = np.empty((n_tests,n_settings))
    likelihoods_irr = np.empty((n_tests,n_settings))
    dist_rev = np.empty((n_tests,n_settings))
    dist_irr = np.empty((n_tests,n_settings))

    for i in range(n_tests):
        dg.random_tree(one_cell_mut = False)
        dg.random_mutations(mut_prop = 1.0)
        ref, alt = dg.generate_reads()
        likelihoods1_true, likelihoods2_true = likelihood_matrices(ref, alt, dg.gt1, dg.gt2)
        likelihoods1, likelihoods2 = likelihoods1_true.copy(), likelihoods2_true.copy()
        true_mean = mean_likelihood(dg.tree, likelihoods1, likelihoods2)

        for j in range(n_settings):
            likelihoods1[:,:n_wrong[j]] = likelihoods2_true[:,:n_wrong[j]].copy()
            likelihoods2[:,:n_wrong[j]] = likelihoods1_true[:,:n_wrong[j]].copy()

            print('Running test %i/%i, setting %i/%i, rev' % (i+1, n_tests, j+1, n_settings), end = '\r')
            optz = TreeOptimizer()
            optz.fit(likelihoods1, likelihoods2, reversible = True)
            optz.optimize(print_info = False)

            likelihoods_rev[i,j] = optz.ct_mean_likelihood - true_mean
            dist_rev[i,j] = path_len_dist(optz.ct, dg.tree)
            
            print('Running test %i/%i, setting %i/%i, irr' % (i+1, n_tests, j+1, n_settings), end = '\r')
            optz = TreeOptimizer()
            optz.fit(likelihoods1, likelihoods2, reversible = False)
            optz.optimize(print_info = False)

            likelihoods_irr[i,j] = optz.ct_mean_likelihood - true_mean
            dist_irr[i,j] = path_len_dist(optz.ct, dg.tree)
    
    print('All tests finished, saving results...')
    if outdir is None:
        outdir = './test_results/reversibility_%ic_%im/' % (n_cells, n_mut)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    np.save(outdir + 'likelihoods_rev.npy', likelihoods_rev)
    np.save(outdir + 'likelihoods_irr.npy', likelihoods_irr)
    np.save(outdir + 'dist_rev.npy', dist_rev)
    np.save(outdir + 'dist_irr.npy', dist_irr)
    print('Done')

    
def test_random_dist(n_cells = None, n_tests = 100, fname = None):
    if n_cells is None:
        n_cells = [25, 50, 100, 200]
        
    dist = np.empty((n_tests, len(n_cells)))
    for j in range(len(n_cells)):
        ct1 = CellTree(n_cells[j])
        ct2 = CellTree(n_cells[j])
        for i in range(n_tests):
            ct1.randomize()
            ct2.randomize()
            dist[i,j] = path_len_dist(ct1, ct2)
    
    if fname is None:
        fname = 'random_dist'
    np.save('./test_results/' + fname + '.npy', dist)











if __name__ == '__main__':
    #alt_sampler = coverage_sampler()
    #test_mutation_detection(25, 200, 0.5, n_tests = 100)
    #test_mutation_detection(400, 200, 0.5, n_tests = 100)
    #test_mutation_detection(25, 200, 0.5, n_tests = 100, sampler = alt_sampler)
    #test_mutation_detection(400, 200, 0.5, n_tests = 100, sampler = alt_sampler)
    
    #test_space_swap(n_cells = 50, n_loci = 200, n_tests = 100)
    #test_space_swap(n_cells = 50, n_loci = 400, n_tests = 100)
    #test_space_swap(n_cells = 50, n_loci = 800, n_tests = 100)

    #test_thresholds(50, 400, 0.5, sampler = coverage_sampler(), n_tests = 100)

    test_reversibility(50, 100, 100)

    #test_random_dist()