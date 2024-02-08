import sys, os
from tqdm import tqdm
sys.path.append('../')

from scite_rna.utilities import *
from scite_rna.data_generator import DataGenerator
from scite_rna.mutation_filter import MutationFilter
from scite_rna.swap_optimizer import SwapOptimizer




def generate_comparison_data(n_cells: int, n_mut: int, size=100, mut_prop=0.75, path='./comparison_data/', seed=None):
    if seed is not None:
        np.random.seed(seed)
    if os.path.exists(path):
        while True:
            ans = input(f'Directory {path} already exists. Existing files will be overwritten. Continue? [Y/N] ')
            match ans:
                case 'Y' | 'y' | 'Yes' | 'yes':
                    break
                case 'N' | 'n' | 'No' | 'no':
                    return
    else:
        os.makedirs(path)
    
    generator = DataGenerator(n_cells, n_mut)
    
    for i in tqdm(range(size)):
        ref, alt = generator.generate_reads()

        mut_indicator = np.zeros((n_cells, n_mut), dtype=bool)
        for j in range(generator.n_mut):
            if generator.gt1[j] == generator.gt2[j]:
                continue # not mutated
            for leaf in generator.ct.leaves(generator.ct.mut_loc[j]):
                mut_indicator[leaf, j] = True
        
        np.savetxt(os.path.join(path, f'ref_{i}.txt'), ref, fmt='%i')
        np.savetxt(os.path.join(path, f'alt_{i}.txt'), alt, fmt='%i')
        np.savetxt(os.path.join(path, f'parent_vec_{i}.txt'), generator.ct.parent_vec, fmt='%i')
        np.savetxt(os.path.join(path, f'mut_indicator_{i}.txt'), mut_indicator, fmt='%i')


def generate_sciterna_results(path='./comparison_data/', n_tests=100):
    optimizer = SwapOptimizer()

    print(f'Running inference on data in {path}')
    for i in tqdm(range(n_tests)):
        ref = np.loadtxt(os.path.join(path, 'ref_%i.txt' % i))
        alt = np.loadtxt(os.path.join(path, 'alt_%i.txt' % i))
        
        mf = MutationFilter()
        selected, gt1, gt2 = mf.filter_mutations(ref, alt, method='threshold', t=0.5)
        llh_1, llh_2 = mf.get_llh_mat(ref[:,selected], alt[:,selected], gt1, gt2)

        np.savetxt(os.path.join(path, f'selected_loci_{i}.txt'), selected, fmt='%i')
        np.savetxt(os.path.join(path, f'inferred_mut_types_{i}.txt'), np.stack((gt1, gt2), axis=0), fmt='%s')

        optimizer.fit_llh(llh_1, llh_2)
        optimizer.optimize()

        np.savetxt(os.path.join(path, f'sciterna_parent_vec_{i}.txt'), optimizer.ct.parent_vec, fmt='%i')
    print('Done.')




if __name__ == '__main__':
    n_tests = 100
    n_cells_list = [50, 50, 100, 100]
    n_mut_list = [200, 400, 200, 400]
    
    for n_cells, n_mut in zip(n_cells_list, n_mut_list):
        path = f'./comparison_data/{n_cells}c{n_mut}m'
        #generate_comparison_data(n_cells, n_mut, n_tests, path=path)
        generate_sciterna_results(path, n_tests)
