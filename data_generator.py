from tree import *
from scipy.stats import poisson, betabinom
from mutation_detection import get_k_mut_priors




class DataGenerator: 
    #genotypes = ['R', 'A', 'H']
    mutation_types = ['RH', 'AH', 'HR', 'HA']
    
    
    def __init__(self, priors = np.ones(4) / 4, f = 0.95, omega = 100, omega_h = 50, coverage_sampler = lambda : poisson.rvs(mu = 8), cell_tree = None):
        self.priors = priors
        self.alpha = f * omega
        self.beta = omega - self.alpha
        self.omega_h = omega_h
        self.coverage_sampler = coverage_sampler  
        self.tree = cell_tree
    
    
    def random_mutations(self, n_mut): 
        mutations = np.random.choice(DataGenerator.mutation_types, size = n_mut, replace = True, p = self.priors)
        gt1 = np.array([mut[0] for mut in mutations], dtype = str)
        gt2 = np.array([mut[1] for mut in mutations], dtype = str)
        return gt1, gt2
    
    
    def random_tree(self, n_cells, n_mut): 
        self.tree = CellTree(n_cells, n_mut)
        mut_locations = np.random.choice(self.tree.n_nodes, size = n_mut, replace = True)
        for i in range(n_mut): 
            self.tree.nodes[mut_locations[i]].add_mutation(i)
                
    
    def generate_single_read(self, genotype, coverage): 
        if genotype == 'R': 
            n_ref = betabinom.rvs(coverage, self.alpha, self.beta)
            n_alt = coverage - n_ref
        elif genotype == 'A': 
            n_alt = betabinom.rvs(coverage, self.alpha, self.beta)
            n_ref = coverage - n_alt
        elif genotype == 'H': 
            n_ref = betabinom.rvs(coverage, self.omega_h / 2, self.omega_h / 2)
            n_alt = coverage - n_ref
        else: 
            print('[generate_single_read] ERROR: invalid genotype.')
            return 0, 0
        return n_ref, n_alt
    
    
    def generate_reads(self, n_cells, n_mut): 
        gt1, gt2 = self.random_mutations(n_mut)
        if self.tree is None: 
            self.random_tree(n_cells, n_mut)
        genotypes = np.empty((n_cells, n_mut), dtype = str)
        for i in range(n_cells):
            cell = self.tree.nodes[i]  
            mutated = np.zeros(n_mut, dtype = bool)
            mutated[cell.mutations] = True
            for ancestor in cell.ancestors: 
                mutated[ancestor.mutations] = True
            genotypes[i,:] = np.where(mutated, gt2, gt1)
        
        ref = np.empty((n_cells, n_mut))
        alt = np.empty((n_cells, n_mut))
        for i in range(n_cells): 
            for j in range(n_mut):
                coverage = self.coverage_sampler()
                ref[i,j], alt[i,j] = self.generate_single_read(genotypes[i,j], coverage)
        
        return alt, ref, gt1, gt2, genotypes
        

        


'''
def coverage_sampler(): 
    return int(np.random.choice(coverages))
dg = DataGenerator(coverage_sampler = coverage_sampler)
ref_test, alt_test = dg.generate_reads(mutations = ['AH' for i in range(3)])
'''