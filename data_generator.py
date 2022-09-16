from tree import *
from scipy.stats import poisson, betabinom
from mutation_detection import get_k_mut_priors
from tqdm.notebook import tqdm




class DataGenerator: 
    #genotypes = ['R', 'A', 'H']
    mutation_types = ['RH', 'AH', 'HR', 'HA']
    
    
    def __init__(self, n_cells = 8, n_mut = 16, priors = np.ones(4) / 4, f = 0.95, omega = 100, omega_h = 50, coverage_sampler = None, cell_tree = None, gt1 = None, gt2 = None, forbid_root_mut = False):
        self.priors = priors
        self.alpha = f * omega
        self.beta = omega - self.alpha
        self.omega_h = omega_h
        if coverage_sampler is None: 
            self.coverage_sampler = lambda : poisson.rvs(mu = 8)
        else: 
            self.coverage_sampler = coverage_sampler 
        
        self.forbid_root_mut = forbid_root_mut
        
        if cell_tree is None: 
            self.random_tree(n_cells, n_mut)
            self.random_mut_locations()
        else: 
            self.tree = cell_tree
        
        if gt1 is None or gt2 is None: 
            self.random_mutations()
        else: 
            self.gt1 = gt1
            self.gt2 = gt2
    
    
    @property
    def n_cells(self): 
        return self.tree.n_cells
    
    
    @property
    def n_mut(self): 
        return self.tree.n_mut
    
    
    def use_tree(self, cell_tree): 
        self.tree = cell_tree
    
    
    def random_mutations(self): 
        mutations = np.random.choice(DataGenerator.mutation_types, size = self.n_mut, replace = True, p = self.priors)
        self.gt1 = np.array([mut[0] for mut in mutations], dtype = str)
        self.gt2 = np.array([mut[1] for mut in mutations], dtype = str)
    
    
    def random_tree(self, n_cells, n_mut): 
        self.tree = CellTree(n_cells, n_mut)
    
    
    def random_mut_locations(self): 
        self.tree.clear_mutations()
        candidates = np.arange(self.tree.n_nodes)
        if self.forbid_root_mut: 
            candidates = np.delete(candidates, self.tree.root.ID)
        mut_locations = np.random.choice(candidates, size = self.n_mut, replace = True)
        for i in range(self.n_mut): 
            self.tree.mutations[mut_locations[i]].append(i)
    
    
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
    
    
    def generate_reads(self): 
        genotypes = np.empty((self.n_cells, self.n_mut), dtype = str)
        for i in range(self.n_cells): # loop through each cell (leaf)
            mutated = np.zeros(self.n_mut, dtype = bool)
            mutated[self.tree.mutations[i]] = True
            for ancestor in self.tree.nodes[i].ancestors: 
                mutated[self.tree.mutations[ancestor.ID]] = True
            genotypes[i,:] = np.where(mutated, self.gt2, self.gt1)
        
        ref = np.empty((self.n_cells, self.n_mut))
        alt = np.empty((self.n_cells, self.n_mut))
        for i in tqdm(range(self.n_cells)): 
            for j in range(self.n_mut): 
                coverage = self.coverage_sampler()
                ref[i,j], alt[i,j] = self.generate_single_read(genotypes[i,j], coverage)
        
        return ref, alt
        

        


'''
def coverage_sampler(): 
    return int(np.random.choice(coverages))
dg = DataGenerator(coverage_sampler = coverage_sampler)
ref_test, alt_test = dg.generate_reads(mutations = ['AH' for i in range(3)])
'''