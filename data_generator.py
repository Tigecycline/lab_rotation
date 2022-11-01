from tree import *
from scipy.stats import poisson, betabinom




class DataGenerator: 
    mutation_types = ['RH', 'AH', 'HR', 'HA']
    
    
    def __init__(self, n_cells = 8, n_loci = 16, f = 0.95, omega = 100, omega_h = 50, cell_tree = None, mut_prop = 1., gt1 = None, gt2 = None, coverage_sampler = None):
        self.n_cells = n_cells
        self.n_loci = n_loci
        self.alpha = f * omega
        self.beta = omega - self.alpha
        self.omega_h = omega_h
        self.mut_prop = mut_prop
        self.tree = cell_tree
        
        if gt1 is None and gt2 is None: 
            self.random_mutations()
        else: 
            self.gt1 = gt1
            self.gt2 = gt2
        
        if coverage_sampler is None: 
            self.coverage_sampler = lambda : poisson.rvs(mu = 8)
        else: 
            self.coverage_sampler = coverage_sampler 
    
    
    def random_mutations(self): 
        self.gt1 = np.random.choice(['R', 'A', 'H'], size = self.n_loci, replace = True)
        self.gt2 = self.gt1.copy()
        mutated_loci = np.random.choice(self.n_loci, size = int(self.n_loci * self.mut_prop), replace = False)
        for j in mutated_loci: 
            if self.gt1[j] == 'H': 
                self.gt2[j] = np.random.choice(['R', 'A']) # mutation HA and HR with equal probability
            else: 
                self.gt2[j] = 'H'
    
    
    def random_tree(self): 
        self.tree = CellTree(self.n_cells, self.n_loci)
        self.tree.randomize()
        self.random_mut_locations()
    
    
    def random_mut_locations(self): 
        self.tree.attachments = np.random.randint(self.tree.n_nodes, size = self.n_loci)
    
    
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
        genotypes = np.empty((self.n_cells, self.n_loci), dtype = str)
        mutations = self.tree.mutations
        for i in range(self.n_cells): # loop through each cell (leaf)
            mutated = np.zeros(self.n_loci, dtype = bool)
            mutated[mutations[i]] = True
            for ancestor in self.tree.nodes[i].ancestors: 
                mutated[mutations[ancestor.ID]] = True
            genotypes[i,:] = np.where(mutated, self.gt2, self.gt1)
        
        ref = np.empty((self.n_cells, self.n_loci))
        alt = np.empty((self.n_cells, self.n_loci))
        for i in range(self.n_cells): 
            for j in range(self.n_loci): 
                coverage = self.coverage_sampler()
                ref[i,j], alt[i,j] = self.generate_single_read(genotypes[i,j], coverage)
        
        return ref, alt
        
