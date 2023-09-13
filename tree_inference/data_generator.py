from .tree import *
from scipy.stats import poisson, geom, betabinom
import os




class DataGenerator: 
    mutation_types = ['RH', 'AH', 'HR', 'HA']
    
    
    def __init__(self, n_cells=3, n_loci=10, f=0.95, omega=100, omega_h=50, cell_tree=None, gt1=None, gt2=None, coverage_sampler=None):
        self.n_cells = n_cells
        self.n_loci = n_loci
        self.alpha = f * omega
        self.beta = omega - self.alpha
        self.omega_h = omega_h
        self.tree = cell_tree
        
        if gt1 is not None and gt2 is not None: 
            self.gt1 = gt1
            self.gt2 = gt2
        
        #if coverage_sampler is None: 
        #    self.coverage_sampler = lambda : poisson.rvs(mu=8)
        #else: 
        #    self.coverage_sampler = coverage_sampler
    

    @property
    def mut_indicator(self):
        result = np.zeros((self.n_cells, self.n_loci), dtype=int)
        for j in range(self.n_loci):
            if self.gt1[j] == self.gt2[j]:
                continue # not mutated
            for node in self.tree.nodes[self.tree.attachments[j]].leaves:
                result[node.ID, j] = True
        
        return result

    
    def random_mutations(self, mutated=None, mut_prop=1., genotype_freq=None):
        if genotype_freq is None:
            genotype_freq = [1/3, 1/3, 1/3] # R, H, A
        self.gt1 = np.random.choice(['R', 'H', 'A'], size = self.n_loci, replace = True, p = genotype_freq)
        self.gt2 = self.gt1.copy()
        if mutated is None:
            mutated = np.random.choice(self.n_loci, size = round(self.n_loci * mut_prop), replace = False)
        for j in mutated:
            if self.gt1[j] == 'H':
                self.gt2[j] = np.random.choice(['R', 'A']) # mutation HA and HR with equal probability
            else:
                self.gt2[j] = 'H'
    
    
    def random_tree(self, one_cell_mut=True): 
        self.tree = CellTree(self.n_cells)
        self.tree.randomize()
        self.random_mut_locations(one_cell_mut)
    
    
    def random_mut_locations(self, one_cell_mut=True):
        low = 0 if one_cell_mut else self.n_cells
        self.tree.attachments = np.random.randint(low, self.tree.n_nodes, size = self.n_loci)
    

    def generate_coverages(self, method='geometric', mean=8):
        match method:
            case 'poisson':
                self.coverages = poisson.rvs(mu=mean, size=(self.n_cells, self.n_loci))
            case 'geometric':
                self.coverages = geom.rvs(p=1/(mean+1), size=(self.n_cells, self.n_loci)) - 1
            case 'constant':
                self.coverages = np.ones((self.n_cells, self.n_loci)) * mean
            case _:
                raise ValueError('[generate_coverages] invalid coverage sampling method.')
    
    
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
        self.generate_coverages()

        # determine genotypes
        genotypes = np.empty((self.n_cells, self.n_loci), dtype=str)
        mutations = self.tree.mutations
        for i in range(self.n_cells): # loop through each cell (leaf)
            mutated = np.zeros(self.n_loci, dtype = bool)
            mutated[mutations[i]] = True
            for ancestor in self.tree.nodes[i].ancestors:
                mutated[mutations[ancestor.ID]] = True
            genotypes[i,:] = np.where(mutated, self.gt2, self.gt1)
        
        # actual reads
        ref = np.empty((self.n_cells, self.n_loci), dtype=int)
        alt = np.empty((self.n_cells, self.n_loci), dtype=int)
        for i in range(self.n_cells):
            for j in range(self.n_loci):
                ref[i,j], alt[i,j] = self.generate_single_read(genotypes[i,j], self.coverages[i,j])
        
        return ref, alt
        

    def generate_comparison_data(self, size=100, mut_prop=0.75, path='./comparison_data/', seed=None):
        if seed is not None:
            np.random.seed(seed)
        if os.path.exists(path):
            while True:
                ans = input('Target path already exists. This can overwrite existing files. Do you want to continue? [Y/N] ')
                match ans:
                    case 'Y' | 'y' | 'Yes' | 'yes':
                        break
                    case 'N' | 'n' | 'No' | 'no':
                        return
        else:
            os.makedirs(path)
        
        for i in range(size):
            self.random_tree()
            self.random_mutations(mut_prop=mut_prop, genotype_freq=[1/3, 1/3, 1/3])
            ref, alt = self.generate_reads()
            
            np.savetxt(os.path.join(path, 'ref_%i.txt' % i), ref, fmt='%i')
            np.savetxt(os.path.join(path, 'alt_%i.txt' % i), alt, fmt='%i')
            np.savetxt(os.path.join(path, 'parent_vec_%i.txt' % i), self.tree.parent_vec, fmt='%i')
            np.savetxt(os.path.join(path, 'mut_indicator_%i.txt' % i), self.mut_indicator, fmt='%i')
