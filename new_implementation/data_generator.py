import os

import numpy as np
from scipy.stats import poisson, geom, betabinom

from .cell_tree import CellTree




class DataGenerator: 
    mutation_types = ['RH', 'AH', 'HR', 'HA']
    
    
    def __init__(self, f=0.95, omega=100, omega_h=50, mut_prop=1., genotype_freq=None, tree=None, gt1=None, gt2=None, coverage_sampler=None):
        self.alpha = f * omega
        self.beta = omega - self.alpha
        self.omega_h = omega_h
        self.mut_prop = mut_prop
        self.genotype_freq = [1/3, 1/3, 1/3] if genotype_freq is None else genotype_freq # [R, H, A]
        self.tree = tree
        
        if gt1 is not None and gt2 is not None:
            self.gt1 = gt1
            self.gt2 = gt2


    @property
    def n_cells(self):
        return self.tree.n_cells
    

    @property
    def n_loci(self):
        return self.tree.n_mut
    

    def mut_indicator(self):
        ''' Return a 2D Boolean array in which [i,j] indicates whether cell i is affected by mutation j '''
        result = np.zeros((self.n_cells, self.n_loci), dtype=bool)
        for j in range(self.n_loci): # determine for each mutation the cells below it in the tree
            # TODO: ignore fake mutations?
            for vtx in self.tree.dfs(self.tree.mut_loc[j]):
                if self.tree.isleaf(vtx):
                    result[vtx, j] = True
        
        return result

    
    def random_mut_types(self):
        self.gt1 = np.random.choice(['R', 'H', 'A'], size = self.n_loci, replace = True, p = self.genotype_freq)
        self.gt2 = np.empty_like(self.gt1)
        mutated = np.random.choice(self.n_loci, size = round(self.n_loci * self.mut_prop), replace = False)
        for j in mutated:
            if self.gt1[j] == 'H':
                self.gt2[j] = np.random.choice(['R', 'A']) # mutation HA and HR with equal probability
            else:
                self.gt2[j] = 'H'
    
    
    def random_tree(self, n_cells, n_loci): 
        self.tree = CellTree(n_cells, n_loci)
        self.tree.rand_structure()
        self.tree.rand_mut_loc()
    

    def generate_coverages(self, method='geometric', mean=8):
        match method:
            case 'poisson':
                coverage = poisson.rvs(mu=mean, size=(self.n_cells, self.n_loci))
            case 'geometric':
                coverage = geom.rvs(p=1/(mean+1), size=(self.n_cells, self.n_loci)) - 1
            case 'constant':
                coverage = np.ones((self.n_cells, self.n_loci), dtype=int) * mean
            case _:
                raise ValueError('Invalid coverage sampling method.')
        
        return coverage
    
    
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
            raise ValueError('[generate_single_read] ERROR: invalid genotype.')
        
        return n_ref, n_alt
    
    
    def generate_reads(self):
        coverage = self.generate_coverages()

        # determine genotypes
        genotype = np.empty((self.n_cells, self.n_loci), dtype=str)
        mut_indicator = self.mut_indicator()
        for i in range(self.n_cells): # loop through each cell (leaf)
            for j in range(self.n_loci):
                genotype[i,j] = self.gt2[j] if mut_indicator[i,j] else self.gt1[j]
        
        # actual reads
        ref = np.empty((self.n_cells, self.n_loci), dtype=int)
        alt = np.empty((self.n_cells, self.n_loci), dtype=int)
        for i in range(self.n_cells):
            for j in range(self.n_loci):
                ref[i,j], alt[i,j] = self.generate_single_read(genotype[i,j], coverage[i,j])
        
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
            self.random_mut_types(mut_prop=mut_prop, genotype_freq=[1/3, 1/3, 1/3])
            ref, alt = self.generate_reads()
            
            np.savetxt(os.path.join(path, 'ref_%i.txt' % i), ref, fmt='%i')
            np.savetxt(os.path.join(path, 'alt_%i.txt' % i), alt, fmt='%i')
            np.savetxt(os.path.join(path, 'parent_vec_%i.txt' % i), self.tree.parent_vec, fmt='%i')
            np.savetxt(os.path.join(path, 'mut_indicator_%i.txt' % i), self.mut_indicator(), fmt='%i')
