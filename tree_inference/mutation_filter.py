import numpy as np
import multiprocessing as mp
from scipy.stats import betabinom

from .utilities import *




class MutationFilter:
    def __init__(self, f=0.95, omega=100, h_factor=0.5, genotype_freq={'R': 1/4, 'H': 1/2, 'A': 1/4}, mut_freq=0.5, min_grp_size=1):
        self.set_betabinom(f, omega, h_factor)
        self.set_mut_type_prior(genotype_freq, mut_freq)
        self.min_grp_size = min_grp_size


    def set_betabinom(self, f, omega, h_factor):
        '''
        [Arguments]
            f: frequency of correct read (i.e. 1 - error rate)
            omega: uncertainty of f, effective number of prior observations (when determining error rate)
        '''
        self.alpha_R = f * omega
        self.beta_R = omega - self.alpha_R
        self.alpha_A = (1 - f) * omega
        self.beta_A = omega - self.alpha_A
        self.alpha_H = omega/2 * h_factor
        self.beta_H = omega/2 * h_factor
    

    def set_mut_type_prior(self, genotype_freq, mut_freq):
        '''
        Calculates and stores the log-prior for each possible mutation type of a locus (including non-mutated)

        [Arguments]
            genotype_freq: priors of the root (wildtype) having genotype R, H or A
            mut_freq: a priori proportion of loci that are mutated
        '''
        self.mut_type_prior = {s: None for s in ['R', 'H', 'A', 'RH', 'HR', 'AH', 'HA']}

        # three non-mutated cases
        self.mut_type_prior['R'] = genotype_freq['R'] * (1 - mut_freq)
        self.mut_type_prior['H'] = genotype_freq['H'] * (1 - mut_freq)
        self.mut_type_prior['A'] = genotype_freq['A'] * (1 - mut_freq)
        # four mutated cases
        self.mut_type_prior['RH'] = genotype_freq['R'] * mut_freq 
        self.mut_type_prior['HA'] = genotype_freq['H'] * mut_freq / 2
        self.mut_type_prior['HR'] = self.mut_type_prior['HA']
        self.mut_type_prior['AH'] = genotype_freq['A'] * mut_freq
        
        # convert to log scale
        for s in self.mut_type_prior: 
            self.mut_type_prior[s] = np.log(self.mut_type_prior[s])
        

    def single_read_llh(self, n_ref, n_alt, genotype):
        '''
        [Arguments]
            n_ref: number of ref reads
            n_alt: number of alt reads
            genotype: the genotype of interest

        [Returns]
            the log-likelihood of observing n_ref, n_alt, given genotype
        '''
        if genotype == 'R':
            result = betabinom.pmf(n_ref, n_ref + n_alt, self.alpha_R, self.beta_R)
        elif genotype == 'A':
            result = betabinom.pmf(n_ref, n_ref + n_alt, self.alpha_A, self.beta_A)
        elif genotype == 'H':
            result = betabinom.pmf(n_ref, n_ref + n_alt, self.alpha_H, self.beta_H)
        else:
            raise ValueError('[MutationFilter.single_read_llh] Invalid genotype.')
        
        return np.log(result)


    def k_mut_llh(self, ref, alt, gt1, gt2): 
        '''
        [Arguments]
            ref, alt: 1D array, read counts at a locus for all cells
            gt1, gt2: genotypes before and after the mutation
        
        [Returns]
            If gt1 is the same as gt2 (i.e. there is no mutation), returns a single joint log-likelihood
            Otherwise, returns a 1D array in which entry [k] is the log-likelihood of having k mutated cells
        '''
        
        N = ref.size # number of cells
        #assert(alt.size == N)
        
        if gt1 == gt2:
            return np.sum([self.single_read_llh(ref[i], alt[i], gt1) for i in range(N)])
        
        k_in_first_n_llh = np.empty((N+1, N+1)) # [n,k]: log-likelihood that k among the first n cells are mutated 
        k_in_first_n_llh[0,0] = 0 # Trivial case: when there is 0 cell in total, the likelihood of having 0 mutated cell is 1
        
        for n in range(N):
            # log-likelihoods of the n-th cell having gt1 and gt2
            gt1_llh = self.single_read_llh(ref[n], alt[n], gt1)
            gt2_llh = self.single_read_llh(ref[n], alt[n], gt2)

            # k = 0 special case
            k_in_first_n_llh[n+1, 0] = k_in_first_n_llh[n, 0] + gt1_llh

            # k = 1 through n
            k_over_n = np.array([k/(n+1) for k in range(1,n+1)])
            log_summand_1 = np.log(1 - k_over_n) + gt1_llh + k_in_first_n_llh[n, 1:n+1]
            log_summand_2 = np.log(k_over_n) + gt2_llh + k_in_first_n_llh[n, 0:n]
            k_in_first_n_llh[n+1, 1:n+1] = np.logaddexp(log_summand_1, log_summand_2)

            # k = n+1 special case
            k_in_first_n_llh[n+1, n+1] = k_in_first_n_llh[n, n] + gt2_llh
        
        return k_in_first_n_llh[N, :]


    def single_locus_posteriors(self, ref, alt, comp_priors): 
        '''
        Calculates the log-posterior of different mutation types for a single locus

        [Arguments]
            ref, alt: 1D arrays containing ref and alt reads of each cell
            comp_priors: log-prior for each genotype composition
        
        [Returns]
            1D numpy array containing posteriors of each mutation type, in the order ['R', 'H', 'A', 'RH', 'HA', 'HR', 'AH']
        
        NB When a mutation affects a single cell or all cells, it is considered non-mutated and assigned to one
        of 'R', 'H' and 'A', depending on which one is the majority
        '''
        llh_RH = self.k_mut_llh(ref, alt, 'R', 'H')
        llh_HA = self.k_mut_llh(ref, alt, 'H', 'A')
        assert(llh_RH[-1] == llh_HA[0])

        joint_R = llh_RH[:1] + comp_priors['R']
        joint_H = llh_HA[:1] + comp_priors['H']
        joint_A = llh_HA[-1:] + comp_priors['A']
        joint_RH = llh_RH[1:] + comp_priors['RH']
        joint_HA = llh_HA[1:] + comp_priors['HA']
        joint_HR = llh_RH[:-1] + comp_priors['HR']
        joint_AH = llh_RH[:-1] + comp_priors['AH']

        joint = np.array([
            logsumexp(np.concatenate((joint_R, joint_RH[:self.min_grp_size], joint_HR[:self.min_grp_size]))),
            logsumexp(np.concatenate((joint_H, joint_RH[-self.min_grp_size:], joint_HR[-self.min_grp_size:], joint_HA[:self.min_grp_size], joint_AH[:self.min_grp_size]))),
            logsumexp(np.concatenate((joint_R, joint_HA[-self.min_grp_size:], joint_AH[-self.min_grp_size:]))),
            logsumexp(joint_RH[self.min_grp_size:-self.min_grp_size]),
            logsumexp(joint_HA[self.min_grp_size:-self.min_grp_size]),
            logsumexp(joint_HR[self.min_grp_size:-self.min_grp_size]),
            logsumexp(joint_AH[self.min_grp_size:-self.min_grp_size])
        ])

        posteriors = lognormalize(joint) # Bayes' theorem
        return posteriors


    def mut_type_posteriors(self, ref, alt, n_threads=4):
        '''
        Calculates the log-prior of different mutation types for all loci
        In case no mutation occurs, all cells have the same genotype (which is either R or H or A)
        In case there is a mutation, each number of mutated cells is considered separately

        [Arguments]
            ref, alt: matrices containing the ref and alt reads
            n_threads: number of threads to use (since different loci can be treated in parallel)
        
        [Returns]
            2D numpy array with n_loci rows and 7 columns, with each column standing for a mutation type
        '''
        n_cells, n_loci = ref.shape
        
        # log-prior for each number of affected cells
        k_mut_priors = np.array([2 * logbinom(n_cells, k) - np.log(2*k-1) - logbinom(2*n_cells, 2*k) for k in range(1, n_cells+1)])

        # composition priors
        comp_priors = {}
        for mut_type in ['R', 'H', 'A']:
            comp_priors[mut_type] = self.mut_type_prior[mut_type]
        for mut_type in ['RH', 'HA', 'HR', 'AH']:
            comp_priors[mut_type] = self.mut_type_prior[mut_type] + k_mut_priors
        for mut_type in ['HR', 'AH']: # The arrays 'HR' and 'AH' are flipped for computational convenience later
            comp_priors[mut_type] = np.flip(comp_priors[mut_type])

        # calculate porsteiors for all loci with the help of multiprocessing
        with mp.Pool(processes=n_threads) as pool:
            result = [pool.apply_async(self.single_locus_posteriors, (ref[:,j], alt[:,j], comp_priors)) for j in range(n_loci)]
            result = np.stack([r.get() for r in result]) # moving this out of with statement results in infinite loop (why?)

        return result


    def filter_mutations(self, ref, alt, method='highest_post', t=None, n_exp=None): 
        '''
        Filters the loci according to the posteriors of each mutation state

        [Arguments]
            method: criterion that determines which loci are considered mutated
            t: the posterior threshold to be used when using the 'threshold' method
            n_exp: the number of loci to be selected when using the 'first_k' method
        '''
        assert(ref.shape == alt.shape)
        posteriors = np.exp(self.mut_type_posteriors(ref, alt))
        # merge mutations of opposite directions
        #posteriors[:,3] += posteriors[:,5]
        #posteriors[:,4] += posteriors[:,6]
        #posteriors = posteriors[:,:5]
        
        if method == 'highest_post': # for each locus, choose the state with highest posterior
            selected = np.where(np.argmax(posteriors, axis=1) >= 3)[0]
        elif method == 'threshold': # choose loci at which mutated posterior > threshold 
            selected = np.where(np.sum(posteriors[:,3:], axis=1) > t)[0]
        elif method == 'first_k': # choose loci with the k highest mutated posteriors
            mut_posteriors = np.sum(posteriors[:,3:], axis=1)
            order = np.argsort(mut_posteriors)
            selected = order[-n_exp:]
        else:
            raise ValueError('[MutationFilter.filter_mutations] Unknown filtering method.')
        
        mut_type = np.argmax(posteriors[selected, 3:], axis=1)
        gt1_inferred = np.choose(mut_type, choices=['R', 'H', 'H', 'A'])
        gt2_inferred = np.choose(mut_type, choices=['H', 'A', 'R', 'H'])

        return selected, gt1_inferred, gt2_inferred
    

    def get_llh_mat(self, ref, alt, gt1, gt2):
        '''
        [Returns]
            llh_mat_1: 2D array in which [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh_mat_2: 2D array in which [i,j] is the log-likelihood of cell i having gt2 at locus j
        '''
        n_cells, n_mut = ref.shape
        llh_mat_1 = np.empty((n_cells, n_mut))
        llh_mat_2 = np.empty((n_cells, n_mut))
        
        for i in range(n_cells):
            for j in range(n_mut):
                llh_mat_1[i,j] = self.single_read_llh(ref[i,j], alt[i,j], gt1[j])
                llh_mat_2[i,j] = self.single_read_llh(ref[i,j], alt[i,j], gt2[j])
        
        return llh_mat_1, llh_mat_2