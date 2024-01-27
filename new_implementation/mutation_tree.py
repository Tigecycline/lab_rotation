import numpy as np
import graphviz
import warnings

from .prunetree_base import PruneTree




class MutationTree(PruneTree):
    def __init__(self, n_mut=2, n_cells=0):
        if n_mut < 2:
            warnings.warn('Mutation tree too small, nothing to explore.', RuntimeWarning)
        
        super().__init__(n_mut+1)
        self.n_mut = n_mut
        self.n_cells = n_cells

        self.reroot(self.wt)
        for vtx in range(self.n_mut):
            self.assign_parent(vtx, self.wt)

    
    @property
    def wt(self):
        return self.n_mut


    @property
    def n_cells(self):
        return len(self.cell_loc)


    @n_cells.setter
    def n_cells(self, n_cells):
        self.cell_loc = np.ones(n_cells, dtype=int) * -1


    def fit_llh(self, llh_1, llh_2):
        '''
        Gets ready to run optimization using provided log-likelihoods.
        The two genotypes involved in the mutation, gt1 and gt2, are not required for optimization.

        [Arguments]
            llh1: 2D array in which entry [i,j] is the log-likelihood of cell i having gt1 at locus j
            llh2: 2D array in which entry [i,j] is the log-likelihood of cell i having gt2 at locus j
        '''
        assert(llh_1.shape == llh_2.shape)

        if llh_1.shape[1] != self.n_mut:
            self.__init__(llh_1.shape[1], llh_1.shape[0])
        elif llh_1.shape[0] != self.n_cells:
            self.n_cells = llh_1.shape[0]

        self.llr = np.empty((self.n_cells, self.n_vtx))
        self.llr[:,:self.n_mut] = llh_2 - llh_1
        self.llr[:,self.n_mut] = 0 # stands for wildtype
        self.cumul_llr = np.empty_like(self.llr)

        # joint likelihood of each locus when all cells have genotype 1 or 2
        self.loc_joint_1 = llh_1.sum(axis=0)
        self.loc_joint_2 = llh_2.sum(axis=0)
        
        self.flipped = np.zeros(self.n_vtx, dtype=bool)

        self.update_all()
    

    def fit_cell_tree(self, ct):
        # TODO: consider randomizing the order of the mutations at the same edge
        # TODO: instead of using np.where, construct a list of lists beforehand
        assert(self.n_mut == ct.n_mut)
        assert(len(ct.roots) == 1) # TODO: consider enabling this when pruned subtrees exist

        mrm = np.empty(ct.n_vtx+1, dtype=int) # mrm for "most recent mutation"
        mrm[-1] = self.wt # put wildtype at sentinel
        
        for cvtx in ct.dfs(ct.main_root):  # cvtx for "cell vertex"
            mut_list = np.where(ct.mut_loc == cvtx)[0] # mutations attached to the edge above cvtx
            parent_mut = mrm[ct.parent(cvtx)]
            if mut_list.size > 0:
                self.assign_parent(mut_list[0], parent_mut)
                for mut1, mut2 in zip(mut_list, mut_list[1:]): 
                    self.assign_parent(mut2, mut1)
                mrm[cvtx] = mut_list[-1]
            else: 
                mrm[cvtx] = mrm[ct.parent(cvtx)]
        
        self.flipped[:-1] = ct.flipped

    
    def update_cumul_llr(self):
        for rt in self.roots:
            for vtx in self.dfs(rt):
                llr_summand =  - self.llr[:,vtx] if self.flipped[vtx] else self.llr[:,vtx]
                # It is not necessary to distinguish between roots and non-roots,
                # as self.cumul_llr[:,self.parent(vtx)] is 0 for all roots
                # This distinguishment serves only to maintain logical consistency
                if self.isroot(vtx):
                    self.cumul_llr[:,vtx] = llr_summand
                else:
                    self.cumul_llr[:,vtx] = self.cumul_llr[:,self.parent(vtx)] + llr_summand
    

    def update_cell_loc(self):
        self.cell_loc = self.cumul_llr.argmax(axis=1)
        wt_llh = np.where(self.flipped[:-1], self.loc_joint_2, self.loc_joint_1).sum()
        self.joint = self.cumul_llr.max(axis=1).sum() + wt_llh


    def update_all(self):
        self.update_cumul_llr()
        self.update_cell_loc()
    

    def greedy_attach(self):
        for subroot in self.pruned_roots():
            main_tree_max = self.cumul_llr[:,list(self.dfs(self.main_root))].max(axis=1)
            subtree_max = self.cumul_llr[:,list(self.dfs(subroot))].max(axis=1)

            best_llr = -np.inf
            for vtx in self.dfs(self.main_root):
                total_llr = np.maximum(main_tree_max, subtree_max + self.cumul_llr[:,vtx]).sum()
                if total_llr > best_llr:
                    best_llr = total_llr
                    best_loc = vtx
            self.assign_parent(subroot, best_loc)
            self.update_all()
    

    def exhaustive_optimize(self):
        for subroot in range(self.n_mut):
            self.prune(subroot)
            self.update_all()
            self.greedy_attach()


    def to_graphviz(self, filename=None, engine='dot'): 
        dgraph = graphviz.Digraph(filename=filename, engine=engine)
        
        dgraph.node(str(self.wt), label='wt', shape='rectangle', color='gray')
        for vtx in range(self.n_mut):
            dgraph.node(str(vtx), shape='rectangle', style='filled', color='gray')
            if self.isroot(vtx):
                # for root, create a corresponding void node
                dgraph.node(f'void_{vtx}', label='', shape='point')
                dgraph.edge(f'void_{vtx}', str(vtx))
            else:
                dgraph.node(str(vtx), shape='rectangle', style='filled', color='gray')
                dgraph.edge(str(self.parent(vtx)), str(vtx))
        
        for i in range(self.n_cells):
            name = 'c' + str(i)
            dgraph.node(name, shape = 'circle')
            # TBC: use undirected edge for cell attachment
            #dg.edge(str(self.attachments[i]), name, dir = 'none')
            dgraph.edge(str(self.cell_loc[i]), name)
        
        return dgraph