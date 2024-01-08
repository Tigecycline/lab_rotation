import numpy as np


from .forest import Forest




class MutationTee(Forest):
    def __init__(self, n_mut=3, n_cells=0):
        super().__init__(n_mut)
        self.n_mut = n_mut
        self.n_cells = n_cells

    
    @property
    def n_cells(self):
        return len(self.cell_loc)
    

    @n_cells.setter
    def n_cells(self, n_cells):
        self.cell_loc = np.ones(n_cells) * -1