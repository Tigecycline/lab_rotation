import numpy as np




class PruneTree:
    '''
    [Basic idea]
        - A tree structure that allows pruning subtrees (where it becomes a forest) and re-inserting them
        - Vertices are represented by integers, ranging from 0 to n_nodes - 1
        - A parent vector and a children list are kept and updated simultaneously
        - An additional sentinel vertex is created and serves as the parent of the root vertex
            This sentinel vertex is represented by index -1
    
    [Notes]
        - Traversing large trees (>1000 vertices) may exceed the max recursion depth
    '''
    
    def __init__(self, n_vertices):
        '''
        Initialize the forest with all vertices being individual trees
        '''
        self._pvec = np.ones(n_vertices, dtype=int) * -1 # parent vector
        self._clist = [[] for i in range(n_vertices)] + [list(range(n_vertices))]
        self._main_root = 0
    

    ######## Tree properties ########
    @property
    def roots(self):
        return self._clist[-1]
    

    @property
    def main_root(self):
        return self._main_root
    

    def reroot(self, new_main_root):
        if not self.isroot(new_main_root):
            self.prune(new_main_root)
        self._main_root = new_main_root


    @property
    def n_vtx(self):
        return len(self._pvec)
    

    @property
    def parent_vec(self):
        return self._pvec.copy()
    

    def use_parent_vec(self, new_parent_vec, main_root=None):
        if len(new_parent_vec) != self.n_vtx:
            raise ValueError('Parent vector must have the same length as number of vertices.')
        
        for vtx in range(self.n_vtx):
            self.assign_parent(vtx, new_parent_vec[vtx])
        
        if main_root is None:
            self._main_root = self.roots[0]
        elif self._pvec[main_root] != -1:
            raise ValueError('Provided main root is not a root.')
        else:
            self._main_root = main_root
    

    def copy_from(self, other):
        self.set_parent_vec(other.parent_vec, other.main_root)
    

    ######## Single-vertex properties ########
    def parent(self, vtx):
        #if vtx < 0:
        #    raise IndexError('Vertex index must be non-negative.')
        return self._pvec[vtx]


    def children(self, vtx):
        return self._clist[vtx]
    

    def isroot(self, vtx):
        return self._pvec[vtx] == -1
    

    def isleaf(self, vtx):
        return len(self._clist[vtx]) == 0


    def isdescendant(self, vtx, ancestor):
        return ancestor in self.ancestors(vtx)
    

    ######## Methods to traverse vertices ########
    def pruned_roots(self):
        ''' Traverse the roots of all pruned subtrees '''
        for rt in self.roots:
            if rt != self._main_root:
                yield rt


    def ancestors(self, vtx):
        ''' Traverse all ancestors of a vertex, NOT including itself '''
        if not self.isroot(vtx):
            yield self._pvec[vtx]
            yield from self.ancestors(self._pvec[vtx])
    

    def siblings(self, vtx):
        ''' Traverse all siblings of a vertex, NOT including itself '''
        for child in self._clist[self._pvec[vtx]]:
            if child != vtx:
                yield child


    def dfs(self, subroot):
        ''' Traverse the subtree rooted at subroot, in DFS order '''
        yield subroot
        for child in self._clist[subroot]:
            yield from self.dfs(child)
    

    def rdfs(self, subroot):
        ''' Traverse the subtree rooted at subroot, in reversed DFS order '''
        for child in self._clist[subroot]:
            yield from self.rdfs(child)
        yield subroot
    

    def leaves(self, subroot):
        for vtx in self.dfs(subroot):
            if self.isleaf(vtx):
                yield vtx

    
    ######## Methods to manipulate tree structure ########
    def assign_parent(self, vtx, new_parent):
        ''' Designate new_parent as the new parent of vtx. If new_parent is already the parent of vtx, nothing happens. '''
        self._clist[self._pvec[vtx]].remove(vtx)
        self._pvec[vtx] = new_parent
        self._clist[new_parent].append(vtx)
    

    #def guarded_assign_parent(self, vtx, new_parent):
    #    ''' assign_parent guarded against loop formation '''
    #    if vtx == new_parent:
    #        raise ValueError(f'Vertex {vtx} cannot become parent of itself.')
    #    if self.isdescendant(new_parent, vtx):
    #        raise ValueError(f'Vertex {new_parent} cannot be a parent of {vtx} because it is already a descendant of {vtx}.')
    #    self.assign_parent(vtx, new_parent)


    def prune(self, subroot):
        ''' Prune the subtree whose root is subroot. If subroot is already a root, nothing happens. '''
        self.assign_parent(subroot, -1)
    

    def splice(self, subroot):
        ''' Prune at subroot and its parent, then attach subroot back to its grandparent. '''
        if self.isroot(subroot):
            raise ValueError('Cannot splice a root')
        
        parent = self.parent(subroot)
        grandparent = self.parent(parent)

        self.prune(parent)
        self.assign_parent(subroot, grandparent)
        if grandparent == -1:
            self._main_root = subroot


    def insert(self, subroot, target):
        ''' Insert the root of a pruned subtree to the edge above target. '''
        self.assign_parent(subroot, self._pvec[target])
        self.assign_parent(target, subroot)
        if target == self._main_root:
            self._main_root = subroot