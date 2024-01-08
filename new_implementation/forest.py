import numpy as np




class Forest:
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
    

    ######## Forest properties ########
    @property
    def roots(self):
        return self._clist[-1]


    @property
    def n_vtx(self):
        return len(self._pvec)
    

    @property
    def parent_vec(self):
        return self._pvec.copy()
    

    @parent_vec.setter
    def parent_vec(self, new_parent_vec):
        if len(new_parent_vec) != self.n_vtx:
            raise ValueError('Parent vector must have the same length as number of vertices.')
        
        self._pvec = new_parent_vec.copy()
        for vtx in range(self.n_vtx+1):
            self._clist[vtx].clear()
        for vtx in range(self.n_vtx):
            self._clist[self.parent(vtx)].append(vtx)
    

    def copy_structure(self, other):
        self.parent_vec = other.parent_vec
    

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
    

    ######## Methods to traverse the tree ########
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


    def transfer_children(self, old_parent, new_parent):
        ''' Transfer all children of old_parent to new_parent. If old_parent == new_parent, nothing happens. '''
        transferred_children = self._clist[old_parent]
        self._clist[old_parent] = []
        self._clist[new_parent] += transferred_children
        for child in transferred_children:
            self._pvec[child] = new_parent
    

    #def guarded_transfer_children(self, old_parent, new_parent):
    #    ''' transfer_children guarded against loop formation '''
    #    if self.isdescendant(new_parent, old_parent):
    #        raise ValueError(f'Cannot transfer the children of {old_parent} to {new_parent} because {new_parent} is already a descendant of {old_parent}.')
    #    self.transfer_children(old_parent, new_parent)


    def prune(self, subroot):
        ''' Prune the subtree whose root is subroot. If subroot is already a root, nothing happens. '''
        self.assign_parent(subroot, -1)
    

    #def cut_splice(self, vtx):
    #    ''' Extract a specific vertex while keeping its children in original location '''
    #    old_parent = self.parent(vtx)
    #    self.prune(vtx)
    #    self.transfer_children(vtx, old_parent)