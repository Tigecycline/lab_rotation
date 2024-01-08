class SubTree:
    '''
    [Idea]
        Tree structure that:
        - allows pruning and merging
        - can be traversed both top-down and bottom-up
        - contains a data vector
    
    [Notes]
        - Traversing large trees (>1000 vertices) may exceed the max recursion depth
    '''
    
    def __init__(self, label=None):
        '''
        Initialize the forest with all vertices being individual trees
        '''
        self.parent = None
        self.children = []
        self.label = label
        self.data = None
    

    @property
    def isleaf(self):
        ''' A Tree is a leaf if it has no children '''
        return bool(self.children)
    
    
    @property
    def isroot(self):
        ''' A Tree is a root if it has no parent '''
        return self.parent is None


    ######## Methods to traverse the tree ########    
    def ancestors(self):
        ''' Interator to traverse all ancestors from most to least recent '''
        if self.parent is not None: 
            yield self.parent
            yield from self.parent.ancestors
    
    
    def siblings(self):
        ''' Iterator to traverse all siblings (excluding self) '''
        for node in self.parent.children:
            if node is not self:
                yield node
    

    def dfs(self):
        ''' Iterator to traverse subtree in DFS order '''
        yield self
        for child in self.children:
            yield from child.dfs()
    
    
    def rev_dfs(self):
        ''' Iterator to traverse subtree in reversed DFS order '''
        for child in self.children:
            yield from child.rev_dfs()
        yield self
    
    
    def leaves(self):
        ''' Iterator to traverse all leaves in the subtree '''
        for node in self.dfs():
            if node.isleaf:
                yield node
    

    ######## Methods to manipulate tree structure ########
    def set_parent(self, new_parent):
        ''' Designate new_parent as the new parent of self. If new_parent is already the parent of vtx, nothing happens. '''
        self.parent.children.remove(self)
        self.parent = new_parent
        new_parent.children.append(self)
    

    def guarded_set_parent(self, new_parent):
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent = new_parent
        if new_parent is not None:
            if new_parent in self.ancestors():
                raise RuntimeError('Loop created during tree operation, debugging adviced.')
            new_parent.children.append(self)
    

    def transfer_children(self, new_parent):
        ''' Transfer all children of self to . If old_parent and new_parent are the same, nothing happens. '''
        for child in self.children:
            child.parent = new_parent
        new_parent.children.extend(self.children)
        self.children.clear()
    

    def guarded_transfer_children(self, new_parent):
        ''' Transfer all children of self to . If old_parent and new_parent are the same, nothing happens. '''
        for child in self.children:
            child.parent = new_parent
        new_parent.children.extend(self.children)
        self.children.clear()
