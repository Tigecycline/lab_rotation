class TreeNode: 
    def __init__(self, name = None, parent = None, children = None, mutations = None): 
        if name is None: 
            self.name = id(self) # TBC: add a separate index/id member
        else: 
            self.name = name
        
        self._parent = None
        if parent is not None: 
            self.assign_parent(parent)
        
        self._children = []
        if children is not None: 
            self.add_children(children)
        
        if mutations is None: 
            self.mutations = []
        else: 
            self.mutations  = mutations
    
    
    @property
    def isleaf(self): 
        return self._children == []
    
    
    @property
    def isroot(self): 
        return self._parent is None
    
    
    @property
    def isempty(self): 
        ''' emtpy means the node contains no mutation '''
        return self.mutations == []
    
    
    def __lt__(self, other): # makes the class sortable
        return self.mutations < other.mutations
    
    
    def __eq__(self, other): 
        ''' 
        Tests equality of two subtrees without considering node names
        If used after self.sort, tests isomorphism of the two subtrees
        '''
        return self.mutations == other.mutations and self._children == other._children
    
    
    def __str__(self): 
        result = ''
        result += '****** node ' + str(self.name) + ' ******\n'
        if not self.isroot: 
            result += 'parent: ' + str(self._parent.name) + '\n'
        else: 
            result += 'has no parent\n' 
        if not self.isleaf: 
            result += 'children: ' + str([child.name for child in self._children]) + '\n'
        else: 
            result += 'has no children\n'
        result += 'mutations: ' + str(self.mutations) + '\n'
        return result
    
    
    def sort(self): 
        ''' Sorts the children and mutations of self and all its descendants '''
        self._children.sort() 
        self.mutations.sort() 
        for child in self._children: 
            child.sort() 
    
    
    def assign_parent(self, new_parent): 
        old_parent = self._parent
        self._parent = new_parent
        new_parent._children.append(self)
        if old_parent is not None: 
            old_parent.remove_child(self)
    
    
    def add_child(self, new_child): 
        new_child.assign_parent(self)
    
    
    def add_children(self, new_children): 
        for new_child in new_children: 
            new_child.assign_parent(self)
    
    
    def remove_child(self, child): 
        for i in range(len(self._children)): 
            if self._children[i] is child: 
                del self._children[i] 
                break
        # TBC: warning when child is not in self._children
        #print('*****************************************************')
        #print('[TreeNode.remove_child] WARNING: Child doesn\'t exist.')
        #print('*****************************************************')
    
    
    def swap_parents(self, other): 
        self.assign_parent(other._parent)
        other.assign_parent(self._parent)
    
    
    #def df_apply(self, func): 
    #    ''' apply func to self and all descendants in a depth-first manner '''
    #    func(self)
    #    for child in self.children: 
    #        child.df_apply(func)
    
    
    def get_parent(self): 
        return self._parent
    
    
    def get_children(self): # TBC: hide children and implement a function to traverse tree
        return self._children
    
    
    def get_nonempty_descendants(self, container = None): 
        ''' 
        A "nonemtpy descendant (NED)" is a descendant node of self such that 
        (1) the NED contains at least one mutation and
        (2) no mutation is contained in nodes between self and the NED
        '''
        if container is None: 
            container = []
        
        for child in self._children: 
            if child.isempty: 
                child.get_nonempty_descendants(container)
            else: 
                container.append(child)
        
        return container
    
    
    def create_NED_tree(self, container = None, parent = None): 
        new_node = self.copy()
        
        if container is None: 
            container = [new_node]
        else: 
            container.append(new_node)
            new_node.assign_parent(parent)
        
        for ned in self.get_nonempty_descendants(): 
            ned.create_NED_tree(container, new_node)
        
        return container
    
    
    def add_mutation(self, mut): 
        self.mutations.append(mut)
    
    
    def clear_mutations(self): 
        self.mutations = []
    
    
    def copy(self, name = None, parent = None, children = None): 
        new_node = TreeNode(name, parent, children)
        new_node.mutations = self.mutations
        return new_node