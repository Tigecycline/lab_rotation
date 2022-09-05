class TreeNode: 
    def __init__(self, name = None, parent = None, children = None, mutations = None): 
        if name is None: 
            self.name = id(self) # TBC: add a separate index/id member
        else: 
            self.name = name
        
        self.parent = None
        if parent is not None: 
            self.assign_parent(parent)
        
        self.children = []
        if children is not None: 
            self.add_children(children)
        
        if mutations is None: 
            self.mutations = []
        else: 
            self.mutations  = mutations
    
    
    @property
    def isleaf(self): 
        return self.children == []
    
    
    @property
    def isroot(self): 
        return self.parent is None
    
    
    @property
    def isempty(self): 
        return self.mutations == []
    
    
    @property
    def grand_parent(self): 
        return self.parent.parent
    
    
    @property
    def ancestors(self): 
        ''' generator to traverse all ancestors from most to least recent '''
        if self.parent is not None: 
            yield self.parent
            yield from self.parent.ancestors
    
    
    @property
    def siblings(self): 
        ''' generator to traverse all siblings '''
        for node in self.parent.children: 
            if node is not self: 
                yield node
    
    
    @property
    def leaves(self): 
        ''' generator to traverse all descendants that are leaves '''
        if node.isleaf:
            yield node
        else: 
            for child in node.children: 
                yield from child.leaves
    
    
    @property
    def DFS(self): 
        ''' generator to traverse subtree in DFS order '''
        yield self
        for child in self.children: 
            yield from child.DFS
    
    
    @property
    def reverse_DFS(self): 
        ''' generator to traverse subtree in reversed DFS order '''
        for child in self.children: 
            yield from child.reverse_DFS
        yield self
    
    
    def __str__(self): 
        result = ''
        result += '****** node "' + str(self.name) + '" ******\n'
        if self.isroot: 
            result += 'has no parent\n' 
        else: 
            result += 'parent: ' + str(self.parent.name) + '\n'
        if self.isleaf: 
            result += 'has no children\n'
        else: 
            result += 'children: ' + str([child.name for child in self.children]) + '\n'
        result += 'mutations: ' + str(self.mutations) + '\n'
        return result
        
    
    def __lt__(self, other): # makes the class sortable
        return self.mutations < other.mutations
    
    
    def isomorphic_to(self, other): 
        ''' 
        Must be used after self.sort, tests isomorphism of the two subtrees
        '''
        if self.mutations != other.mutations: 
            return False
        if len(self.children) != len(other.children): 
            return False
        for i in range(len(self.children)): 
            if not self.children[i].isomorphic_to(other.children[i]): 
                return False
        return True
    
    
    def sort(self): 
        ''' Sorts the children and mutations of self and all its descendants '''
        self.children.sort() 
        self.mutations.sort() 
        for child in self.children: 
            child.sort() 
    
    
    def descends_from(self, other): 
        ''' N.B. a node is considered descendant of itself '''
        # alternative implementation:
        # return self is other or self.parent is not None and self.parent.descends_from(other)
        try: 
            return self is other or self.parent.descends_from(other)
        except: 
            return False
    
    
    def assign_parent(self, new_parent): 
        old_parent = self.parent
        self.parent = new_parent
        if new_parent is not None: 
            new_parent.children.append(self)
        if old_parent is not None: 
            old_parent.remove_child(self)
    
    
    def add_child(self, new_child): 
        new_child.assign_parent(self)
    
    
    def add_children(self, new_children): 
        for new_child in new_children: 
            new_child.assign_parent(self)
    
    
    def remove_child(self, child): 
        for i in range(len(self.children)): 
            if self.children[i] is child: 
                del self.children[i] 
                break
        # TBC: warning when child is not in self.children
        #print('*****************************************************')
        #print('[TreeNode.remove_child] WARNING: Child doesn\'t exist.')
        #print('*****************************************************')
    
    
    #def df_apply(self, func): 
    #    ''' apply func to self and all descendants in a depth-first manner '''
    #    func(self)
    #    for child in self.children: 
    #        child.df_apply(func)
    
    
    def get_nonempty_descendants(self, container = None): 
        ''' 
        A "nonemtpy descendant (NED)" is a descendant node of self such that 
        (1) the NED contains at least one mutation and
        (2) no mutation is contained in nodes between self and the NED
        '''
        if container is None: 
            container = []
        
        for child in self.children: 
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

