from bisect import insort




class TreeNode: 
    def __init__(self, ID, parent = None, children = None): 
        self.ID = ID # index of cell / mutation etc. that this node represents
        
        self.parent = None
        if parent is not None: 
            self.assign_parent(parent)
        
        self.children = []
        if children is not None: 
            self.add_children(children)
    
    
    @property
    def isleaf(self): 
        return self.children == []
    
    
    @property
    def isroot(self): 
        return self.parent is None
    
    
    @property
    def grand_parent(self): 
        return self.parent.parent
    
    
    @property
    def ancestors(self): 
        ''' traverse all ancestors from most to least recent '''
        if self.parent is not None: 
            yield self.parent
            yield from self.parent.ancestors
    
    
    @property
    def siblings(self): 
        ''' traverse all siblings (excluding self) '''
        for node in self.parent.children: 
            if node is not self: 
                yield node
    
    
    @property
    def DFS(self): 
        ''' traverse subtree in DFS order '''
        yield self
        for child in self.children: 
            yield from child.DFS
    
    
    @property
    def reverse_DFS(self): 
        ''' traverse subtree in reversed DFS order '''
        for child in self.children: 
            yield from child.reverse_DFS
        yield self
    
    
    def __str__(self): 
        result = ''
        result += '****** node "' + str(self.ID) + '" ******\n'
        if self.isroot: 
            result += 'has no parent\n' 
        else: 
            result += 'parent: ' + str(self.parent.ID) + '\n'
        if self.isleaf: 
            result += 'has no children\n'
        else: 
            result += 'children: ' + str([child.ID for child in self.children]) + '\n'
        return result
        
    
    def __lt__(self, other): # make the class sortable, so that bisect.insort can be used
        return self.ID < other.ID
    
    
    #def isomorphic_to(self, other): 
    #    ''' 
    #    Must be used after self.sort, tests isomorphism of the two subtrees
    #    '''
    #    if self.mutations != other.mutations: 
    #        return False
    #    if len(self.children) != len(other.children): 
    #        return False
    #    for i in range(len(self.children)): 
    #        if not self.children[i].isomorphic_to(other.children[i]): 
    #            return False
    #    return True
    
    
    def sort(self): 
        ''' Sorts the children (according to ID) of self and all descendants '''
        self.children.sort() 
        for child in self.children: 
            child.sort() 
    
    
    def descends_from(self, other): 
        ''' a node is also considered descendant of itself '''
        # alternative implementation:
        # return self is other or self.parent is not None and self.parent.descends_from(other)
        if self is other or self in other.ancestors: 
            return True
        else: 
            return False
    
    
    def assign_parent(self, new_parent): 
        old_parent = self.parent
        self.parent = new_parent
        insort(new_parent.children, self) # keep children sorted (by ID)
        if old_parent is not None: 
            old_parent.remove_child(self)
    
    
    def add_child(self, new_child): 
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
    
    
    def copy(self, name = None, parent = None, children = None): 
        new_node = TreeNode(name, parent, children)
        new_node.mutations = self.mutations
        return new_node

