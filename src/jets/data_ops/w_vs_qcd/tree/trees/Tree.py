

def dfs(v, node_ids, tree_content):
    #marked[v] = True
    left, right = node_ids[v]
    if left != -1:
        assert right != -1
        tree = Tree(tree_content[v])
        tree.add_child(dfs(left,node_ids, tree_content))
        tree.add_child(dfs(right,node_ids, tree_content))
        return tree
    leaf = Tree(tree_content[v])
    return leaf

class Tree:
    def __init__(self, data):

        self.parent = None
        self.num_children = 0
        self.children = list()
        self.data = data

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
