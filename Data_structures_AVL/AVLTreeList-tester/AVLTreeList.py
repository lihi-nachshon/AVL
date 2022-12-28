# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info
import random

"""A class represnting a node in an AVL tree"""

class AVLNode(object):
    """Constructor, you are allowed to add more fields.
	@type value: str
	@param isReal: boolean optional
	@param value: data of your node
	"""

    def __init__(self, value):
        self.setLeft(VirtualNode())
        self.setRight(VirtualNode())
        self.height = 0
        self.size = 1
        self.parent = None
        self.value = value


    """returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child"""
    """Time complexity: O(1)"""
    def getLeft(self):
        return self.left

    """returns the right child
	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""
    """Time complexity: O(1)"""
    def getRight(self):
        return self.right

    """returns the parent 
	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""
    """Time complexity: O(1)"""
    def getParent(self):
        return self.parent

    """return the value
	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""
    """Time complexity: O(1)"""
    def getValue(self):
        return self.value

    """returns the height
	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""
    """Time complexity: O(1)"""
    def getHeight(self):
        return self.height

    """returns the size
    @rtype int
    @returns: the size of self, 0 if node is virtual
    """
    """Time complexity: O(1)"""
    def getSize(self):
        return self.size

    """sets left child
	@type node: AVLNode
	@param node: a node
	"""
    """Time complexity: O(1)"""
    def setLeft(self, node):
        node.setParent(self)
        self.left = node

    """sets right child
	@type node: AVLNode
	@param node: a node
	"""
    """Time complexity: O(1)"""
    def setRight(self, node):
        node.setParent(self)
        self.right = node

    """sets parent
	@type node: AVLNode
	@param node: a node
	"""
    """Time complexity: O(1)"""
    def setParent(self, node):
        self.parent = node

    """sets value
	@type value: str
	@param value: data
	"""
    """Time complexity: O(1)"""
    def setValue(self, value):
        self.value = value

    """sets the balance factor of the node
	@type h: int
	@param h: the height
	"""
    """Time complexity: O(1)"""
    def setHeight(self, h):
        self.height = h

    """sets the size of the node
    @type s: int
    @param s: the size
    """
    """Time complexity: O(1)"""
    def setSize(self, s):
        self.size = s

    """returns whether self is not a virtual node 
	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""
    """Time complexity: O(1)"""
    def isRealNode(self):
        return True

class VirtualNode(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0
        self.value = None

    """returns the left child
     @rtype: AVLNode
     @returns: the left child of self, None if there is no left child"""
    """Time complexity: O(1)"""

    def getLeft(self):
        return self.left

    """returns the right child
    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """
    """Time complexity: O(1)"""

    def getRight(self):
        return self.right

    """returns the parent 
    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """
    """Time complexity: O(1)"""

    def getParent(self):
        return self.parent

    """return the value
    @rtype: str
    @returns: the value of self, None if the node is virtual
    """
    """Time complexity: O(1)"""

    def getValue(self):
        return self.value

    """returns the height
    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """
    """Time complexity: O(1)"""

    def getHeight(self):
        return self.height

    """returns the size
    @rtype int
    @returns: the size of self, 0 if node is virtual
    """
    """Time complexity: O(1)"""

    def getSize(self):
        return self.size

    """sets parent
    @type node: AVLNode
    @param node: a node
    """
    """Time complexity: O(1)"""

    def setParent(self, node):
        self.parent = node


    """returns whether self is not a virtual node 
    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """
    """Time complexity: O(1)"""

    def isRealNode(self):
        return False

"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
	Constructor, you are allowed to add more fields.
	"""

    def __init__(self):
        self.size = 0
        self.root = None
        self.firstItem = None
        self.lastItem = None

    """returns whether the list is empty
	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""
    """Time complexity: O(1)"""
    def empty(self):
        return self.size == 0

    """retrieves the value of the i'th item in the list
	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the the value of the i'th item in the list
	"""
    """Time complexity: O(logn), due to Select's time complexity"""
    def retrieve(self, i):
        if 0 <= i < self.size:
            return self.Tree_select(i + 1).getValue()
        return None

    """inserts val at position i in the list
	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we inserts
	@rtype: list
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""
    """Time complexity: O(logn)"""
    def insert(self, i, val):
        n = self.size
        # checks if i is legal
        if 0 > i or i > n:
            return 0

        new_node = AVLNode(val)

        # if tree is empty
        if n == 0:
            return self.insert_first_node(new_node)
        # if insert first
        if i == 0:
            node_first = self.firstItem
            node_first.setLeft(new_node)
            self.firstItem = new_node
        # if insert last
        elif i == n:
            node_last = self.lastItem
            node_last.setRight(new_node)
            self.lastItem = new_node
        else:
            # finds the current node in position and inserts before it
            node_successor = self.Tree_select(i + 1)
            # if the current node in position has no left child
            if not node_successor.getLeft().isRealNode():
                node_successor.setLeft(new_node)
            else:
                node_predecessor = self.predecessor(node_successor)
                node_predecessor.setRight(new_node)

        self.size += 1
        cnt_rotations = self.rebalance_tree(new_node)

        return cnt_rotations

    """Inserts a node when tree is empty
        @type node: AVLNode
        @param node: node to insert
        @rtype: int
        @returns: 0, the number of rebalancing operation due to AVL rebalancing when inserting a first node
		Time complexity: O(1)"""
    def insert_first_node(self, node):
        self.root = node
        self.firstItem = node
        self.lastItem = node
        self.size = 1
        return 0

    """deletes the i'th item in the list
	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""
    """Time complexity: O(logn)"""
    def delete(self, i):
        n = self.size
        # checks if i is legal
        if 0 > i or i >= n:
            return -1

        # finds relevant node
        node = self.Tree_select(i + 1)

        # deletion when the tree has one node
        if n == 1:
            self.root = None
            self.lastItem = None
            self.firstItem = None
            self.size -= 1
            return 0

        # if deletes firstItem
        if i == 0:
            self.firstItem = self.successor(node)
        # if deletes lastItem
        if i == n - 1:
            self.lastItem = self.predecessor(node)

        # if node is a leaf
        if not node.getLeft().isRealNode() and not node.getRight().isRealNode():
            return self.delete_leaf(node)
        # if node has one right child
        elif not node.getLeft().isRealNode():
            return self.delete_node_with_right_child(node)
        # if node has one left child
        elif not node.getRight().isRealNode():
            return self.delete_node_with_left_child(node)
        # if node has two children
        else:
            return self.delete_node_with_two_children(node, i)

    """deletes leaf
        @type node: AVLNode
        @param node: the node to be deleted
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing 
    """
    """Time complexity: O(logn)"""
    def delete_leaf(self, node):
        node_parent = node.getParent()
        if node_parent.getLeft() == node:
            node_parent.setLeft(VirtualNode())
        else:
            node_parent.setRight(VirtualNode())
        node.setParent(None)
        self.size -= 1
        cnt_rotations = self.rebalance_tree(node_parent)

        return cnt_rotations

    """deletes node with only a right child
        @type node: AVLNode
        @param node: node to be deleted
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing
	"""
    """Time complexity: O(logn)"""
    def delete_node_with_right_child(self, node):
        isRoot = (node == self.root)
        child = node.getRight()
        # if deletes the root (possible only when the tree has two nodes)
        if isRoot:
            self.root = child
            self.lastItem = child
            self.firstItem = child
            self.size -= 1
            self.update_fields(child)
            self.root.setParent(None)
            return 0
        node_parent = node.getParent()
        if node_parent.getLeft() == node:
            node_parent.setLeft(child)
        else:
            node_parent.setRight(child)
        node.setParent(None)
        node.setRight(VirtualNode())

        self.size -= 1
        cnt_rotations = self.rebalance_tree(node_parent)

        return cnt_rotations

    """deletes node with only a left child
		@type node: AVLNode
        @param node: node to be deleted
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing
	"""
    """Time complexity: O(logn)"""
    def delete_node_with_left_child(self, node):
        isRoot = (node == self.root)
        child = node.getLeft()
        # if deletes the root (possible only when the tree has two nodes)
        if isRoot:
            self.root = child
            self.lastItem = child
            self.firstItem = child
            self.size -= 1
            self.update_fields(child)
            self.root.setParent(None)
            return 0
        node_parent = node.getParent()
        if node_parent.getLeft() == node:
            node_parent.setLeft(child)
        else:
            node_parent.setRight(child)
        node.setParent(None)
        node.setLeft(VirtualNode())

        self.size -= 1
        cnt_rotations = self.rebalance_tree(node_parent)

        return cnt_rotations

    """deletes node with two children
		@type node: AVLNode
        @param node: node to be deleted
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing
	"""
    """Time complexity: O(logn)"""
    def delete_node_with_two_children(self, node, index):
        node_successor = self.successor(node)

        # checks if the successor is the lastItem
        suc_last = False
        if node_successor == self.lastItem:
            suc_last = True

        # counts the rotations of the actual deletion of the successor
        cnt_rotations = self.delete(index + 1)

        if suc_last:
            self.lastItem = node_successor

        node_lchild = node.getLeft()
        node_rchild = node.getRight()
        node_parent = node.getParent()

        # if deletes the root
        if node_parent is None:
            self.root = node_successor

        else:
            if node_parent.getLeft() == node:
                node_parent.setLeft(node_successor)
            else:
                node_parent.setRight(node_successor)

        node_successor.setRight(node_rchild)
        node_successor.setLeft(node_lchild)
        self.update_fields(node_successor)

        return cnt_rotations

    """returns the value of the first item in the list
	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""
    """Time complexity: O(1)"""
    def first(self):
        if self.firstItem is None:
            return None
        return self.firstItem.getValue()

    """returns the value of the last item in the list
	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""
    """Time complexity: O(1)"""
    def last(self):
        if self.lastItem is None:
            return None
        return self.lastItem.getValue()

    """returns an array representing list 
	@rtype: list
	@returns: a list of strings representing the data structure
	"""
    """Time complexity: O(n)"""
    def listToArray(self):
        lst = []
        if self.size > 0:
            self.in_order(lst)
        return lst

    """adds the values to a list via in order walk
        @type lst: list
        @param lst: the list into which we add the item in order
		Time complexity: O(n)"""
    def in_order(self, lst):
        self.in_order_rec(lst, self.root)

    def in_order_rec(self, lst, node):
        if node.isRealNode():
            self.in_order_rec(lst, node.getLeft())
            lst.append(node.getValue())
            self.in_order_rec(lst, node.getRight())

    """returns the size of the list 
	@rtype: int
	@returns: the size of the list
	"""
    """Time complexity: O(1)"""
    def length(self):
        return self.size

    """sort the info values of the list
	@rtype: list
	@returns: an AVLTreeList where the values are sorted by the info of the original list.
	"""
    """Time complexity: O(nlogn), using mergesort"""
    def sort(self):
        # if tree is empty returns a new empty tree
        if self.size == 0:
            return AVLTreeList()
        arr = self.listToArray()
        sorted_arr = self.mergesort(arr)
        return self.create_tree_from_array(sorted_arr)

    """sorts a list
        @type lst: list
        @param lst: list to sort
        @rtype list
        @returns a sorted list
    """
    """Time complexity: O(nlogn)"""
    def mergesort(self, lst):
        n = len(lst)
        if n <= 1:
            return lst
        else:
            return self.merge(self.mergesort(lst[0:n // 2]), self.mergesort(lst[n // 2:]))

    """Time complexity: O(n)"""
    def merge(self, A, B):
        n = len(A)
        m = len(B)
        C = [0 for i in range(n + m)]
        a = b = c = 0
        while a < n and b < m:
            if A[a] < B[b]:
                C[c] = A[a]
                a += 1
            else:
                C[c] = B[b]
                b += 1
            c += 1
        if a == n:
           while b < m:
               C[c] = B[b]
               b += 1
               c += 1
        else:
            while a < n:
                C[c] = A[a]
                a += 1
                c += 1
        return C

    """permute the info values of the list 
	@rtype: list
	@returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
	"""
    """Time complexity: O(n)"""
    def permutation(self):
        # if tree is empty returns a new empty tree
        if self.size == 0:
            return AVLTreeList()
        arr = self.listToArray()
        # shuffles arr
        for i in range(len(arr) - 1, 0, -1):
            j = random.randint(0, i)
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp

        return self.create_tree_from_array(arr)

    """creates a new AVLTreeList from an array
        @type arr: lis
        @param arr: list of which to create an AVL tree
        @rtype AVLListTree
        @returns an AVL tree with the items in the list
		Time complexity: O(n)"""
    def create_tree_from_array(self, arr):
        # creates the structure
        root = self.create_tree_from_array_rec(arr, 0, len(arr)-1)
        # makes structure a tree
        tree = AVLTreeList()
        tree.root = root
        tree.size = len(arr)
        tree.firstItem = tree.get_min(tree.root)
        tree.lastItem = tree.get_max(tree.root)
        root.setParent(None)

        return tree

    """creates an AVLTreeList structure from an array
        recursive function
        @type arr: list
        @param arr: list to create an AVL from
        @rtype AVLNode
        @returns a root of an sub AVL tree
		Time complexity: O(n)"""
    def create_tree_from_array_rec(self, arr, i, j):
        if i > j:
            return VirtualNode()
        if i == j:
            return AVLNode(arr[i])
        mid = (j+i)//2
        node_parent = AVLNode(arr[mid])
        l_child = self.create_tree_from_array_rec(arr, i, mid-1)
        r_child = self.create_tree_from_array_rec(arr, mid+1, j)
        node_parent.setRight(r_child)
        node_parent.setLeft(l_child)
        self.update_fields(node_parent)

        return node_parent

    """concatenates lst to self
        @type lst: AVLTreeList
        @param lst: a list to be concatenated after self
        @rtype: int
        @returns: the absolute value of the difference between the height of the AVL trees joined
        n1 - number on nodes in self, n2 - number of nodes in lst
        Time complexity: O(log(n1+n2))"""
    def concat(self, lst):
        if self.size == 0:
            if lst.size == 0:
                return 0
            self.size = lst.size
            self.root = lst.root
            self.firstItem = lst.firstItem
            self.lastItem = lst.lastItem
            return lst.root.height + 1
        if lst.size == 0:
            return self.root.height + 1
        height_difference = abs(self.root.height - lst.root.height)
        if self.root.height > lst.root.height:
            self.right_concat(lst)
        elif self.root.height < lst.root.height:
            self.left_concat(lst)
        else:
            self.same_height_concat(lst)

        return height_difference

    """concats the lists when self's height is bigger
        @type lst: AVLTreeList
        @param lst: a list to be concatenated after self
        @rtype: int
        @returns: the absolute value of the difference between the height of the AVL trees joined
        n1 - number on nodes in self, n2 - number of nodes in lst
		Time complexity: O(log(n1+n2))"""
    def left_concat(self, lst):
        h1 = self.root.getHeight()
        r_node = lst.root
        while r_node.getHeight() > h1:
            r_node = r_node.getLeft()
        l_node = self.root
        # adds node x for the join function
        x = AVLNode(None)
        x_index = self.size
        self.join(l_node, x, r_node, "left")
        self.root = lst.root
        self.size += lst.size + 1
        self.lastItem = lst.lastItem
        # deletes node x added for join
        self.delete(x_index)

    """concats the lists when self's height is smaller
        @type lst: AVLTreeList
        @param lst: a list to be concatenated after self
        @rtype: int
        n1 - number on nodes in self, n2 - number of nodes in lst
		Time complexity: O(log(n1+n2))"""
    def right_concat(self, lst):
        h2 = lst.root.getHeight()
        l_node = self.root
        while l_node.getHeight() > h2:
            l_node = l_node.getRight()
        r_node = lst.root
        # adds node x for the join function
        x = AVLNode(None)
        x_index = self.size
        self.join(l_node, x, r_node, "right")
        self.size += lst.size + 1
        self.lastItem = lst.lastItem
        # deletes node x added for join
        self.delete(x_index)

    """concatenate the lists when self's height is the same as lst's
        @type lst: AVLTreeList
        @param lst: a list to be concatenated after self
        @rtype: int
        @returns: the absolute value of the difference between the height of the AVL trees joined
		Time complexity: O(logn)"""
    def same_height_concat(self, lst):
        # adds node x for the join function
        x = AVLNode(None)
        x_index = self.size
        a = self.root
        b = lst.root
        x.setLeft(a)
        x.setRight(b)
        self.root = x
        self.lastItem = lst.lastItem
        self.size += lst.size + 1
        self.update_fields(x)
        # deletes node x added for join
        self.delete(x_index)

    """@pre dir == "right" or dir == "left"
        joins self, x, lst when self < x < lst (by indexes in the new list)
        @type a,x,b: AVLNode, dir:str
        @param a: node in self to join
        @param x: node to use for the join function
        @param b: node in list to concat to join
        @param dir: indicates which concat function did we use depending on which tree is higher. 
        h1 - height of self, h2 - height of lst
		Time complexity: O(abs(h1-h2))"""
    def join(self, a, x, b, dir):
        if dir == "left":
            c = b.getParent()
        else:
            c = a.getParent()
        x.setLeft(a)
        x.setRight(b)
        if dir == "left":
            c.setLeft(x)
        else:
            c.setRight(x)
        self.update_fields(x)
        self.rebalance_tree(x)

    """searches for a *value* in the list
	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""
    """Time complexity: O(n)"""
    def search(self, val):
        arr = self.listToArray()
        for i in range(len(arr)):
            if arr[i] == val:
                return i
        else:
            return -1

    """returns the root of the tree representing the list
	@rtype: AVLNode
	@returns: the root, None if the list is empty
	"""
    """Time complexity: O(1)"""
    def getRoot(self):
        if self.root is None:
            return None
        return self.root


    """***************AUXILIARY FUNCTIONS*************************************"""

    """returns the K-th item
        @type k: int
        @param k: the rank of the node we want to retrieve 
        @rtype AVLNode
        @returns the k-th item in the tree
		Time complexity: O(logn)"""
    def Tree_select(self, k):
        return self.Tree_select_rec(self.root, k)

    def Tree_select_rec(self, node, k):
        curr_rank = node.getLeft().getSize() + 1
        if k == curr_rank:
            return node
        elif k < curr_rank:
            return self.Tree_select_rec(node.getLeft(), k)
        else:
            return self.Tree_select_rec(node.getRight(), k - curr_rank)

    """returns the last node in a node's subtree
        @type node: AVLNode
        @param node: the node to return the last node in its sub-tree
        @rtype AVLNode
        @returns the last node in node's subtree
		Time complexity: O(logn)"""
    def get_max(self, node):
        while node.getRight().isRealNode():
            node = node.getRight()

        return node

    """returns the first node in a node's subtree
        @type node: AVLNode
        @param node: the node to return the first node in its sub-tree
        @rtype AVLNode
        @returns the first node in node's subtree
        Time complexity: O(logn)"""
    def get_min(self, node):
        while node.getLeft().isRealNode():
            node = node.getLeft()

        return node

    """returns the predecessor of a node
        @type node: AVLNode
        @param node: a node which predecessor we are to return
        @rtype AVLNode
        @returns node's predecessor
        Time complexity: O(logn)"""
    def predecessor(self, node):
        if node.getLeft().isRealNode():
            return self.get_max(node.getLeft())
        par = node.getParent()
        while par is not None and node == par.getLeft():
            node = par
            par = node.getParent()

        return par

    """returns the successor of a node
        @type node: AVLNode
        @param node: a node which successor we are to return
        @rtype AVLNode
        @returns node's successor
        Time complexity: O(logn)"""
    def successor(self, node):
        if node.getRight().isRealNode():
            return self.get_min(node.getRight())
        par = node.getParent()
        while par is not None and node == par.getRight():
            node = par
            par = node.getParent()

        return par

    """updates a node's size
        @type node: AVLNode
        @param node: the node to updates its size field
		Time complexity: O(1)"""
    def update_size(self, node):
        node.setSize(node.getLeft().getSize() + node.getRight().getSize() + 1)

    """updates a node's height
        @type node: AVLNode
        @param node: the node to updates its height field
		Time complexity: O(1)"""
    def update_height(self, node):
        node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)

    """updates a node's size and height
        @type node: AVLNode
        @param node: the node to updates its size and height fields
		Time complexity: O(1)"""
    def update_fields(self, node):
        self.update_height(node)
        self.update_size(node)

    """calculates the BF of a node
        @type node: AVLNode
        @param node: the node to calculate its balance factor
        @rtype int
        @returns node's balance factor
		Time complexity: O(1)"""
    def get_BF(self, node):
        return node.getLeft().getHeight() - node.getRight().getHeight()

    """rotates to the left the edge between the node and it's right child
        @type node: AVLNode
        @param node: the node with the BF violation from which we need to rotate
		Time complexity: O(1)"""
    def L_rotation(self, node):
        r_node = node.getRight()
        node_parent = node.getParent()
        rl_node = r_node.getLeft()
        if node == self.root:
            self.root = r_node
            r_node.setParent(None)
        else:
            if node_parent.getLeft() == node:
                node_parent.setLeft(r_node)
            else:
                node_parent.setRight(r_node)

        node.setRight(rl_node)
        r_node.setLeft(node)

        self.update_fields(node)
        self.update_fields(r_node)

    """a double rotation: rotates to the left the edge between the node's left child and it's right child
		then, rotates to the right the edge between the node and it's left child
		@type node: AVLNode
        @param node: the node with the BF violation from which we need to rotate
		Time complexity: O(1)"""
    def LR_rotation(self, node):
        l_node = node.getLeft()
        self.L_rotation(l_node)
        self.R_rotation(node)

    """rotates to the right the edge between the node and it's left child
        @type node: AVLNode
        @param node: the node with the BF violation from which we need to rotate
		Time complexity: O(1)"""
    def R_rotation(self, node):
        l_node = node.getLeft()
        node_parent = node.getParent()
        lr_node = l_node.getRight()
        if node == self.root:
            self.root = l_node
            l_node.setParent(None)
        else:
            if node_parent.getLeft() == node:
                node_parent.setLeft(l_node)
            else:
                node_parent.setRight(l_node)

        node.setLeft(lr_node)
        l_node.setRight(node)

        self.update_fields(node)
        self.update_fields(l_node)

    """a double rotation: rotates to the right the edge between the node's right child and it's left child
		then, rotates to the left the edge between the node and it's right child
		@type node: AVLNode
        @param node: the node with the BF violation from which we need to rotate
		Time complexity: O(1)"""
    def RL_rotation(self, node):
        r_node = node.getRight()
        self.R_rotation(r_node)
        self.L_rotation(node)

    """checks which BF violation occurred
		returns the number of rotations in this occurrence
		@type node: AVLNode
		@para node: a node with a BF violation
		@rtype int
		@returns the number of rotations in this occurrence, 1 if a single rotations, 2 if a double rotation
		Time complexity: O(1)"""
    def check_and_rotate(self, node):
        BF = self.get_BF(node)
        if BF == -2:
            BF_r = self.get_BF(node.getRight())
            if BF_r == -1 or BF_r == 0:
                self.L_rotation(node)
                return 1
            if BF_r == 1:
                self.RL_rotation(node)
                return 2
        if BF == 2:
            BF_l = self.get_BF(node.getLeft())
            if BF_l == -1:
                self.LR_rotation(node)
                return 2
            if BF_l == 1 or BF_l == 0:
                self.R_rotation(node)
                return 1
        return 0

    """goes up the tree and checks if the BF invariant is violated 
		returns the number of rotations made to fix the tree
		@type node: AVLNode
		@param node: the parent of the node we inserted/deleted
		@rtype int
		@returns the number of rebalancing operation due to AVL rebalancing
		Time complexity: O(logn)"""
    def rebalance_tree(self, node):
        cnt_rotations = 0
        while node is not None:
            self.update_fields(node)
            BF = self.get_BF(node)
            if abs(BF) < 2 and BF == 0:
                node = node.getParent()
            elif abs(BF) < 2 and BF != 0:
                node = node.getParent()
            else:
                num_rot = self.check_and_rotate(node)
                cnt_rotations += num_rot
                node = node.getParent()

        return cnt_rotations


    """"""""""""""""""""""""""""""""""""""

    def printt(self):
        out = ""
        for row in self.printree(self.root):  # need printree.py file
            out = out + row + "\n"
        print(out)

    def printree(self, t, bykey=True):
        # for row in trepr(t, bykey):
        #        print(row)
        return self.trepr(t, False)

    def trepr(self, t, bykey=False):
        if t == None:
            return ["#"]

        thistr = str(t.key) if bykey else str(t.getValue())

        return self.conc(self.trepr(t.left, bykey), thistr, self.trepr(t.right, bykey))

    def conc(self, left, root, right):

        lwid = len(left[-1])
        rwid = len(right[-1])
        rootwid = len(root)

        result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

        ls = self.leftspace(left[0])
        rs = self.rightspace(right[0])
        result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid *
                      " " + "\\" + rs * "_" + (rwid - rs) * " ")

        for i in range(max(len(left), len(right))):
            row = ""
            if i < len(left):
                row += left[i]
            else:
                row += lwid * " "

            row += (rootwid + 2) * " "

            if i < len(right):
                row += right[i]
            else:
                row += rwid * " "

            result.append(row)

        return result

    def leftspace(self, row):
        # row is the first row of a left node
        # returns the index of where the second whitespace starts
        i = len(row) - 1
        while row[i] == " ":
            i -= 1
        return i + 1

    def rightspace(self, row):
        # row is the first row of a right node
        # returns the index of where the first whitespace ends
        i = 0
        while row[i] == " ":
            i += 1
        return i

    def append(self, val):
        self.insert(self.length(), val)


"""********************************linked list****************************"""
class Node:
    def __init__(self, val):
        self.value = val
        self.next = None

    def __repr__(self):
        # return str(self.value)
        # This shows pointers as well for educational purposes:
        return "(" + str(self.value) + ", next: " + str(id(self.next)) + ")"

class Linked_list:
    def __init__(self, seq=None):
        self.head = None
        self.len = 0
        if seq != None:
            for x in seq[::-1]:
                self.add_at_start(x)

    def __repr__(self):
        out = ""
        p = self.head
        while p != None:
            out += str(p) + ", "  # str(p) invokes __repr__ of class Node
            p = p.next
        return "[" + out[:-2] + "]"

    def __len__(self):
        ''' called when using Python's len() '''
        return self.len

    def add_at_start(self, val):
        ''' add node with value val at the list head '''
        tmp = self.head
        self.head = Node(val)
        self.head.next = tmp
        self.len += 1

    def find(self, val):
        ''' find (first) node with value val in list '''
        p = self.head
        # loc = 0     # in case we want to return the location
        while p != None:
            if p.value == val:
                return p
            else:
                p = p.next
                # loc=loc+1   # in case we want to return the location
        return None

    def __getitem__(self, loc):
        ''' called when using L[i] for reading
            return node at location 0<=loc<len '''
        assert 0 <= loc < len(self)
        p = self.head
        for i in range(0, loc):
            p = p.next
        return p

    def __setitem__(self, loc, val):
        ''' called when using L[loc]=val for writing
            assigns val to node at location 0<=loc<len '''
        assert 0 <= loc < len(self)
        p = self.head
        for i in range(0, loc):
            p = p.next
        p.value = val
        return None

    def insert(self, loc, val):
        ''' add node with value val after location 0<=loc<len of the list '''
        assert 0 <= loc <= len(self)
        if loc == 0:
            self.add_at_start(val)
        else:
            p = self.head
            for i in range(0, loc - 1):
                p = p.next
            tmp = p.next
            p.next = Node(val)
            p.next.next = tmp
            self.len += 1

    def delete(self, loc):
        ''' delete element at location 0<=loc<len '''
        assert 0 <= loc < len(self)
        if loc == 0:
            self.head = self.head.next
        else:
            p = self.head
            for i in range(0, loc - 1):
                p = p.next
            # p is the element BEFORE loc
            p.next = p.next.next
        self.len -= 1






import timeit
#Q2
# print("insert-first")
# print("AVLTree")
# for i in range(1,11):
#     n = 1500*i
#     T = AVLTreeList()
#     start = timeit.default_timer()
#     for j in range(n):
#         T.insert(0, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
# print("linkedlist")
# for i in range(1,11):
#     n = 1500*i
#     T = Linked_list()
#     start = timeit.default_timer()
#     for j in range(n):
#         T.insert(0, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
# print("array")
# for i in range(1,11):
#     n = 1500*i
#     T = []
#     start = timeit.default_timer()
#     for j in range(n):
#         T.insert(0, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
#
# print("insert randomly")
#
# print("AVLTree")
# for i in range(1,11):
#     n = 1500*i
#     T = AVLTreeList()
#     start = timeit.default_timer()
#     for j in range(n):
#         index = random.randint(0, T.size)
#         T.insert(index, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
# print("linkedlist")
# for i in range(1,11):
#     n = 1500*i
#     T = Linked_list()
#     start = timeit.default_timer()
#     for j in range(n):
#         index = random.randint(0, T.len)
#         T.insert(index, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
# print("array")
# for i in range(1,11):
#     n = 1500*i
#     T = []
#     start = timeit.default_timer()
#     for j in range(n):
#         index = random.randint(0, len(T))
#         T.insert(index, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
#
# print("insert-last")
# print("AVLTree")
# for i in range(1,11):
#     n = 1500*i
#     T = AVLTreeList()
#     start = timeit.default_timer()
#     for j in range(n):
#         T.insert(T.size, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
# print("linkedlist")
# for i in range(1,11):
#     n = 1500*i
#     T = Linked_list()
#     start = timeit.default_timer()
#     for j in range(n):
#         T.insert(T.len, j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))
# print("array")
# for i in range(1,11):
#     n = 1500*i
#     T = []
#     start = timeit.default_timer()
#     for j in range(n):
#         T.insert(len(T), j)
#     stop = timeit.default_timer()
#     time = stop - start
#     print(str(time/n))


# for i in range(1,11):
#     T = AVLTreeList()
#     n = 1500*(2**i)
#     sum_insert = 0
#     sum_insert_delete = 0
#     for j in range(n//2):
#         index = random.randint(0, T.size)
#         sum_insert += T.insert(index,index)
#
#
#
#     for k in range(n//4):
#         index = random.randint(0, T.size)
#         sum_insert_delete += T.insert(index, index)
#         index = random.randint(0, T.size)
#         sum_insert_delete += T.delete(index)
#
#     print(sum_insert_delete)






