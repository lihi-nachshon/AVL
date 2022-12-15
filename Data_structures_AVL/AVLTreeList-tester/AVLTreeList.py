#username - complete info
#id1      - complete info
#name1    - complete info
#id2      - complete info
#name2    - complete info  
import random


"""A class represnting a node in an AVL tree"""

class AVLNode(object):

	"""Constructor, you are allowed to add more fields.
	@type value: str
	@param value: data of your node
	"""
	def __init__(self, value, isReal = True, parent = None):
		self.value = value
		if(isReal):
			self.left = AVLNode(None, False, self)
			self.right = AVLNode(None, False, self)
			self.parent = None
			self.height = 0
			self.size = 1
		# if is a virtual node
		else:
			self.left = None
			self.right = None
			self.parent = parent
			self.height = -1
			self.size = 0
		self.is_real = isReal

	"""returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""
	def getLeft(self):
		if self.is_real:
			return self.left
		return None

	"""returns the right child
	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""
	def getRight(self):
		if self.is_real:
			return self.right
		return None

	"""returns the parent 
	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""
	def getParent(self):
		return self.parent

	"""return the value
	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""
	def getValue(self):
		if self.is_real:
			return self.value
		return None

	"""returns the height
	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""
	def getHeight(self):
		if self.is_real:
			return self.height
		return -1

	def getSize(self):
		if self.is_real:
			return self.size
		return 0

	"""sets left child
	@type node: AVLNode
	@param node: a node
	"""
	def setLeft(self, node):
		self.left = node

	"""sets right child
	@type node: AVLNode
	@param node: a node
	"""
	def setRight(self, node):
		self.right = node

	"""sets parent
	@type node: AVLNode
	@param node: a node
	"""
	def setParent(self, node):
		self.parent = node

	"""sets value
	@type value: str
	@param value: data
	"""
	def setValue(self, value):
		self.value = value

	"""sets the balance factor of the node
	@type h: int
	@param h: the height
	"""
	def setHeight(self, h):
		self.height = h

	def setSize(self, s):
		self.size = s

	"""returns whether self is not a virtual node 
	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""
	def isRealNode(self):
		return self.is_real



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
			new_node.setParent(node_first)
			self.firstItem = new_node
		# if insert last
		elif i == n:
			node_last = self.lastItem
			node_last.setRight(new_node)
			new_node.setParent(node_last)
			self.lastItem = new_node
		else:
			# finds the current node in position and inserts before it
			node_successor = self.Tree_select(i + 1)
			# if the current node in position has no left child
			if not node_successor.getLeft().isRealNode():
				node_successor.setLeft(new_node)
				new_node.setParent(node_successor)
			else:
				node_predecessor = self.predecessor(node_successor)
				node_predecessor.setRight(new_node)
				new_node.setParent(node_predecessor)

		self.size += 1
		self.update_size_till_root(new_node)
		cnt_rotations = self.insert_fix_tree(new_node)

		return cnt_rotations

	"""Inserts a node when tree is empty
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
		Time complexity: O(logn)"""
	def delete_leaf(self, node):
		node_parent = node.getParent()
		if node_parent.getLeft() == node:
			node_parent.setLeft(AVLNode(None, False, node_parent))
		else:
			node_parent.setRight(AVLNode(None, False, node_parent))
		node.setParent(None)
		self.size -= 1
		self.update_size_till_root(node_parent)
		cnt_rotations = self.delete_fix_tree(node_parent)

		return cnt_rotations

	"""deletes node with only a right child
		Time complexity: O(logn)"""
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
		child.setParent(node_parent)
		if node_parent.getLeft() == node:
			node_parent.setLeft(child)
		else:
			node_parent.setRight(child)
		node.setParent(None)
		node.setRight(AVLNode(None, False, node))

		self.size -= 1
		self.update_size_till_root(node_parent)
		cnt_rotations = self.delete_fix_tree(node_parent)

		return cnt_rotations

	"""deletes node with only a left child
		Time complexity: O(logn)"""
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
		child.setParent(node_parent)
		if node_parent.getLeft() == node:
			node_parent.setLeft(child)
		else:
			node_parent.setRight(child)
		node.setParent(None)
		node.setLeft(AVLNode(None, False, node))
		self.size -= 1
		self.update_size_till_root(node_parent)
		cnt_rotations = self.delete_fix_tree(node_parent)

		return cnt_rotations

	"""deletes node with two children
		Time complexity: O(logn)"""
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
			node_successor.setParent(node_parent)

		node_lchild.setParent(node_successor)
		node_rchild.setParent(node_successor)
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
		list = []
		if self.size > 0:
			self.in_order(list)
		return list

	"""adds the values to a list via in order walk
		Time complexity: O(n)"""
	def in_order(self, list):
		self.in_order_rec(list, self.root)

	def in_order_rec(self, list, node):
		if node.isRealNode():
			self.in_order_rec(list, node.getLeft())
			list.append(node.getValue())
			self.in_order_rec(list, node.getRight())

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
		arr = self.listToArray()
		self.mergesort(arr)
		return self.create_tree_from_array(arr)

	"""Time complexity: O(nlogn)"""
	def mergesort(self, lst):
		n = len(lst)
		if n <= 1:
			return lst
		else:
			return self.merge(self.mergesort(lst[0:n//2]),self.mergesort(lst[n//2:]))

	"""Time complexity: O(n)"""
	def merge(self,A, B):
		n = len(A)
		m = len(B)
		C = [0 for i in range(n + m)]
		a, b, c = 0
		while a < n and b < m:
			if A[a] < B[b]:
				C[c] = A[a]
				a += 1
			else:
				C[c] = B[b]
				b += 1
			c += 1
		C[c:] = A[a:] + B[b:]

		return C

	"""permute the info values of the list 
	@rtype: list
	@returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
	"""
	"""Time complexity: O(n)"""
	def permutation(self):
		arr = self.listToArray()
		# shuffles arr
		for i in range(len(arr) - 1, 0, -1):
			j = random.randint(0, i)
			temp = arr[j]
			arr[j] = arr[i]
			arr[i] = temp

		return self.create_tree_from_array(arr)

	"""creates a new AVLTreeList from an array
		Time complexity: O(n)"""
	def create_tree_from_array(self, arr):
		# creates the structure
		root = self.create_tree_from_array_rec(arr)
		# makes structure a tree
		tree = AVLTreeList()
		tree.root = root
		tree.size = len(arr)
		tree.firstItem = tree.get_min(tree.root)
		tree.lastItem = tree.get_max(tree.root)

		return tree

	"""creates an AVLTreeList structure from an array
		Time complexity: O(n)"""
	def create_tree_from_array_rec(self, arr):
		if len(arr) == 0:
			return AVLNode(None, False)
		if len(arr) == 1:
			return AVLNode(arr[0], True)
		mid = int((len(arr))/2)
		node_parent = AVLNode(arr[mid])
		left = arr[:mid]
		right = arr[mid + 1:]
		l_child = self.create_tree_from_array_rec(left)
		r_child = self.create_tree_from_array_rec(right)
		node_parent.setRight(r_child)
		node_parent.setLeft(l_child)
		l_child.setParent(node_parent)
		r_child.setParent(node_parent)
		self.update_fields(node_parent)

		return node_parent

	"""concatenates lst to self
	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""
	"""Time complexity: O(logn)"""
	def concat(self, lst):
		if self.size == 0:
			if lst.size == 0:
				return 0
			self.size = lst.size
			self.root = lst.root
			self.firstItem = lst.firstItem
			self.lastItem = lst.lastItem
			return lst.root.height + 1  # empty tree height is -1
		if lst.size == 0:
			return self.root.height + 1  # empty tree height is -1
		height_difference = abs(self.root.height - lst.root.height)
		if self.root.height > lst.root.height:
			self.right_concat(lst)
		elif self.root.height < lst.root.height:
			self.left_concat(lst)
		else:
			self.same_height_concat(lst)

		return height_difference

	"""concats the lists when self's height is bigger
		Time complexity: O(logn)"""
	def left_concat(self, lst):
		h1 = self.root.getHeight()
		r_node = lst.root
		while(r_node.getHeight() > h1):
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
		Time complexity: O(logn)"""
	def right_concat(self, lst):
		h2 = lst.root.getHeight()
		l_node = self.root
		while (l_node.getHeight() > h2):
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

	"""concats the lists when self's height is the same as lst's
		Time complexity: O(logn)"""
	def same_height_concat(self, lst):
		# adds node x for the join function
		x = AVLNode(0)
		x_index = self.size
		a = self.root
		b = lst.root
		x.setLeft(a)
		x.setRight(b)
		a.setParent(x)
		b.setParent(x)
		self.root = x
		self.lastItem = lst.lastItem
		self.size += lst.size + 1
		self.update_fields(x)
		# deletes node x added for join
		self.delete(x_index)

	"""joins self, x, lst when self < x < lst (by indexes in the new list)
		Time complexity: O(logn)"""
	def join(self, a, x, b, dir):
		x.setLeft(a)
		x.setRight(b)
		if dir == "left":
			c = b.getParent()
		else:
			c = a.getParent()
		a.setParent(x)
		b.setParent(x)
		x.setParent(c)
		if dir == "left":
			c.setLeft(x)
		else:
			c.setRight(x)
		self.update_fields(x)
		self.update_size_till_root(c)
		self.insert_fix_tree(x)

	"""searches for a *value* in the list
	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""
	"""Time complexity: O(n)"""
	def search(self, val):
		arr = self.listToArray()
		if val in arr:
			return arr.index(val)
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


	"""auxiliary functions"""

	"""returns the K-th item
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
		Time complexity: O(logn)"""
	def get_max(self, node):
		while(node.getRight().isRealNode()):
			node = node.getRight()

		return node

	"""returns the first node in a node's subtree
		Time complexity: O(logn)"""
	def get_min(self, node):
		while(node.getLeft().isRealNode()):
			node = node.getLeft()

		return node

	"""Time complexity: O(logn)"""
	def predecessor(self, node):
		if node.getLeft().isRealNode():
			return self.get_max(node.getLeft())
		par = node.getParent()
		while(par is not None and node == par.getLeft()):
			node = par
			par = node.getParent()

		return par

	"""Time complexity: O(logn)"""
	def successor(self, node):
		if node.getRight().isRealNode():
			return self.get_min(node.getRight())
		par = node.getParent()
		while(par is not None and node == par.getRight()):
			node = par
			par = node.getParent()

		return par

	"""updates a node's size
		Time complexity: O(1)"""
	def update_size(self, node):
		node.setSize(node.getLeft().getSize() + node.getRight().getSize() + 1)

	"""updates a node's height
		Time complexity: O(1)"""
	def update_height(self, node):
		node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)

	"""updates a node's size and height
		Time complexity: O(1)"""
	def update_fields(self, node):
		self.update_height(node)
		self.update_size(node)

	"""calculates the BF of a node
		Time complexity: O(1)"""
	def get_BF(self, node):
		return node.getLeft().getHeight() - node.getRight().getHeight()

	"""rotates to the left the edge between the node and it's right child
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
			r_node.setParent(node_parent)
		node.setParent(r_node)
		node.setRight(rl_node)
		rl_node.setParent(node)
		r_node.setLeft(node)

		self.update_fields(node)
		self.update_fields(r_node)

	"""a double rotation: rotates to the left the edge between the node's left child and it's right child
		then, rotates to the right the edge between the node and it's left child
		Time complexity: O(1)"""
	def LR_rotation(self, node):
		l_node = node.getLeft()
		self.L_rotation(l_node)
		self.R_rotation(node)

	"""rotates to the right the edge between the node and it's left child
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
			l_node.setParent(node_parent)
		node.setParent(l_node)
		node.setLeft(lr_node)
		lr_node.setParent(node)
		l_node.setRight(node)

		self.update_fields(node)
		self.update_fields(l_node)

	"""a double rotation: rotates to the right the edge between the node's right child and it's left child
		then, rotates to the left the edge between the node and it's right child
		Time complexity: O(1)"""
	def RL_rotation(self, node):
		r_node = node.getRight()
		self.R_rotation(r_node)
		self.L_rotation(node)

	"""checks which BF violation occurred
		returns the number of rotations in this occurrence 
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
		Time complexity: O(logn)"""
	def insert_fix_tree(self, node):
		cnt_rotations = 0
		y = node.getParent()
		while(y is not None):
			prev_height = y.getHeight()
			self.update_fields(y)
			BF = self.get_BF(y)
			if abs(BF) < 2 and prev_height == y.getHeight():
				break
			elif abs(BF) < 2 and prev_height != y.getHeight():
				y = y.getParent()
				continue
			else:
				num_rot = self.check_and_rotate(y)
				cnt_rotations += num_rot
				break

		return cnt_rotations

	"""goes up the tree till root and updates the size field of the nodes
		Time complexity: O(logn)"""
	def update_size_till_root(self, node):
		while(node is not None):
			self.update_size(node)
			node = node.getParent()

	"""goes up the tree and checks if the BF invariant is violated 
		returns the number of rotations made to fix the tree
		Time complexity: O(logn)"""
	def delete_fix_tree(self, node):
		cnt_rotations = 0
		y = node
		while(y is not None):
			prev_height = y.getHeight()
			self.update_fields(y)
			BF = self.get_BF(y)
			if abs(BF) < 2 and prev_height == y.getHeight():
				y = y.getParent()
				continue
			elif abs(BF) < 2 and prev_height != y.getHeight():
				y = y.getParent()
				continue
			else:
				num_rot = self.check_and_rotate(y)
				cnt_rotations += num_rot
				y = y.getParent()

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


T = AVLTreeList()
T.insert(0,"4") ; T.insert(1,"6") ; T.insert(0,"2") ; T.insert(0,"1")
T.insert(2,"3") ; T.insert(4,"5") ; T.insert(6,"7")
print(T.delete(2))
print(T.delete(0))