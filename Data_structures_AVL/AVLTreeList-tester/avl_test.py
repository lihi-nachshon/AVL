import unittest
import AVLTreeList


class TestAVLList(unittest.TestCase):
    def test_retrieve_sanity(self):
        tree = self._create_tree([10, 5, 7, 3, 0])
        self._valid_avl_tree(tree)
        self.assertEqual(10, tree.retrieve(0))
        self.assertEqual(5, tree.retrieve(1))
        self.assertEqual(7, tree.retrieve(2))
        self.assertEqual(3, tree.retrieve(3))
        self.assertEqual(0, tree.retrieve(4))
        self.assertIsNone(tree.retrieve(5))
        self.assertIsNone(tree.retrieve(10))

    def test_first_last_sanity(self):
        tree1 = self._create_tree([10, 5, 7, 3, 0])
        self.assertEqual(10, tree1.first())
        self.assertEqual(0, tree1.last())

        tree2 = self._create_tree([1, 1])
        self.assertEqual(1, tree2.first())
        self.assertEqual(1, tree2.last())

        tree3 = self._create_tree([0])
        self.assertEqual(0, tree3.first())
        self.assertEqual(0, tree3.last())

        tree4 = self._create_tree([])
        self.assertIsNone(tree4.first())
        self.assertIsNone(tree4.last())

    def test_list_to_array_sanity(self):
        lst1 = [10, 5, 7, 3, 0]
        tree1 = self._create_tree(lst1)
        self.assertEqual(lst1, tree1.listToArray())

        lst2 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        tree2 = self._create_tree(lst2)
        self.assertEqual(lst2, tree2.listToArray())

        lst3 = []
        tree3 = self._create_tree(lst3)
        self.assertEqual(lst3, tree3.listToArray())

    def test_search_sanity(self):
        tree1 = self._create_tree([1, 2, 5, 7, 2, 5])
        self.assertEqual(0, tree1.search(1))
        self.assertEqual(1, tree1.search(2))
        self.assertEqual(-1, tree1.search(3))

        tree2 = self._create_tree([2, 2, 2, 5])
        self.assertEqual(0, tree2.search(2))
        self.assertEqual(-1, tree2.search(3))

        tree3 = self._create_tree([])
        self.assertEqual(-1, tree3.search(2))
        self.assertEqual(-1, tree3.search(3))

    def test_insert_sanity(self):
        tree = self._create_tree([])

        count = self._insert(tree, 0, 1)
        self.assertEqual(0, count)
        self.assertEqual([1], tree.listToArray())

        count = self._insert(tree, 0, 2)
        self.assertEqual(1, count)
        self.assertEqual([2, 1], tree.listToArray(),)

        # trigger left right rotation
        count = self._insert(tree, 1, 3)
        self.assertEqual(3, count)
        self.assertEqual([2, 3, 1], tree.listToArray())

        # trigger two height changes
        count = self._insert(tree, 3, 4)
        self.assertEqual(2, count)
        self.assertEqual([2, 3, 1, 4], tree.listToArray())

        self._valid_avl_tree(tree)

    def test_concat_sanity(self):
        tree1 = self._create_tree([0])
        tree2 = self._create_tree([10])
        self._concat(tree1, tree2)
        self.assertEqual(0, tree1.search(0))
        self.assertEqual(1, tree1.search(10))
        self.assertEqual(0, tree1.retrieve(0))
        self.assertEqual(10, tree1.retrieve(1))

        tree3 = self._create_tree([1, 2, 3])
        tree4 = self._create_tree([1, 2, 3])
        self._concat(tree3, tree4)
        self.assertEqual(0, tree3.search(1))
        self.assertEqual(1, tree3.search(2))
        self.assertEqual(2, tree3.search(3))
        self.assertEqual(1, tree3.retrieve(0))
        self.assertEqual(2, tree3.retrieve(4))
        self.assertEqual(3, tree3.retrieve(5))

        tree5 = self._create_tree([1, 2])
        tree6 = self._create_tree([1, 2, 3, 4, 5, 6])
        self._concat(tree5, tree6)
        self.assertEqual(0, tree5.search(1))
        self.assertEqual(1, tree5.search(2))
        self.assertEqual(4, tree5.search(3))
        self.assertEqual(1, tree5.retrieve(0))
        self.assertEqual(2, tree5.retrieve(3))
        self.assertEqual(4, tree5.retrieve(5))

        tree7 = self._create_tree([1, 2, 8, 9, 23])
        tree8 = self._create_tree([0])
        self._concat(tree7, tree8)
        self.assertEqual(0, tree7.search(1))
        self.assertEqual(5, tree7.search(0))
        self.assertEqual(4, tree7.search(23))
        self.assertEqual(1, tree7.retrieve(0))
        self.assertEqual(23, tree7.retrieve(4))
        self.assertEqual(0, tree7.retrieve(5))

    def test_split_sanity(self):
        tree1 = self._create_tree([1])
        ret = self._split(tree1, 0)
        self.assertEqual(-1, ret[0].search(1))
        self.assertEqual(-1, ret[2].search(1))
        self.assertTrue(ret[0].empty())
        self.assertTrue(ret[2].empty())

        tree2 = self._create_tree([1, 2, 3])
        ret = self._split(tree2, 1)
        self.assertEqual(-1, ret[0].search(2))
        self.assertEqual(-1, ret[2].search(2))
        self.assertFalse(ret[0].empty())
        self.assertFalse(ret[2].empty())

        tree3 = self._create_tree([1, 2, 3, 4, 5])
        ret = self._split(tree3, 4)
        self.assertEqual(-1, ret[0].search(5))
        self.assertEqual(-1, ret[2].search(5))
        self.assertFalse(ret[0].empty())
        self.assertTrue(ret[2].empty())

        tree4 = self._create_tree([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ret = self._split(tree4, 2)
        self.assertEqual(-1, ret[0].search(3))
        self.assertEqual(-1, ret[2].search(3))
        self.assertFalse(ret[0].empty())
        self.assertFalse(ret[2].empty())

    def test_case_empty_tree(self):
        tree = AVLTreeList()
        self.assertTrue(tree.empty())
        self._valid_avl_tree(tree)

        self.assertEqual(-1, tree.delete(1))
        self.assertEqual(-1, tree.delete(0))

        self.assertEqual(-1, tree.search(-1))
        self.assertEqual(-1, tree.search(0))
        self.assertEqual(-1, tree.search(1))
        self.assertTrue(tree.empty())
        self._valid_avl_tree(tree)

    def test_case_delete_root(self):
        tree = self._create_tree([0, 1, 2, 3, 4])
        # no rotations
        count = self._delete(tree, 1)
        self.assertEqual(0, count)
        self.assertEqual(-1, tree.search(1))

        # trigger no rotation
        count = self._delete(tree, 1)
        self.assertEqual(0, count)
        self.assertEqual(-1, tree.search(2))

    def test_case_delete_first_last(self):
        # trigger left rotation
        tree = self._create_tree([0, 1, 2, 3, 4])
        count = self._delete(tree, 0)
        self.assertEqual(-1, tree.search(0))
        self.assertEqual(1, tree.retrieve(0))
        self.assertEqual(1, tree.first())
        self.assertEqual(1, count)

        # trigger left right rotation
        count = self._delete(tree, 3)
        self.assertEqual(-1, tree.search(4))
        self.assertEqual(3, tree.retrieve(2))
        self.assertEqual(3, tree.last())
        self.assertEqual(2, count)

    def test_case_insert_delete(self):
        tree = AVLTreeList()
        tree.insert(0, "value")
        self.assertEqual(1, tree.length())
        self._valid_avl_tree(tree)

        self.assertNotEqual(-1, tree.delete(0))
        self.assertTrue(tree.empty())
        self._valid_avl_tree(tree)

    def test_case_multiple_rotations(self):
        tree1 = self._create_tree([4, 8, 11, 15, 20, 22, 24])
        self._insert(tree1, 0, 2)
        self._insert(tree1, 3, 9)
        self._insert(tree1, 5, 12)
        self._insert(tree1, 7, 18)
        self._insert(tree1, 6, 13)
        # trigger right rotation and then left right rotation
        count = self._delete(tree1, 11)
        self.assertEqual(-1, tree1.search(24))
        self.assertEqual(3, count)

    def test_case_right_rotations(self):
        tree2 = self._create_tree([1, 2, 4, 5, 6])
        # trigger right left rotation with two height changes
        count = self._insert(tree2, 2, 3)
        self.assertEqual(4, count)
        self.assertEqual(3, tree2.retrieve(2))
        self.assertEqual(2, tree2.search(3))

        tree3 = self._create_tree([3, 4, 5])
        count = self._insert(tree3, 0, 2)
        self.assertEqual(2, count)
        # trigger right rotation
        count = self._insert(tree3, 0, 1)
        self.assertEqual(2, count)

    def test_case_split_zigzag(self):
        tree = self._create_tree([4, 8, 14, 17, 20, 22, 24])
        self._insert(tree, 0, 2)
        self._insert(tree, 3, 11)
        self._insert(tree, 5, 15)
        self._insert(tree, 7, 18)
        self._insert(tree, 3, 10)
        self._insert(tree, 5, 12)
        self._split(tree, 4)

    def _insert(self, tree, i, v):
        size = tree.length()
        count = tree.insert(i, v)
        self.assertEqual(size + 1, tree.length())
        self.assertEqual(v, tree.retrieve(i))
        return count

    def _delete(self, tree, i):
        size = tree.length()
        if i >= size:
            self.assertEqual(-1, tree.delete(i))
            self.assertEqual(size, tree.length())
            self._valid_avl_tree(tree)
            return -1

        next_item = tree.retrieve(i+1)
        count = tree.delete(i)
        self.assertEqual(next_item, tree.retrieve(i))
        self.assertEqual(size - 1, tree.length())
        self._valid_avl_tree(tree)
        return count

    def _concat(self, tree1, tree2):
        size1 = tree1.length()
        size2 = tree2.length()
        diff = abs(tree1.getHeight() - tree2.getHeight())

        ret = tree1.concat(tree2)
        self.assertEqual(diff, ret)
        self.assertEqual(size1 + size2, tree1.length())
        self._valid_avl_tree(tree1)

        return tree1

    def _split(self, tree, i):
        val = tree.retrieve(i)
        size = tree.length()
        as_array = tree.listToArray()
        next_ = tree.retrieve(i+1) if i < size else None
        prev_ = tree.retrieve(i-1) if i > 0 else None

        ret = tree.split(i)
        self.assertEqual(val, ret[1])
        self.assertTrue(isinstance(ret[0], AVLTreeList))
        self.assertTrue(isinstance(ret[2], AVLTreeList))

        left = ret[0].listToArray()
        right = ret[2].listToArray()
        self.assertEqual(as_array, left + [val] + right)
        self.assertEqual(i, ret[0].length())
        self.assertEqual(size - i - 1, ret[2].length())

        if i > 0:
            self.assertEqual(prev_, ret[0].last())
        else:
            self.assertTrue(ret[0].empty())

        if i < size:
            self.assertEqual(next_, ret[2].first())
        else:
            self.assertTrue(ret[2].empty())

        self._valid_avl_tree(ret[0])
        self._valid_avl_tree(ret[2])

        return ret

    def _valid_avl_tree(self, tree):
        size = tree.length()
        # height = tree.getHeight()
        first = tree.first()
        last = tree.last()
        empty = tree.empty()

        values = tree.listToArray()
        root = tree.getRoot()

        # first and last elements
        if not empty:
            self.assertEqual(first, values[0])
            self.assertEqual(last, values[len(values) - 1])

        # ranks and values match index
        # for i in range(tree.length()):
        #    i_node = tree.retrieve(i, pointer=True)
        #    self.assertEqual(tree.retrieve(i), values[i])
        #    self.assertEqual(i+1, i_node.getRank())

        if empty:
            self.assertTrue(root is None or not root.isRealNode())
            # self.assertEqual(height, -1)
            self.assertEqual(0, size)
            self.assertEqual(0, len(values))
            self.assertIsNone(first)
            self.assertIsNone(last)
            return

        # self.assertEqual(root.getHeight(), tree.getHeight())
        self.assertIsNotNone(first)
        self.assertIsNotNone(last)

    @staticmethod
    def _create_tree(values):
        assert isinstance(values, list)

        tree = AVLTreeList()
        for v in values:
            tree.insert(tree.length(), v)

        return tree
