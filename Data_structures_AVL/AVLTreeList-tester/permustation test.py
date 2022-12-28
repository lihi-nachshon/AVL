def test_permutations(self):
    T2 = AVLTreeList()

    for i in range(50):
        T2.insert(i, i)
    for i in range(100):
        y = T2.permutation()
        self.in_order(y, y.getRoot(), self.check_size)
        self.in_order(y, y.getRoot(), self.check_height)
        self.in_order(y, y.getRoot(), self.check_BF)
        self.in_order(y, y.getRoot(), self.check_family)

def test_sort(self):
    T2 = AVLTreeList()

    for i in range(50):
        T2.insert(0, i)
    y = T2.sort()
    self.in_order(y, y.getRoot(), self.check_size)
    self.in_order(y, y.getRoot(), self.check_height)
    self.in_order(y, y.getRoot(), self.check_BF)
    self.in_order(y, y.getRoot(), self.check_family)