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
print("insert-first")
print("AVLTree")
for i in range(1,11):
    n = 1500*i
    T = AVLTreeList()
    start = timeit.default_timer()
    for j in range(n):
        T.insert(0, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))
print("linkedlist")
for i in range(1,11):
    n = 1500*i
    T = Linked_list()
    start = timeit.default_timer()
    for j in range(n):
        T.insert(0, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))
print("array")
for i in range(1,11):
    n = 1500*i
    T = []
    start = timeit.default_timer()
    for j in range(n):
        T.insert(0, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))

print("insert randomly")

print("AVLTree")
for i in range(1,11):
    n = 1500*i
    T = AVLTreeList()
    start = timeit.default_timer()
    for j in range(n):
        index = random.randint(0, T.size)
        T.insert(index, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))
print("linkedlist")
for i in range(1,11):
    n = 1500*i
    T = Linked_list()
    start = timeit.default_timer()
    for j in range(n):
        index = random.randint(0, T.len)
        T.insert(index, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))
print("array")
for i in range(1,11):
    n = 1500*i
    T = []
    start = timeit.default_timer()
    for j in range(n):
        index = random.randint(0, len(T))
        T.insert(index, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))

print("insert-last")
print("AVLTree")
for i in range(1,11):
    n = 1500*i
    T = AVLTreeList()
    start = timeit.default_timer()
    for j in range(n):
        T.insert(T.size, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))
print("linkedlist")
for i in range(1,11):
    n = 1500*i
    T = Linked_list()
    start = timeit.default_timer()
    for j in range(n):
        T.insert(T.len, j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))
print("array")
for i in range(1,11):
    n = 1500*i
    T = []
    start = timeit.default_timer()
    for j in range(n):
        T.insert(len(T), j)
    stop = timeit.default_timer()
    time = stop - start
    print(str(time/n))



