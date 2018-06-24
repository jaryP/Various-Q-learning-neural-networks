import numpy
import json
import pickle

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2*capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def save(self, path):
        d = {
            'tree': self.tree,
            'data': self.data,
            'write': self.write,
            'capacity': self.capacity
        }
        with open(path, 'wb') as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for k, v in d.items():
            setattr(self, k, v)
