import numpy as np
import json
import pickle
import os


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

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

        if self.write >= self.capacity:
            self.write = 0

        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s, story_len=4):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        story = []
        for i in range(1, story_len):
            off = dataIdx-i
            if off > 0:
                story.append(self.data[off][0])
        return idx, self.tree[idx], self.data[dataIdx], story

    def save(self, path):
        d = {
            'tree': self.tree,
            'data': self.data,
            'write': self.write,
            'capacity': self.capacity
        }
        with open(os.path.join(path, 'memory.mem'), 'wb') as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(os.path.join(path, 'memory.mem'), 'rb') as f:
            d = pickle.load(f)
        for k, v in d.items():
            setattr(self, k, v)
