import numpy as np
from abc import ABC, abstractmethod
import numpy
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


class Buffer(ABC):
    def __init__(self, maxlen, maxsave=None):
        self.data = [None] * (maxlen + 1)
        self.start = 0
        self.maxlen = maxlen
        self.length = 0
        if maxsave is None:
            maxsave = maxlen+1
        self.maxsave = maxsave

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        if i < 0 or i >= self.length:
            raise IndexError()
        return self.data[(self.start + i) % self.maxlen]

    @abstractmethod
    def random_sample(self, **kwargs):
        pass

    def update_tree(self, i, error):
        pass

    def append(self, v, **kwargs):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.data, file)

    def load(self, path):
        with open(path, "rb") as file:
            self.data = pickle.load(file)
        self.maxlen = len(self.data)
        self.length = len([i for i in self.data if i is not None])


class SimpleMemory(Buffer):
    def __init__(self, max_len):
        super(SimpleMemory, self).__init__(max_len)

    def random_sample(self, **kwargs):

        size = kwargs['size']
        depth = kwargs.get('depth', 4)
        shape = kwargs.get('shape', (84, 84))

        assert (len(self) > 0)
        if size > self.maxlen:
            size = self.maxlen

        idx = np.random.choice(np.arange(len(self)), size, replace=False)

        _states = np.zeros((size, depth, shape[0], shape[1]), dtype=np.uint8)
        _next_states = np.zeros((size, depth, shape[0], shape[1]), dtype=np.uint8)

        _actions = np.zeros(size, dtype=np.uint8)
        _rewards = np.zeros(size)
        _is_done = np.zeros(size, dtype=bool)

        for b, i in enumerate(idx):
            d = self.data[i]

            stack = np.asarray(d[0])
            nstack = np.asarray(d[3])

            _states[b] = stack
            _next_states[b] = nstack
            _actions[b] = d[1]
            _rewards[b] = d[2]
            _is_done[b] = d[4]

        return _states, _actions, _rewards, _next_states, _is_done


class PrioritizedMemory:

    def __init__(self, max_len, alpha=0.6, eps=0.001, beta=0.4):
        self.tree = SumTree(max_len)
        self.alpha = alpha
        self.eps = eps
        self.maxlen = max_len
        self.beta = 0.6
        # self.ann = AnnealedPolicy(inner_policy=EpsPolicy(self.beta), attr='eps', value_max=1.0, value_min=0.6,
        #                           value_test=0.5, nb_steps=5000, to_max=True)

    def update_tree(self, idx, error):
        error = (np.clip(error, 0.0, None) + self.eps) ** self.beta
        self.tree.update(idx, error)

    def append(self, v, **kwargs):
        error = kwargs['error']
        error = (np.clip(error, 0, None) + self.eps) ** self.beta
        self.tree.add(error, v)

    def __len__(self):
        return self.tree.write

    def random_sample(self, **kwargs):
        size = kwargs['size']
        depth = kwargs.get('depth', 4)
        shape = kwargs.get('shape', (84, 84))

        assert (len(self) > 0)
        if size > len(self):
            size = len(self)

        indexes = []
        segment = self.tree.total() / size

        _states = np.zeros((size, depth, shape[0], shape[1]), dtype=np.uint8)
        _next_states = np.zeros((size, depth, shape[0], shape[1]), dtype=np.uint8)

        _actions = np.zeros(size, dtype=np.uint8)
        _rewards = np.zeros(size)
        _is_done = np.zeros(size, dtype=bool)

        for i in range(size):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, d = self.tree.get(lower_bound)
            indexes.append(idx)
            _states[i] = np.asarray(d[0])
            _next_states[i] = np.asarray(d[3])
            _actions[i] = d[1]
            _rewards[i] = d[2]
            _is_done[i] = d[4]
        return _states, _actions, _rewards, _next_states, _is_done, indexes#, np.ones(size)
