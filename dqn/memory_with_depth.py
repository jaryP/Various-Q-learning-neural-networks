import numpy as np
import pickle
from abc import ABC, abstractmethod
from tree import SumTree


class Buffer(ABC):
    def __init__(self, maxlen):
        self.data = [None] * (maxlen + 1)
        self.start = 0
        self.maxlen = maxlen
        self.length = 0

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


class BinaryHeatTree:
    def __init__(self, max_len):
        pass
        self.last_val = 0
        self.max_len = max_len
        self.tree = np.zeros(2 * max_len - 1)
        self.indexes = np.zeros(max_len, dtype=object)

    def root(self):
        return self.tree[0]

    def add_new_node(self, p, ix):

        leaf = self.last_val + self.max_len - 1
        self.indexes[self.last_val] = ix
        self.update(leaf, p)
        self.last_val += 1
        if self.last_val >= self.max_len:
            self.last_val = 0

    def update(self, i, p):

        delta = p - self.tree[i]
        self.tree[i] = p
        self._propagate_error(i, delta)

    def _propagate_error(self, i, delta):

        parent = (i - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            # aggiorna ricorsivamente se non è la radice dell'albero
            self._propagate_error(parent, delta)

    def get_leaf(self, p):

        leaf_idx = self._walk(p)
        data_idx = leaf_idx - self.max_len + 1
        return [leaf_idx, self.tree[leaf_idx], self.indexes[data_idx]]

    def _walk(self, prob, i=0):

        left = 2 * i + 1
        right = left + 1

        if self.tree[left] == 0:
            return i
        if left >= self.last_val:  # end search when no more child
            return i

        # if self.tree[left_i] == 0 and self.tree[right_i] == 0:

        if self.tree[left] == self.tree[right]:
            # i valori sono uguali quindi ne prendo uno a caso
            return self._walk(prob, np.random.choice([left, right]))
        if prob <= self.tree[left]:
            return self._walk(prob, left)
        else:
            return self._walk(prob - self.tree[left], right)


class SimpleMemory(Buffer):
    def __init__(self, max_len):
        super(SimpleMemory, self).__init__(max_len)

    def random_sample(self, **kwargs):

        size = kwargs['size']
        depth = kwargs.get('depth', 4)

        assert (len(self) > 0)
        if size > self.maxlen:
            size = self.maxlen

        idx = np.random.choice(np.arange(len(self)), size, replace=False)
        _states = []
        _actions = []
        _rewards = []
        _next_states = []
        _is_done = []

        for i in idx:
            s = self.data[i][-1].copy()
            _states.append(np.asarray(s))
            _actions.append(self.data[i][1])
            _rewards.append(self.data[i][2])
            s.appendleft(self.data[i][3])
            _next_states.append(np.asarray(s))
            _is_done.append(self.data[i][4])

        return np.asarray(_states), np.asarray(_actions), np.asarray(_rewards), np.asarray(_next_states), \
               np.asarray(_is_done)


class PrioritizedMemory:

    def __init__(self, max_len, alpha=0.6, eps=0.001):
        self.tree = SumTree(max_len)
        self.alpha = alpha
        self.eps = eps
        self.maxlen = max_len

    def update_tree(self, idx, error):
        error = (np.clip(error, 0.0, None) + self.eps)**self.alpha
        self.tree.update(idx, error)

    def append(self, v, **kwargs):
        error = kwargs['error']
        error = (np.clip(error, 0, None) + self.eps) ** self.alpha
        self.tree.add(error, v)

    def __len__(self):
        return self.tree.write

    def random_sample(self, **kwargs):
        size = kwargs['size']
        depth = kwargs.get('depth', 4)

        assert (len(self) > 0)
        if size > len(self):
            size = len(self)

        indexes = []

        segment = self.tree.total() / size

        _states = []
        _actions = []
        _rewards = []
        _next_states = []
        _is_done = []
        for i in range(size):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get(lower_bound)
            indexes.append(idx)

            s = data[-1].copy()
            _states.append(np.asarray(s))
            _actions.append(data[1])
            _rewards.append(data[2])

            s.appendleft(data[3])
            _next_states.append(np.asarray(s))
            del s
            _is_done.append(data[4])

        return np.asarray(_states), np.asarray(_actions), np.asarray(_rewards), np.asarray(_next_states), \
               np.asarray(_is_done), indexes

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)

    def load(self, path):
        with open(path, "rb") as file:
            self.tree = pickle.load(file)
        self.maxlen = len(self)


