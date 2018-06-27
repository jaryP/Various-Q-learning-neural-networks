import numpy as np
import pickle
from abc import ABC, abstractmethod
from dqn.tree_less_mem import SumTree
import os


class Buffer(ABC):
    def __init__(self, maxlen, max_save=None):
        self.data = [None] * (maxlen + 1)
        self.start = 0
        self.maxlen = maxlen
        self.length = 0
        if max_save is None:
            max_save = maxlen+1
        self.max_save = max_save

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
        if len(self) >= self.max_save:
            d = {
                'data': self.data[-self.max_save:],
                'start': self.max_save,
                'maxlen': self.maxlen,
                'length': self.max_save
            }
        else:
            d = {
                'data': self.data,
                'start': self.start,
                'maxlen': self.maxlen,
                'length': self.length
            }
        with open(os.path.join(path, 'memory.mem'), 'wb') as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(os.path.join(path, 'memory.mem'), 'rb') as f:
            d = pickle.load(f)

        for k, v in d.items():
            if k == 'data':
                continue
            setattr(self, k, v)

        self.data = [None] * (self.maxlen + 1)
        for i, v in enumerate(d['data']):
            self.data[i] = v


class SimpleMemory(Buffer):
    def __init__(self, max_len, maxsave=None):
        super(SimpleMemory, self).__init__(max_len, maxsave)

    def random_sample(self, **kwargs):
        size = kwargs['size']
        depth = kwargs.get('depth', 4)

        assert (len(self) > 0)
        if size > len(self):
            size = len(self)

        idx = np.random.choice(np.arange(len(self)), size, replace=False)

        data_shape = self.data[0][3].shape

        image = False
        if len(data_shape) == 1:
            _states = np.zeros((size, depth+1, data_shape[0]))
        else:
            _states = np.zeros((size, depth+1, data_shape[0], data_shape[1]), dtype=np.uint8)
            image = True

        _actions = []
        _rewards = np.zeros(size)
        _is_done = []

        for b, i in enumerate(idx):
            _states[b, 0] = self.data[i][3]
            _states[b, 1] = list(self.data[i][0])[0]
            for j in range(1, depth):
                off = i - j
                if off > 0:
                    _states[b, j+1] = self.data[off][3]
            _rewards[b] = self.data[i][2]
            _actions.append(self.data[i][1])
            _is_done.append(self.data[i][4])
        if image:
            return _states[:, 1:, :, :], _actions, _rewards, _states[:, :-1, :, :], np.asarray(_is_done)
        else:
            return _states[:, 1:, :], _actions, _rewards, _states[:, :-1, :], np.asarray(_is_done)
    # def save(self, path):
    #     self.save(path)
    #
    # def load(self, path):
    #     super().load(path)


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

        data_shape = self.tree.data[0][3].shape

        image = False
        if len(data_shape) == 1:
            _states = np.zeros((size, depth+1, data_shape[0]))
        else:
            _states = np.zeros((size, depth+1, data_shape[0], data_shape[1]), dtype=np.uint8)
            image = True

        _actions = []
        _rewards = np.empty(size)
        # _next_states = np.zeros((size, depth, data_shape[0], data_shape[1]), dtype=np.uint8)

        _is_done = []
        for i in range(size):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data, story = self.tree.get(lower_bound, story_len=depth)
            indexes.append(idx)

            # s = data[0].copy()
            _states[i, 0] = data[3]
            # _next_states[i, 1] = data[0]
            _states[i, 1] = data[0]

            for j, v in enumerate(story):
                _states[i, j+2] = v
            _rewards[i] = data[2]
            _actions.append(data[1])
            _is_done.append(data[4])

        if image:
            return _states[:, 1:, :, :], _actions, _rewards, _states[:, :-1, :, :], np.asarray(_is_done)
        else:
            return _states[:, 1:, :], _actions, _rewards, _states[:, :-1, :], np.asarray(_is_done)


    def save(self, path):
        self.tree.save(path)

    def load(self, path):
        self.tree.load(path)
        # with open(path, "rb") as file:
        #     self.tree = pickle.load(file)
        # self.maxlen = len(self)

# m = PrioritizedMemory(5)
# s = (prev_states, action, reward, next_state, done, error=reward
# m.append(1, error=2)
# m.append(1, error=2)
# m.append(1, error=2)
# m.append(1, error=2)


