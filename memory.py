import numpy as np


class Buffer(object):
    def __init__(self, maxlen):
        self.data = [None] * (maxlen + 1)
        self.start = 0
        self.maxlen = maxlen
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < 0 or i >= self.length:
            raise IndexError()
        return self.data[(self.start + i) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class Memory():
    def __init__(self, maxlen):
        self._mem = Buffer(maxlen)

    def append(self, r):
        self._mem.append(r)

    def random_sample(self, size):
        ln_mem = self._mem
        if size > len(ln_mem):
            size = len(ln_mem)
        idx = np.random.choice(np.arange(len(ln_mem)), size, replace=False)
        sample = []
        for i in idx:
            sample.append(self._mem[i])

        return sample


class SimpleMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.actions = Buffer(maxlen)
        self.rewards = Buffer(maxlen)
        self.is_done = Buffer(maxlen)
        self.states = Buffer(maxlen)
        self.next_states = Buffer(maxlen)

    def random_sample(self, size):
        if size > self.maxlen:
            size = self.maxlen

        exp = []
        idx = np.random.choice(np.arange(self.maxlen), size, replace=False)
        for i in idx:
            exp.append((self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.is_done))
        return exp

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.is_done.append(done)

b = Buffer(10)
b.append(1)
print(b.data)