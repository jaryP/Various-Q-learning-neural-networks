import tqdm
import os
os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32"
# os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
os.environ['CPLUS_INCLUDE_PATH'] = '/usr/local/cuda/include'
from abc import ABC, abstractmethod
from dqn.memory_with_depth import PrioritizedMemory
import gym
import policy
from dqn import models_with_depth as models
import numpy as np
import json
import pickle
from collections import  deque


class AbstractAgent(ABC):
    def __init__(self,  network, max_len_memory, to_observe, pol, gamma, log_dir, load_prev, game):

        self.env = gym.make(game)
        self.env.seed(19)
        self.action_meaning = self.env.env.get_action_meanings()
        self.no_op_ep = 30
        self.env._max_episode_steps = None
        self.model = network
        print(network.model.summary())
        self.batch_size = 32
        self.to_observe = to_observe
        self.state_size = network.state_size
        self.action_size = network.action_size
        self.log_dir = log_dir
        self.depth = network.depth

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        attr = {'batch size': self.batch_size, 'to observe': self.to_observe,
                'depth': self.depth, 'no_op_ep': 30}

        self.results = {'info': attr}

        self.memory = PrioritizedMemory(max_len=max_len_memory)

        if load_prev:
            path = sorted([int(x) for x in os.listdir(self.log_dir) if os.path.isdir(
                os.path.join(self.log_dir, x))])
            if len(path) != 0:
                load_prev = self.load(os.path.join(self.log_dir, str(path[-1])))
            else:
                load_prev = False

        if not load_prev:
            if pol is None:
                self.pol = policy.GreedyPolicy()
            else:
                self.pol = pol

            if gamma is None:
                gamma = policy.EpsPolicy(0.99)

            if isinstance(gamma, policy.AnnealedPolicy):
                self.gamma = gamma.linear_step
            elif isinstance(gamma, policy.Policy):
                self.gamma = gamma.get_value

    def add_replay(self, state, action, r, next_state, finished, e, prev_states, error):
        self.memory.append((state, action, r, next_state, finished, e, prev_states), error=error)

    def train(self):
        if len(self.memory) < self.to_observe:
            return None

        batch_size = min(self.batch_size, len(self.memory))
        # states, actions, rewards, next_states, done = self.memory.random_sample(size=batch_size, depth=self.depth)
        errors = []

        ret = self.memory.random_sample(size=batch_size, depth=1)

        prior = len(ret) == 6

        states, actions, rewards, next_states, done = ret[:5]

        target, target_val = self.model.predict([states, np.ones((len(states), self.action_size), dtype=np.uint)],
                                                [next_states, np.ones((len(states), self.action_size), dtype=np.uint)])

        t = []
        acts = []

        for i in range(batch_size):

            ac = np.zeros(self.action_size)
            ac[actions[i]] = 1
            acts.append(ac)
            com = target[i]

            if done[i]:
                com[actions[i]] = 0
            else:
                v = rewards[i] + self.gamma() * (np.amax(target_val[i]))
                com[actions[i]] = v
            t.append(com)
            errors.append(abs(target[i][actions[i]] - com[actions[i]]))

        t = np.asarray(t)
        acts = np.asarray(acts)\

        h = self.model.fit([states, acts], t, batch=batch_size,
                           epochs=1, verbose=0)

        if prior:
            idx = ret[-1]
            for i in range(len(idx)):
                self.memory.update_tree(idx[i], errors[i])

        return h

    def save(self):
        pt = os.path.join(self.log_dir, str(self.results['last ep']))
        if not os.path.exists(pt):
            os.makedirs(pt)

        try:
            with open(os.path.join(pt, 'results.json'), 'w') as f:
                json.dump(self.results, f)
            with open(os.path.join(pt, 'policy.pol'), 'wb') as f:
                pickle.dump(self.pol, f)
            with open(os.path.join(pt, 'gamma.pol'), 'wb') as f:
                pickle.dump(self.gamma, f)

            self.memory.save(os.path.join(pt, 'memory.mem'))
            self.model.save(pt)

            print('Last calculated reward: {} at episode: {} with mean: {} eps: {} and memory size: {}'.format(
                self.results['rewards'][-1],
                self.results['last ep'],
                np.mean(self.results['rewards']),
                self.pol.get_value(),
                len(self.memory)))
            print('Agent saved')
        except Exception as e:
            print('Cannot save state of the agent: {}'.format(e))

    def load(self, path):
        try:
            print('Loading from path: ' + path)
            with open(os.path.join(path, 'results.json'), 'r') as f:
                self.results = json.load(f)
                for k, v in self.results['info'].items():
                    setattr(self, k, v)

            with open(os.path.join(path, 'policy.pol'), 'rb') as f:
                self.pol = pickle.load(f)

            with open(os.path.join(path, 'gamma.pol'), 'rb') as f:
                self.gamma = pickle.load(f)

            self.memory.load(os.path.join(path, 'memory.mem'))
            self.model.load(path)

            print('Agent correctly loaded')

            print('Last calculated reward: {} at episode: {} with mean: {} eps: {} and memory size: {}'.format(
                self.results['rewards'][-1],
                self.results['last ep'],
                np.mean(self.results['rewards']),
                self.pol.get_value(),
                len(self.memory)))
            return True
        except Exception as e:
            print('Error loading state of agent: {}'.format(e))
            return False

    def act(self, s, no_op):
        s = [s, np.ones((1, self.action_size), dtype=np.uint)]
        act_values = self.model.predict(s)[0]
        action = self.pol.select_action(q_values=act_values)
        if action == 0 and no_op == self.no_op_ep:
            action = np.random.choice(range(1, self.action_size))
        return action


class RamAgent(AbstractAgent):
    def __init__(self, network, max_len_memory=20000, to_observe=5000, pol=None, gamma=None, log_dir='',
                 load_prev=False):

        super(RamAgent, self).__init__(network=network, max_len_memory=max_len_memory, to_observe=to_observe,
                                       pol=pol, gamma=gamma, log_dir=log_dir, load_prev=load_prev,
                                       game='Breakout-ramDeterministic-v4')

    def get_past_states(self, observation, e):
        ln_mem = len(self.memory)-1
        size = self.depth
        if ln_mem == 0:
            r = np.zeros((self.depth, len(observation)))
            r[0] = observation
            return r

        if self.depth > ln_mem:
            size = ln_mem

        r = np.zeros((self.depth, len(observation)))
        r[0] = observation
        for i in range(1, size-1):
            if self.memory[size-i][-1] == e:
                r[i] = self.memory[size-i][0]
        return r

    def train(self):
        if len(self.memory) < self.to_observe:
            return None

        batch_size = min(self.batch_size, len(self.memory))

        ret = self.memory.random_sample(size=batch_size, depth=1)

        prior = len(ret) == 6

        states, actions, rewards, next_states, done = ret[:5]

        target, target_val = self.model.predict([states, np.ones((len(states), self.action_size), dtype=np.uint)],
                                                [next_states, np.ones((len(states), self.action_size), dtype=np.uint)])

        target_val[done] = 0
        targets = rewards + self.gamma() * np.amax(target_val, axis=1)
        actions = np.asarray(actions)
        mask = np.zeros((batch_size, self.action_size))
        mask[np.arange(batch_size), actions] = 1

        targets = mask * targets[:, np.newaxis]

        errors = target - targets
        errors = np.abs(errors[np.arange(batch_size), actions])

        h = self.model.fit([states, mask], targets, batch=batch_size,
                           epochs=1, verbose=0)

        if prior:
            idx = ret[-1]
            for i in range(len(idx)):
                self.memory.update_tree(idx[i], errors[i])

        return h

    def learn(self, episodes, visualize=False, verbose=False, cap=0, backup=10):

        mean_rwd = 0
        means = []
        start = self.results.get('last ep', 0)+1

        for e in tqdm.tqdm(range(start, episodes + 1), initial=start):

            observation = self.env.reset()
            done = False
            i = 0
            no_op = 0
            prev_states = deque(maxlen=self.depth)
            for i in range(self.depth):
                prev_states.append(np.zeros(self.state_size))
            idx = 0

            while not done:

                for _ in range(self.depth):

                    last = np.asarray(prev_states)
                    last = last[np.newaxis, :, :]
                    action = self.act(last, no_op)

                    if action == 0:
                        no_op += 1

                    next_state, reward, done, _ = self.env.step(action)
                    prev_states.appendleft(next_state)

                    reward = np.clip(reward, -1, 1)
                    i += reward
                    self.add_replay(observation, action, reward, next_state, done, e, prev_states, error=reward)
                    observation = next_state

                    idx += 1

                    if done:
                        break

                self.train()

                if visualize:
                    self.env.render()

                if i >= cap > 0:
                    done = True

                if done:
                    mean_rwd += i
                    means.append(mean_rwd / e)

                    rewards = self.results.get('rewards', [])
                    rewards.append(i)

                    nop = self.results.get('no_op', [])
                    nop.append(no_op)

                    self.results.update({'last ep': e, 'rewards': rewards, 'no_op': nop})

                    if verbose:
                        print("episode: {}/{} ({:3f}%), score: {}, no op: {},  memory len: {}, epsilon: {:4f}".format(e,
                                 episodes, 100*(float(e)/episodes), i, no_op, len(self.memory), self.pol.get_value()))
                    if e % backup == 0 and backup > 0:
                        self.save()

        self.env.close()

        mean_rwd /= episodes

        return self.results.get('rewards', [])

