import tqdm
import os
os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32"
os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
os.environ['CPLUS_INCLUDE_PATH'] = '/usr/local/cuda-9/include'
from abc import ABC, abstractmethod
from dqn.memory_with_depth import PrioritizedMemory
import gym
import policy
import traceback
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import json
import pickle
from random import randint
from collections import deque
from dqn.memory_with_depth_less_memory import PrioritizedMemory as pr_less
from dqn.memory_with_depth_less_memory import SimpleMemory
np.random.seed(19)


class AbstractAgent(ABC):
    def __init__(self,  network, max_len_memory, to_observe, pol, gamma, log_dir, load_prev, game):

        self.env = gym.make(game)
        self.env.seed(19)
        self.action_meaning = self.env.env.get_action_meanings()
        self.no_op_ep = 30
        self.env._max_episode_steps = None
        self.model = network
        network.model.summary()
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

        self.memory = SimpleMemory(max_len=max_len_memory)

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

    def add_replay(self, prev_states, action, r, next_state, finished, error):
        self.memory.append((prev_states, action, r, next_state, finished), error=error)

    def train(self):

        if len(self.memory) < self.to_observe:
            return None

        batch_size = min(self.batch_size, len(self.memory))

        ret = self.memory.random_sample(size=batch_size, depth=self.depth)

        states, actions, rewards, next_states, done = ret[:5]
        mask = np.ones((batch_size, self.action_size), dtype=np.uint)

        target, target_val = self.model.predict([states, mask], [next_states, mask])

        # target_val[done] = rewards[done]

        for i, d in enumerate(done):
            if d:
                target_val[i] = rewards[i]

        targets = rewards + self.gamma() * np.amax(target_val, axis=1)
        actions = np.asarray(actions)
        mask = np.zeros((batch_size, self.action_size))
        mask[np.arange(batch_size), actions] = 1

        targets = mask * targets[:, np.newaxis]

        h = self.model.fit([states, mask], targets, batch=batch_size, epochs=1, verbose=0)

        if len(ret) == 6:

            errors = target - targets
            errors = np.abs(errors[np.arange(batch_size), actions])

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
                pickle.dump(self.pol, f,  protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(pt, 'gamma.pol'), 'wb') as f:
                pickle.dump(self.gamma, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.memory.save(pt)
            self.model.save(pt)

            print('Last calculated reward: {} at episode: {} with mean: {:3f} eps: {:3f} and memory size: {}'.format(
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

            self.memory.load(path)
            self.model.load(path)

            print('Agent correctly loaded')

            print('Last calculated reward: {} at episode: {} with mean: {:3f} eps: {:3f} and memory size: {}'.format(
                self.results['rewards'][-1],
                self.results['last ep'],
                np.mean(self.results['rewards']),
                self.pol.get_value(),
                len(self.memory)))
            return True
        except Exception as e:
            print('Error loading state of agent: {}'.format(traceback.print_exc()))
            return False

    def act(self, s, no_op):
        s = [s, np.ones((1, self.action_size), dtype=np.uint)]
        act_values = self.model.predict(s)[0]
        action = self.pol.select_action(q_values=act_values)
        # if action == 0 and no_op == self.no_op_ep:
        #     while action == 0:
        #         action = np.random.choice(range(1, self.action_size))
        return int(action), [float(q) for q in act_values]

    def update_results(self, r, noop, qvals, f):

        rewards = self.results.get('rewards', [])
        rewards.append(r)

        nop = self.results.get('no_op', [])
        nop.append(noop)

        qs = self.results.get('q_values', [])
        qs.append(qvals)

        self.results.update({'last ep': f, 'rewards': rewards, 'no_op': nop, 'q_values': qs})


class RamAgent(AbstractAgent):
    def __init__(self, network, max_len_memory=20000, to_observe=5000, pol=None, gamma=None, log_dir='',
                 load_prev=False):

        super(RamAgent, self).__init__(network=network, max_len_memory=max_len_memory, to_observe=to_observe,
                                       pol=pol, gamma=gamma, log_dir=log_dir, load_prev=load_prev,
                                       game='Breakout-ramDeterministic-v4')

    def learn(self, episodes, visualize=False, verbose=False, cap=0, backup=10):

        starting_eps = self.results.get('last ep', 0)+1

        for ep in range(starting_eps, episodes):

            observation = self.env.reset()

            for _ in range(randint(1, self.no_op_ep)):
                observation, _, _, _ = self.env.step(0)

            done = False
            no_op = 0
            prev_states = deque(maxlen=self.depth)

            for i in range(self.depth):
                prev_states.append(np.zeros(self.state_size))
            prev_states.appendleft(observation)

            R = 0

            eps_q_vals = []

            while not done:

                for _ in range(self.depth):

                    last = np.asarray(prev_states)
                    last = last[np.newaxis, :, :]
                    action, q_vals = self.act(last, no_op)

                    if action == 0:
                        no_op += 1
                    # if frames >= self.to_observe*2:
                    #     self.no_op_ep = -1
                    eps_q_vals.append((action, q_vals))

                    next_state, reward, done, _ = self.env. step(action)
                    prev_states.appendleft(next_state)

                    reward = np.clip(reward, -1, 1)

                    R += reward
                    self.add_replay(observation, action, reward, next_state, done, error=reward)

                    observation = next_state
                    if done:
                        break

                self.train()

                if visualize:
                    self.env.render()

                # if i >= cap > 0:
                #     done = True

                if done:
                    self.update_results(r=R, noop=no_op, qvals=eps_q_vals, f=ep)
                    if backup > 0 and ep % backup == 0:
                        self.save()
                    if verbose:
                        print("episode: {}/{} ({}%), score: {} (last 100 games r average: {}), no op: {}"
                              ",  memory len: {}, epsilon: {}".format(ep, episodes,
                                                                      round(100 * (float(ep) / episodes), 4), R,
                                                                      round(np.mean(self.results['rewards'][-100:]), 3),
                                                                      no_op, len(self.memory),
                                                                      round(self.pol.get_value(), 2)))

        self.env.close()

        if backup > 0:
            self.update_results(r=R, noop=no_op, qvals=eps_q_vals, f=ep)
            self.save()

        return self.results.get('rewards', [])


class ImageAgent(AbstractAgent):
    def __init__(self, network, max_len_memory=20000, to_observe=5000, pol=None, gamma=None, log_dir='',
                 load_prev=False):

        super(ImageAgent, self).__init__(network=network, max_len_memory=max_len_memory, to_observe=to_observe,
                                         pol=pol, gamma=gamma, log_dir=log_dir, load_prev=load_prev,
                                         game='Breakout-Deterministic-v4')

    def pre_processing(self, s):
        processed_observe = np.uint8(
            resize(rgb2gray(s), (self.state_size[0], self.state_size[0]), mode='constant') * 255)
        return processed_observe

    def learn(self, episodes, visualize=False, verbose=False, cap=0, backup=10):

        frames = self.results.get('last ep', 0)+1

        while frames < episodes:

            observation = self.env.reset()
            observation = self.pre_processing(observation)
            done = False
            i = 0
            no_op = 0
            prev_states = deque(maxlen=self.depth)

            for i in range(self.depth):
                prev_states.append(np.zeros(self.state_size))
            prev_states.appendleft(observation)

            while not done:

                eps_q_vals = []

                for _ in range(self.depth):
                    last = np.asarray(prev_states)
                    last = last[np.newaxis, :, :]
                    action, q_vals = self.act(last, no_op)
                    eps_q_vals.append((action, q_vals))

                    if action == 0:
                        no_op += 1

                    next_state, reward, done, _ = self.env. step(action)
                    next_state = self.pre_processing(next_state)
                    prev_states.appendleft(next_state)

                    reward = np.clip(reward, -1, 1)

                    i += reward
                    self.add_replay(observation, action, reward, next_state, done, error=reward)

                    observation = next_state
                    if done:
                        break

                self.train()

                if backup > 0 and frames % backup == 0:
                    self.upadate_results(r=i, noop=no_op, qvals=eps_q_vals, f=frames)
                    self.save()

                frames += 1

                if visualize:
                    self.env.render()

                if i >= cap > 0:
                    done = True

                if done or frames == episodes-1:
                    self.upadate_results(r=i, noop=no_op, qvals=eps_q_vals, f=frames)
                    if verbose == True:
                        print("episode: {}/{} ({}%), score: {} (last 100 games r average: {}), no op: {}"
                              ",  memory len: {}, epsilon: {}".format(frames, episodes,
                                                                      round(100 * (float(frames) / episodes), 4), i,
                                                                      round(np.mean(self.results['rewards'][-100:]), 3),
                                                                      no_op, len(self.memory),
                                                                      round(self.pol.get_value(), 2)))

        self.env.close()
        if backup > 0:
            self.save()

        return self.results.get('rewards', [])
