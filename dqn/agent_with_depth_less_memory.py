import tqdm
import numpy as np
np.random.seed(19)
import os
from abc import ABC, abstractmethod
from dqn.memory_with_depth import SimpleMemory
import gym
import policy
import traceback
from skimage.color import rgb2gray
from skimage.transform import resize
import json
import pickle
from random import randint
from collections import deque
# from dqn.memory_with_depth_less_memory import SimpleMemory
import time
import multiprocessing as mp
from dqn.wrappers import wrap_dqn


class AbstractAgent(ABC):
    def __init__(self,  network, max_len_memory, to_observe, pol, gamma, log_dir, load_prev, game):

        self.env = wrap_dqn(gym.make(game))
        self.env.seed(19)
        # self.action_meaning = self.env.env.get_action_meanings()
        self.env._max_episode_steps = None
        self.model = network
        network.model.summary()
        self.batch_size = 32*3
        self.to_observe = to_observe
        self.state_size = network.state_size
        self.action_size = network.action_size
        self.log_dir = log_dir
        self.depth = network.depth
        # self.lives = self.env.env.ale.lives()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        attr = {'batch size': self.batch_size, 'to observe': self.to_observe,
                'depth': self.depth}

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
                gamma = policy.EpsPolicy(0.95)
            elif isinstance(gamma, float):
                gamma = policy.EpsPolicy(gamma)

            if isinstance(gamma, policy.AnnealedPolicy):
                self.gamma = gamma.linear_step
            elif isinstance(gamma, policy.Policy):
                self.gamma = gamma.get_value

    def add_replay(self, prev_states, action, r, next_state, finished, error):
        self.memory.append((prev_states, action, r, next_state, finished), error=error)

    def train(self):

        batch_size = min(self.batch_size, len(self.memory))

        ret = self.memory.random_sample(size=batch_size, depth=self.depth)

        states, actions, rewards, next_states, done = ret[:5]
        mask = np.ones((batch_size, self.action_size), dtype=np.uint)

        target, target_val = self.model.predict([states, mask], [next_states, mask])

        targets = rewards + self.gamma() * np.amax(target_val, axis=1)
        actions = np.asarray(actions)
        mask = np.zeros((batch_size, self.action_size))
        mask[np.arange(batch_size), actions] = 1

        targets = mask * targets[:, np.newaxis]
        targets[done, actions[done]] = rewards[done]

        h = self.model.fit([states, mask], targets, batch=8, epochs=1, verbose=0)

        if len(ret) == 6:

            errors = target_val - targets
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
            with open(os.path.join(pt, 'results.json'), 'wb') as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(pt, 'policy.pol'), 'wb') as f:
                pickle.dump(self.pol, f,  protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(pt, 'gamma.pol'), 'wb') as f:
                pickle.dump(self.gamma, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(pt, 'memory.mem'), 'wb') as f:
                pickle.dump(self.memory, f, protocol=pickle.HIGHEST_PROTOCOL)

            # self.memory.save(pt)
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
            with open(os.path.join(path, 'results.json'), 'rb') as f:
                self.results = pickle.load(f)
                for k, v in self.results['info'].items():
                    setattr(self, k, v)

            with open(os.path.join(path, 'policy.pol'), 'rb') as f:
                self.pol = pickle.load(f)

            with open(os.path.join(path, 'gamma.pol'), 'rb') as f:
                self.gamma = pickle.load(f)

            with open(os.path.join(path, 'memory.mem'), 'rb') as f:
                self.memory = pickle.load(f)

            # self.memory.load(path)
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

    def act(self, s):
        s = [s, np.ones((1, self.action_size))]
        act_values = self.model.predict(s)[0]
        action = self.pol.select_action(q_values=act_values)
        return action, [float(q) for q in act_values]

    def update_results(self, r, qvals, e, frames):

        rewards = self.results.get('rewards', [])
        rewards.append(float(r))

        qs = self.results.get('q_values', [])
        qs.append(qvals)

        fs = self.results.get('frames', [])
        fs.append(frames)

        self.results.update({'last ep': e, 'rewards': rewards, 'q_values': qs, 'frames': fs})


class ImageAgent(AbstractAgent):
    def __init__(self, network, max_len_memory=20000, to_observe=5000, pol=None, gamma=None, log_dir='',
                 load_prev=False):

        super(ImageAgent, self).__init__(network=network, max_len_memory=max_len_memory, to_observe=to_observe,
                                         pol=pol, gamma=gamma, log_dir=log_dir, load_prev=load_prev,
                                         game='PongNoFrameskip-v4')

    def learn(self, episodes, visualize=False, verbose=False, cap=0, backup=10):

        starting_eps = self.results.get('last ep', 0) + 1
        frames = self.results.get('frames', [0])[-1]

        for ep in range(starting_eps, episodes):

            observation = self.env.reset()
            done = False

            R = 0
            start = time.time()

            while not done:

                eps_q_vals = []
                action, q_vals = self.act(np.asarray(observation)[np.newaxis])
                eps_q_vals.append((action, q_vals))

                frames += 1

                next_state, reward, done, _ = self.env. step(action)

                R += reward

                self.add_replay(observation, action, reward, next_state, done, error=reward)
                observation = next_state

                if visualize:
                    self.env.render()

                if len(self.memory) >= self.to_observe:
                    self.train()

                if done:
                    self.update_results(r=R, qvals=eps_q_vals, e=ep, frames=frames)
                    if backup > 0 and ep % backup == 0:
                        self.save()
                    if verbose:
                        print("episode: {}/{} ({}%), score: {} (last 100 games r average: {}), frame number: {}"
                              ",  memory len: {}, epsilon: {}".format(ep, episodes,
                                                                      round(100 * (float(ep) / episodes), 4), R,
                                                                      round(np.mean(self.results['rewards'][-100:]), 3),
                                                                      frames, len(self.memory),
                                                                      round(self.pol.get_value(), 2)))
                        print('Elapsed time: ', time.time()-start)

        self.env.close()
        if backup > 0:
            self.save()

        return self.results
