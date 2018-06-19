from collections import deque
from cartpole.abstractnet import DenseDDQN, DuelDenseDQN, DenseDQN
import random
import numpy as np
import gym
import policy
import os
import json
np.random.seed(19)

class Agent:
    def __init__(self, env, max_memory=5000, log_dir='./logs/prova_cartpole', weight_name=None,
                 pol=None, agent_name = 'agent'):

        self.env = env
        #self.env._max_episode_steps = None
        self.name = agent_name

        self.memory = deque(maxlen=max_memory)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.log_dir = os.path.join(log_dir, agent_name)

        self.model = DenseDQN(log_dir=self.log_dir, weight_name=weight_name, action_size=self.action_size,
                                  state_size=self.state_size, lr=0.001, layer_size=(24, 24))

        self.model.model.summary()

        if pol is None:
            self.pol = policy.GreedyPolicy()
        else:
            self.pol = pol

        self.episodes_to_watch = 1000
        self.batch_size = 32
        self.gamma = 0.95

    def add_replay(self, state, action, r, next_state, finished):
        self.memory.append((state, action, r, next_state, finished))

    def train(self, to_update=False):

        if len(self.memory) < self.episodes_to_watch:
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        if isinstance(self.model, DenseDDQN) or isinstance(self.model, DuelDenseDQN):
            target, target_val = self.model.predict(update_input, update_target)
        else:
            target = self.model.predict(update_input)
            target_val = target

        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (
                    np.amax(target_val[i]))

        h = self.model.fit(update_input, target, batch=batch_size,
                           epochs=1, verbose=0)

        #if to_update and isinstance(self.model, DenseDDQN) or isinstance(self.model, DuelDenseDQN):
        #self.model.copy_weights()

        return h

    def act(self, s):

        act_values = self.model.predict(s)[0]
        action = self.pol.select_action(q_values=act_values)

        return action

    def learn(self, episodes, visualize=False, backup=10, verbose=False, cap=0):

        r = {}

        mean_rwd = 0

        for e in range(1, episodes+1):

            er = {}
            observation = self.env.reset()
            observation = np.reshape(observation, [1, self.state_size])
            done = False
            i = 0
            j = 0

            while not done:
                action = self.act(observation)
                next_state, reward, done, _ = self.env.step(action)
                #reward = reward if not done else -10
                # if i+1 % 100 == 0:
                #     reward += 100
                i += reward
                j += 1
                next_state = np.reshape(next_state, [1, self.state_size])
                self.add_replay(observation, action, reward, next_state, done)

                self.train(done)

                observation = next_state

                if visualize:
                    self.env.render()

                if j >= cap > 0:
                    done = True

                if done:
                    mean_rwd += i
                    er.update({'r': i, 'mean': mean_rwd/(e+1)})
                    r.update({e: er})
                    #self.model.learning_rate = max(self.model.learning_rate *0.999995, 0.0001)
                    if verbose:
                        print("episode: {}/{}, score: {}, memory len: {}, epsilon: {}"
                              .format(e, episodes, i, len(self.memory), self.pol.get_value()))
                    done = True
                    # if isinstance(self.model, DDQN):
                    #     self.model.copy_weights()

            if backup != -1 and e % backup == 0:
                self.model.save_weight('{:04d}.h5'.format(int(e / 10)))

        self.env.close()

        mean_rwd /= episodes
        r.update({'mean': mean_rwd})
        with open(os.path.join(self.log_dir,'results.json'), 'w') as fp:
            json.dump(r, fp)

        return r

    def see_learned(self, cap=5000):
        observation = self.env.reset()
        observation = np.reshape(observation, [1, self.state_size])
        i = 0
        for time in range(cap):
            i += 1
            action = self.act(observation)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            observation = next_state
            self.env.render()
            if done:
                break


if __name__ == '__main__':

    eps = 5000
    env = gym.make('CartPole-v0')
    env.seed(19)

    # pol = policy.BoltzmannExplorationPolicy()
    # inner_pol = policy.EpsGreedyPolicy(1.0)
    # lower_prob_pol = policy.BoltzmannPolicy()
    #
    # pol = policy.AnnealedPolicy(inner_policy=inner_pol, attr='eps', value_max=1.0, value_min=0.02, value_test=0.5,
    #                             nb_steps=2000, lower_prob_policy=lower_prob_pol)
    # a = Agent(env=env, log_dir='./logs/prova_pole', pol=pol, agent_name='{}'.format(pol.name))
    #
    # print(a.learn(eps, True, 10, verbose=True)['mean'])
    #
    # exit()
    pol = policy.EpsGreedyPolicy(1.0)
    other_pol = policy.GreedyPolicy()
    pol = policy.AnnealedPolicy(inner_policy=pol, attr='eps', value_max=1.0, value_min=0.1, value_test=0.5,
                                nb_steps=200, lower_prob_policy=other_pol)
    a = Agent(env=env, log_dir='./logs/prova_pole', pol=pol, agent_name='{}'.format(pol.name))
    print(a.learn(eps, True, 10, True)['mean'])
    exit()

    pol = policy.EpsGreedyPolicy(0.1)
    a = Agent(env=env, log_dir='./logs/prova_pole', pol=pol, agent_name='{}'.format(pol.name))
    print(a.learn(eps, False, 10)['mean'])

    pol = policy.GreedyPolicy()
    a = Agent(env=env, log_dir='./logs/prova_pole', pol=pol, agent_name='{}'.format(pol.name))
    print(a.learn(eps, False, 10)['mean'])
