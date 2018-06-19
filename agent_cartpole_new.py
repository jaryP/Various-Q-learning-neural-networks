from collections import deque
import models
import random
import numpy as np
import gym
import policy
import os
import tqdm
np.random.seed(19)

class Agent:
    def __init__(self, game, net, max_memory=5000, log_dir='./logs/prova_cartpole', weight_name=None,
                 pol=None, agent_name='agent'):

        self.env = game
        self.env._max_episode_steps = 500
        self.name = agent_name

        self.model = net
        self.memory = deque(maxlen=max_memory)
        self.state_size = net.state_size
        self.action_size = net.action_size
        self.log_dir = os.path.join(log_dir, agent_name)

        #self.model.model.summary()

        if pol is None:
            self.pol = policy.GreedyPolicy()
        else:
            self.pol = pol

        self.episodes_to_watch = 32
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

        if isinstance(self.model, models.DoubleDQNWrapper):
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

        h = self.model.fit(update_input, target, batch=1,
                           epochs=1, verbose=0)

        #if to_update and isinstance(self.model, DenseDDQN) or isinstance(self.model, DuelDenseDQN):
        #self.model.copy_weights()

        return h

    def replay(self):
        batch_size = self.batch_size
        if batch_size > len(self.memory):
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.empty((batch_size, self.state_size))
        targets = np.empty((batch_size, self.action_size))
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            #
            # if isinstance(self.model, models.DoubleDQNWrapper):
            #     target_f, target_val = self.model.predict(state, update_target)
            # else:
            #     target_f = self.model.predict(update_input)
            #     target_val = target
            #t = self.model.predict(state, next_state)
            states[i] = state
            target_f = self.model.predict(state)
            n = self.model.predict(next_state)
            if isinstance(self.model, models.DoubleDQNWrapper):
                target_f, n = self.model.predict(state, next_state)

            if not done:
                target = (reward + self.gamma *
                          np.amax(n[0]))
            target_f[0][action] = target
            targets[i] = target_f

        self.model.fit(states, targets, epochs=1, verbose=0, batch=1)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #

    def act(self, s):

        act_values = self.model.predict(s)[0]
        action = self.pol.select_action(q_values=act_values)

        return action

    def learn(self, episodes, visualize=False, backup=10, verbose=False, cap=0):

        r = {}

        mean_rwd = 0
        means = []

        for e in tqdm.tqdm(range(1, episodes+1)):

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

                # self.train(done)
                self.replay()

                observation = next_state

                if visualize:
                    self.env.render()

                if j >= cap > 0:
                    done = True

                if done:
                    mean_rwd += i
                    means.append(mean_rwd/e)
                    er.update({'r': i, 'mean': mean_rwd/(e+1)})
                    r.update({e: er})
                    #self.model.learning_rate = max(self.model.learning_rate *0.999995, 0.0001)
                    if verbose:
                        print("episode: {}/{}, score: {}, memory len: {}, epsilon: {}"
                              .format(e, episodes, i, len(self.memory), self.pol.get_value()))
                    done = True
                    # if isinstance(self.model, DDQN):
                    #     self.model.copy_weights()

            # if backup != -1 and e % backup == 0:
            #     self.model.save_weight('{:04d}.h5'.format(int(e / 10)))

        self.env.close()

        mean_rwd /= episodes
        r.update({'mean': mean_rwd})
        # with open(os.path.join(self.log_dir,'results.json'), 'w') as fp:
        #     json.dump(r, fp)

        return means

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


def exp_duel():
    import matplotlib.pyplot as plt
    eps = 1000

    env = gym.make('CartPole-v0')
    env.seed(19)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500)

    log_dir = './logs/prova_pole' + pol.name

    net = models.DenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                          lr=0.001)

    a = Agent(game=env, net=net, log_dir=log_dir, pol=pol)

    r = a.learn(eps, False, 10, verbose=False)
    print(r[-1])
    plt.plot(range(eps), r, label='DQN')

    pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500)

    net = models.DuelDenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                              lr=0.001, layer_size_val=(12, 12))
    env.seed(19)
    a = Agent(game=env, net=net, log_dir=log_dir, pol=pol)

    r = a.learn(eps, False, 10, verbose=False)
    print(r[-1])
    plt.plot(range(eps), r, label='Duel DQN 12 12')

    pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500)

    net = models.DuelDenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                              lr=0.001, layer_size_val=(8, 8))
    env.seed(19)
    a = Agent(game=env, net=net, log_dir=log_dir, pol=pol)

    r = a.learn(eps, False, 10, verbose=False)
    print(r[-1])
    plt.plot(range(eps), r, label='Duel DQN 8 8')

    pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500)

    net = models.DuelDenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                              lr=0.001, layer_size_val=(4, 4))

    a = Agent(game=env, net=net, log_dir=log_dir, pol=pol)

    r = a.learn(eps, False, 10, verbose=False)
    plt.plot(range(eps), r, label='Duel DQN 4 4')
    print(r[-1])

    pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500)

    net = models.DuelDenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                              lr=0.001, layer_size_val=(24, 24))

    a = Agent(game=env, net=net, log_dir=log_dir, pol=pol)

    r = a.learn(eps, False, 10, verbose=False)
    plt.plot(range(eps), r, label='Duel DQN 24 24')
    print(r[-1])
    plt.legend()
    plt.savefig('exp_duel.png')


def exp_double_duel():
    import matplotlib.pyplot as plt
    eps = 1000

    env = gym.make('CartPole-v0')
    env.seed(19)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n


    log_dir = './logs/prova_pole'

    net = models.DenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                          lr=0.001)

    a = Agent(game=env, net=net, log_dir=log_dir, pol=policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500))

    r = a.learn(eps, False, 10, verbose=False)
    plt.plot(range(eps), r, label='DQN')

    net = models.DuelDenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                              lr=0.001, layer_size_val=(4, 4))
    env.seed(19)
    a = Agent(game=env, net=net, log_dir=log_dir, pol=policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500))

    r = a.learn(eps, False, 10, verbose=False)
    plt.plot(range(eps), r, label='Duel DQN 4 4')

    for i in [50, 100, 200, 300, 500, 750, 1000, 2000, 3000]:

        net = models.DuelDenseDQN(log_dir=log_dir, action_size=action_size, state_size=state_size, layer_size=(24, 24),
                                  lr=0.001, layer_size_val=(4, 4))
        n = models.DoubleDQNWrapper(network=net, update_time=i)

        a = Agent(game=env, net=n, log_dir=log_dir, pol=policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500))

        r = a.learn(eps, False, 10, verbose=False)

        plt.plot(range(eps), r, label='Double Duel DQN 4 4 '+str(i))

    plt.legend()
    plt.savefig('exp_double_duel.png')

def exp_ddqn():
    import matplotlib.pyplot as plt
    eps = 1000

    env = gym.make('CartPole-v0')
    env.seed(19)

    pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500)

    log_dir = './logs/prova_pole'+pol.name

    n = models.DenseDQN(log_dir=log_dir, action_size=env.action_space.n, state_size=env.observation_space.shape[0],
                        layer_size=(24, 24), lr=0.001)

    a = Agent(game=env, net=n, log_dir=log_dir, pol=pol)

    r = a.learn(eps, False, 10, verbose=False)
    plt.plot(range(eps), r, label='DQN')

    for i in [50, 100, 200, 300, 500, 750, 1000, 2000, 3000]:
        pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                                    value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=500)

        log_dir = './logs/prova_pole'+pol.name

        n = models.DoubleDQNWrapper(network=models.DenseDQN(log_dir=log_dir, action_size=env.action_space.n,
                                    state_size=env.observation_space.shape[0], layer_size=(24, 24), lr=0.001),
                                    update_time=i)

        a = Agent(game=env, net=n, log_dir=log_dir, pol=pol)

        r = a.learn(eps, False, 10, verbose=False)

        plt.plot(range(eps), r, label='Update time: {}'.format(i))

    plt.legend()
    plt.savefig('exp_ddqn.png')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.pyplot  import ion
    exp_duel()
    exp_double_duel()
    exp_ddqn()