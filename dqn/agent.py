import tqdm
import numpy as np
np.random.seed(19)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['KERAS_BACKEND'] = 'tensorflow'

from dqn.memory_with_depth import SimpleMemory
import gym
import policy
import traceback
import pickle
import time
import matplotlib.pyplot as plt
from dqn.wrappers import wrap_dqn
from keras.layers import Dense, Lambda, Add, Multiply, Flatten, Subtract
from keras.layers.convolutional import  Conv2D
from keras.models import load_model, Model, Input, clone_model
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import time


def huber_loss(y_true, y_pred, clip_value=1.0):

    x = y_true - y_pred
    if np.isinf(clip_value):
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    import tensorflow as tf
    if hasattr(tf, 'select'):
        ret = tf.select(condition, squared_loss, linear_loss)
    else:
        ret = tf.where(condition, squared_loss, linear_loss)

    return ret


class Agent:
    def __init__(self,  config=None):

        if config is None:
            config = {}
        self.env = wrap_dqn(gym.make(config.get('game', 'PongNoFrameskip-v4')))
        self.action_size = self.env.action_space.n

        self.to_vis = config.get('visualize', False)
        self.verbose = config.get('verbose', True)
        self.backup = config.get('backup', 25)
        self.episodes = config.get('episodes', 300)

        self.depth = config.get('depth', 4)
        self.state_size = config.get('space', (84, 84))
        self.model = None
        self._target_model = None

        self.prioritized = config.get(('prioritized', False))
        self.memory = SimpleMemory(max_len=config.get('mem_size', 100000))

        if config.get('duel', False):
            self.model = self._duel_conv()
        else:
            self.model = self._conv()

        self.model.compile(Adam(lr=config.get('lr', 1e-4)), loss=huber_loss)

        if config.get('target', True):
            self._target_model = clone_model(self.model)
            self._target_model.set_weights(self.model.get_weights())
            self._time = 0
            self.update_time = config.get('target_update', 1000)

        self.env._max_episode_steps = None
        self.batch_size = config.get('batch', 32*3)
        self.to_observe = config.get('to_observe', 10000)
        self.log_dir = config['log_dir']

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        attr = {'batch size': self.batch_size, 'to observe': self.to_observe,
                'depth': self.depth}

        self.results = {'info': attr}


        load_prev = config.get('load', False)

        self.gamma = None
        pol = None

        if 'pol' in config:
            if config['pol'] == 'random':
                pol = policy.RandomPolicy()
            elif config['pol'] == 'eps':
                pol = policy.EpsPolicy(config.get('pol_eps', 0.1))

        self.pol = pol

        if load_prev:
            path = sorted([int(x) for x in os.listdir(self.log_dir) if os.path.isdir(
                os.path.join(self.log_dir, x))])
            if len(path) != 0:
                load_prev = self.load(os.path.join(self.log_dir, str(path[-1])))

        if self.pol is None:
            self.pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()),
                                             attr='eps', value_max=1.0, value_min=config.get('ex_min', 0.02),
                                             value_test=0.5, nb_steps=config.get('ex_steps', 100000))
        if self.gamma is None:
            self.gamma = policy.EpsPolicy(float(config.get('gamma', 0.99))).get_value

    def add_replay(self, prev_states, action, r, next_state, finished, error):
        self.memory.append((prev_states, action, r, next_state, finished), error=error)

    def get_batch(self, size):
        ret = self.memory.random_sample(size=size, depth=self.depth)
        return ret

    def train(self):

        batch_size = min(self.batch_size, len(self.memory))

        ret = self.get_batch(batch_size)

        states, actions, rewards, next_states, done = ret[:5]
        mask = np.ones((batch_size, self.action_size), dtype=np.uint)

        target, target_val = self.predict([states, mask], [next_states, mask])

        targets = rewards + self.gamma() * np.amax(target_val, axis=1)
        actions = np.asarray(actions)
        mask = np.zeros((batch_size, self.action_size))
        mask[np.arange(batch_size), actions] = 1

        targets = mask * targets[:, np.newaxis]
        targets[done, actions[done]] = rewards[done]

        h = self.fit([states, mask], targets, batch=8, epochs=1, verbose=0)

        if len(ret) == 6:

            errors = target_val - targets
            errors = np.abs(errors[np.arange(batch_size), actions])
            idx = ret[-1]
            for i in range(len(idx)):
                self.memory.update_tree(idx[i], errors[i])

        return h

    def learn(self):

        self.env.seed(19)

        starting_eps = self.results.get('last ep', 0) + 1
        frames = self.results.get('frames', [0])[-1]

        for ep in range(starting_eps, self.episodes+1):

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

                if self.to_vis:
                    self.env.render()

                if len(self.memory) >= self.to_observe:
                    self.train()

                if done:
                    self.update_results(r=R, qvals=eps_q_vals, e=ep, frames=frames)
                    if self.backup > 0 and ep % self.backup == 0:
                        self.save()
                    if self.verbose:
                        print("episode: {}/{} ({}%), score: {} (last 50 games r average: {}), frame number: {}"
                              ",  memory len: {}, epsilon: {}".format(ep, self.episodes,
                                                                      round(100 * (float(ep) / self.episodes), 4), R,
                                                                      round(np.mean(self.results['rewards'][-50:]), 3),
                                                                      frames, len(self.memory),
                                                                      round(self.pol.get_value(), 2)))
                        print('Elapsed time: ', time.time()-start)

        self.env.close()
        if self.backup > 0:
            self.save()

        return self.results

    def act(self, s):
        s = [s, np.ones((1, self.action_size))]
        act_values = self.model.predict(s)[0]
        action = self.pol.select_action(q_values=act_values)
        return action, [float(q) for q in act_values]

    ##################################################
    # MODELS
    ##################################################

    def predict(self, x, x_t=None, batch=None, verbose=0):
        online = self.model.predict(x, batch_size=batch, verbose=verbose)
        if x_t is None:
            return online
        elif self._target_model is None:
            target = self.model.predict(x, batch_size=batch, verbose=verbose)
        else:
            target = self._target_model.predict(x_t, batch_size=batch, verbose=verbose)
        return online, target

    def fit(self, x, y, batch=None, epochs=1, verbose=0):
        h = self.model.fit(x, y, batch_size=batch, epochs=epochs, verbose=verbose)
        if self._target_model is not None:
            self._time += 1
            if self._time >= self.update_time:
                self._target_model.set_weights(self.model.get_weights())
                self._time = 0
        return h

    def _conv(self):
        in_shape = (self.depth, self.state_size[0], self.state_size[1], )

        frames_input = Input(shape=in_shape, name='frames')
        mask = Input(shape=(self.action_size,), name='mask')

        normz = Lambda(lambda x: x / 255.0, output_shape=in_shape)(frames_input)

        conv = Conv2D(kernel_size=8, filters=16, strides=4, activation='relu',
                      data_format='channels_first')(normz)
        conv = Conv2D(kernel_size=4, filters=32, strides=2, activation='relu',
                      data_format='channels_first')(conv)
        flt = Flatten(data_format='channels_first')(conv)

        flt = Dense(256, activation='relu', name='last_layer')(flt)
        flt = Dense(self.action_size)(flt)

        flt = Multiply()([flt, mask])

        model = Model(input=[frames_input, mask], output=flt)
        return model

    def _duel_conv(self):
        in_shape = (self.depth, self.state_size[0], self.state_size[1],)

        frames_input = Input(shape=in_shape, name='frames')
        mask = Input(shape=(self.action_size,), name='mask')

        normz = Lambda(lambda x: x / 255.0, output_shape=in_shape)(frames_input)

        conv = Conv2D(kernel_size=8, filters=16, strides=4, activation='relu',
                      data_format='channels_first')(normz)
        conv = Conv2D(kernel_size=4, filters=32, strides=2, activation='relu',
                      data_format='channels_first')(conv)
        flt = Flatten(data_format='channels_first')(conv)

        advantage_flt = Dense(256, activation='relu', name='last_layer')(flt)

        flt = Dense(self.action_size)(flt)
        advantage = Dense(self.action_size, name='advantege_layer')(advantage_flt)

        advantage_sub = Lambda(function=lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
        advantage = Subtract()([advantage, advantage_sub])

        value_flt = Dense(128, activation='relu', )(flt)
        value = Dense(1, name='value_layer')(value_flt)

        value = Lambda(function=lambda x: K.repeat_elements(x, self.action_size, axis=1))(value)

        out = Add()([advantage, value])

        flt = Multiply()([out, mask])

        model = Model(input=[frames_input, mask], output=flt)
        return model

    ##################################################
    # UTILS
    ##################################################

    def plot(self):
        rewards = self.results.get('rewards', [])
        qs = self.results.get('q_values', [])

        rewards_mean = [np.mean(rewards[max(0, i-50):i]) for i in range(1, len(rewards)+1)]
        rewards_mean_100 = [np.mean(rewards[max(0, i-100):i]) for i in range(1, len(rewards)+1)]

        q_values = np.asarray([q[0][1] for q in qs])

        actions = {}
        for i, action in enumerate(self.env.unwrapped.get_action_meanings()):
            if action not in ['NOOP', 'LEFT', 'RIGHT']:
                continue
            actions.update({action: q_values[:, i]})

        eps = range(len(rewards))

        plt.plot(eps, rewards, label='reward')
        plt.plot(eps, rewards_mean, label='mean reward in last 50 games')
        plt.plot(eps, rewards_mean_100, label='mean reward in last 100 games')

        plt.plot(eps, [0]*len(eps))
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.show()

        # for a, v in actions.items():
        #     plt.plot(eps, v, label=a)
        plt.plot(eps, np.max(q_values, axis=1), label='Q values')
        plt.plot(eps, [0]*len(eps))
        plt.xlabel('episodes')
        plt.ylabel('q value')
        plt.legend()
        plt.show()
        print()

    def play(self):

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)

        done = False
        m = clone_model(self.model)
        m.layers.pop()
        m.layers.pop()
        m.layers.pop()
        m.summary()
        r = 0
        from gym.wrappers.monitoring.video_recorder import VideoRecorder
        rec = VideoRecorder(self.env, base_path=os.path.join(self.log_dir, self.log_dir.rsplit('/', 1)[1]))

        for i in range(5):
            observation = self.env.reset()
            r = 0
            done = False
            while not done:
                action, q_vals = self.act(np.asarray(observation)[np.newaxis])

                next_state, reward, done, _ = self.env.step(action)

                r += reward

                observation = next_state

                self.env.render()
                rec.capture_frame()
                time.sleep(0.05)
            print(r)

        print('Game ended with score: ', r)
        self.env.close()
        rec.close()

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
            self.savel_models(pt)

            print('Last calculated reward: {} at episode: {} with mean in last 50 episodes: {:3f} eps: {:3f} and memory size: {}'.format(
                self.results['rewards'][-1],
                self.results['last ep'],
                np.mean(self.results['rewards'][-50:]),
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

            self.load_models(path)

            print('Agent correctly loaded')

            print('Last calculated reward: {} at episode: {} with mean in last 50 episodes: {:3f} eps: '
                  '{:3f} and memory size: {}'.format(self.results['rewards'][-1],
                    self.results['last ep'],
                    np.mean(self.results['rewards'][-50:]),
                    self.pol.get_value(),
                    len(self.memory)))
            return True
        except Exception as e:
            print('Error loading state of agent: {}'.format(traceback.print_exc()))
            return False

    def update_results(self, r, qvals, e, frames):

        rewards = self.results.get('rewards', [])
        rewards.append(float(r))

        qs = self.results.get('q_values', [])
        qs.append(qvals)

        fs = self.results.get('frames', [])
        fs.append(frames)

        self.results.update({'last ep': e, 'rewards': rewards, 'q_values': qs, 'frames': fs})

    def savel_models(self, path):
        self.model.save(os.path.join(path, 'model.h5'))
        self.model.save_weights(os.path.join(path, 'model_weights.h5'))
        # self._target_net.save(os.path.join(path, 'target_model.h5'))
        if self._target_model is not None:
            self._target_model.save_weights(os.path.join(path, 'target_model_weights.h5'))

    def load_models(self, path):
        self.model.load_weights(os.path.join(path, 'model_weights.h5'))
        self.model = load_model(os.path.join(path, 'model.h5'), custom_objects={'huber_loss': huber_loss})
        if self._target_model is not None:
            self._target_model.load_weights(os.path.join(path, 'target_model_weights.h5'))


if __name__ == '__main__':

    conf = {
        'log_dir': '/media/jary/DATA/Uni/Ai/Iocchi/pong/DDQN',
        'visualize': True,
        'load': True,
        'episodes': 300,
    }

    a = Agent(conf)
    a.plot()
    # a.learn()
    #
    # conf = {
    #     'log_dir': '/media/jary/DATA/Uni/Ai/Iocchi/pong/duel_DDQN',
    #     'visualize': False,
    #     'load': True,
    #     'episodes': 300,
    # }
    #
    # a = Agent(conf)
    # a.learn()


