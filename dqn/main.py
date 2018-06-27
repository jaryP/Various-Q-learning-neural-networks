import os
os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32"
os.environ['CPLUS_INCLUDE_PATH'] = '/usr/local/cuda-9.0/include'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
np.random.seed(19)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32"
# os.environ['CPLUS_INCLUDE_PATH'] = '/usr/local/cuda-9.0/include'

import sys; sys.path.append('..')
import policy
from dqn.agent_with_depth_less_memory import ImageAgent as ia_less
from dqn.models_with_depth import DenseDQN, DoubleDQNWrapper, ConvDQM, ConvDDQN


n = ConvDQM(action_size=6, state_size=(84, 84), depth=4, lr=1e-4)

n = DoubleDQNWrapper(n, 10000)

# n = DenseDQN(action_size=3, state_size=6, depth=4, lr=0.001, layer_size=(64, 64))
pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                            value_max=1.0, value_min=0.02, value_test=0.5, nb_steps=100000)


agent = ia_less(pol=pol, network=n, to_observe=10000, max_len_memory=100000,
                log_dir='../pong/good_wrappers_DDQN_32x3-8/', load_prev=True, gamma=0.99)

# agent = ram_less(pol=pol, network=n, to_observe=50000, max_len_memory=1000000,
#                  log_dir='../logs/pong/ram/depth2_huber_DQN/', load_prev=False)

agent.learn(1000,  verbose=True, backup=25, visualize=False)
