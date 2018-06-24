import sys; sys.path.append('..')
import policy
from dqn.agent_with_depth import RamAgent, ImageAgent
from dqn.agent_with_depth_less_memory import ImageAgent as ia_less
from dqn.agent_with_depth_less_memory import RamAgent as ram_less
from dqn.models_with_depth import DenseDQN, DoubleDQNWrapper, ConvDQM, ConvDDQN


# n = ConvDQM(state_size=(84, 84), depth=4, action_size=6, lr=0.00025)
#
# n = ConvDDQN(state_size=(84, 84), depth=4, action_size=4, lr=0.00025)
# n.model.summary()
# exit()

n = DenseDQN(action_size=4, state_size=128, depth=2, lr=0.00025, layer_size=(128, 128, 128, 128))

pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                            value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=1000000)

agent = ram_less(pol=pol, network=n, to_observe=50000, max_len_memory=500000,
                 log_dir='../logs/break/ram/depth2_huber_DQN_128x4/', load_prev=True)

agent.learn(50000,  verbose=True, backup=1000, visualize=False)

exit()

n = DenseDQN(action_size=4, state_size=128, layer_size=(128, 128), lr=0.00025, depth=4)

pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                            value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=1000000)

agent = RamAgent(pol=pol, network=n, to_observe=50000, max_len_memory=1000000,
                 log_dir='../logs/break/conv/prior_replay/', load_prev=True)

agent.learn(5000000, verbose=True, backup=25000)

exit()
n = DenseDQN(action_size=4, state_size=128, layer_size=(256, ), lr=0.00025, depth=4)
n = DoubleDQNWrapper(network=n, update_time=10000)

pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                            value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=1000000)

agent = RamAgent(pol=pol, network=n, to_observe=50000, max_len_memory=1000000,
                 log_dir='../logs/break/double_256_128_prior_replay/', load_prev=True)

agent.learn(100000, verbose=False, backup=250)


