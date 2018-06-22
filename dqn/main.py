import policy
from dqn.agent_with_depth import RamAgent
from dqn.models_with_depth import DenseDQN, DoubleDQNWrapper


n = DenseDQN(action_size=4, state_size=128, layer_size=(256, 128), lr=0.00025, depth=4)

pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                            value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=1000000)

agent = RamAgent(pol=pol, network=n, to_observe=5000, max_len_memory=1000000,
                 log_dir='../logs/break/depth/256_128_prior_replay/', load_prev=True)

agent.learn(500000, verbose=False, backup=250)


n = DenseDQN(action_size=4, state_size=128, layer_size=(256, 128), lr=0.00025, depth=4)
n = DoubleDQNWrapper(network=n, update_time=10000)

pol = policy.AnnealedPolicy(inner_policy=policy.EpsPolicy(1.0, other_pol=policy.GreedyPolicy()), attr='eps',
                            value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=1000000)

agent = RamAgent(pol=pol, network=n, to_observe=50000, max_len_memory=1000000,
                 log_dir='../logs/break/depth/double_256_128_prior_replay/', load_prev=True)

agent.learn(500000, verbose=False, backup=250)


