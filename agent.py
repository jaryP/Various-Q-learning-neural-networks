from abc import ABC, abstractmethod
import time

class AbstractAgent(ABC):

    def __init__(self, net_name):
        pass


import gym


env = gym.make('Centipede-v0')

obs = env.reset()

print(obs.shape)


env.render()

is_done = False
while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  act = env.action_space.sample()
  frame, reward, is_done, _ = env.step(act)
  # print(env.env.get_action_meanings()[act])
  print(reward)
  # Render
  env.render()
  time.sleep(0.2)
env.close()