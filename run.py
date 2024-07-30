from rl_algorithms.tabular.exploration import SoftMax

import numpy as np
import gymnasium as gym

explorer = SoftMax(epsilon=0.2)
action_space = gym.spaces.Discrete(n=10)
qtable = np.random.randint(0, 10, (10,10))

for _ in range(10):
    print(explorer.choose_action(action_space=action_space, state=4, qtable=qtable))

