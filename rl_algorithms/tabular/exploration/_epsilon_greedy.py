from ._greedy import Greedy

import numpy as np
import gymnasium as gym

class EpsilonGreedy(Greedy):
    def __init__(self, epsilon, epsilon_decay_rate=0.0, seed=None):
        super(EpsilonGreedy, self).__init__(seed=seed)
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        

    def choose_action(self, action_space:gym.spaces.Discrete, state: int, qtable: np.ndarray, mask: np.ndarray | None=None)  -> int:
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explore_exploit_tradeoff = self.rng.uniform(0, 1)

        if mask is None:
            mask = np.ones_like(qtable)

        # Exploration
        if explore_exploit_tradeoff < self.epsilon:
            action = action_space.sample(mask[state, :])

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            action = self.argmax(qtable[state, :], mask[state, :])
            
        return int(action)


    def decay_epsilon(self, episode):
        self.epsilon = self.initial_epsilon / (1.0 + self.epsilon_decay_rate * episode)


    def reset_epsilon(self):
        self.epsilon = self.initial_epsilon
