from . import EpsilonGreedy

import numpy as np
import gymnasium as gym

class SoftMax(EpsilonGreedy):
    def choose_action(self, action_space:gym.spaces.Discrete, state: int, qtable: np.ndarray, mask: np.ndarray | None=None)  -> int:
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explore_exploit_tradeoff = self.rng.uniform(0, 1)

        if mask is None:
            mask = np.ones_like(qtable).astype(np.int8)

        # Exploration
        if explore_exploit_tradeoff < self.epsilon: 
            max_val = np.max(np.abs(qtable[state, :])) + 1e-40
            probs = np.exp(qtable[state, :] / max_val) * mask[state, :] + mask[state, :] * 1e-40
            probs = probs / np.sum(probs)
            action = self.rng.choice(np.arange(action_space.n) ,p=probs)

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            action = self.argmax(qtable[state, :], mask[state, :])
            
        return int(action)
