import numpy as np
import gymnasium as gym


class Greedy:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def choose_action(self, action_space:gym.spaces.Discrete, state: int, qtable: np.ndarray, mask: np.ndarray | None=None) -> int:
        if mask is None:
            mask = np.ones_like(qtable).astype(np.int8)

        action = self.argmax(qtable[state, :], mask[state, :])

        return int(action)


    def argmax(self, q_values:np.ndarray, mask:np.ndarray | None=None):
        """argmax
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value, with ties randomly broken
        """

        top = float("-inf")
        ties = []

        mask = mask.astype("bool")

        for i in range(len(q_values)):
            if mask[i]:
                if q_values[i] > top:
                    top = q_values[i]
                    ties = []

                if q_values[i] == top:
                    ties.append(i)

        return self.rng.choice(ties)