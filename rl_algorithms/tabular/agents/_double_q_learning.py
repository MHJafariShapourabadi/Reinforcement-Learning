from . import QLearning

import numpy as np

from tqdm import tqdm



class DoubleQLearning(QLearning): # It doesn't work well with negative rewards
    def __init__(self, env, learning_rate, gamma, learning_rate_dacay=0.0, initial_qtable=None, seed=None):
        super(QLearning, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, learning_rate_dacay=learning_rate_dacay, initial_qtable=initial_qtable)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        

    def reset_qtable(self, initial_qtable=None):
        """Reset the Q-table."""
        if initial_qtable is None:
            self.qtable1 = np.zeros((self.state_size, self.action_size))
            self.qtable2 = np.zeros((self.state_size, self.action_size))
        else:
            self.qtable1 = initial_qtable
            self.qtable2 = initial_qtable
        self.qtable = (self.qtable1 + self.qtable2) # / 2.0


    def update(self, state, action, new_state, reward):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""

        q_update=None

        if self.rng.random() > 0.5:
            qarray1 = self.qtable1[new_state, self.mask[new_state, :].astype("bool")]
            arg1 = self.rng.choice(np.flatnonzero(qarray1 == qarray1.max()))
            # arg1 = np.argmax(self.qtable1[new_state, self.mask[new_state, :].astype("bool")])

            delta1 = (
                reward
                + self.gamma * self.qtable2[new_state, self.mask[new_state, :].astype("bool")][arg1]
                - self.qtable1[state, action]
            )
            q_update1 = self.qtable1[state, action] + self.learning_rate * delta1
            self.qtable1[state, action] = q_update1
            q_update = q_update1

        else:
            qarray2 = self.qtable2[new_state, self.mask[new_state, :].astype("bool")]
            arg2 = self.rng.choice(np.flatnonzero(qarray2 == qarray2.max()))
            # arg2 = np.argmax(self.qtable2[new_state, self.mask[new_state, :].astype("bool")])

            delta2 = (
                reward
                + self.gamma * self.qtable1[new_state, self.mask[new_state, :].astype("bool")][arg2]
                - self.qtable2[state, action]
            )
            q_update2 = self.qtable2[state, action] + self.learning_rate * delta2
            self.qtable2[state, action] = q_update2
            q_update = q_update2

        self.qtable[state, action] = (self.qtable1[state, action] + self.qtable2[state, action]) # / 2
        
        return q_update