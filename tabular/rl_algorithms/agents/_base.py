from ..exploration import Greedy

from abc import ABC, abstractmethod
import numpy as np 
from tqdm import tqdm
import gymnasium as gym
from queue import PriorityQueue


class Agent(ABC):
    def __init__(self, env, learning_rate, gamma, learning_rate_dacay=0.0, initial_qtable=None):
        self.env = env
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.gamma = gamma
        self.initial_qtable = initial_qtable
        self.reset_qtable(initial_qtable)
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.learning_rate_dacay = learning_rate_dacay
        if hasattr(env.unwrapped, "mask"):
            self.mask = env.unwrapped.mask
        else:
            self.mask = np.ones((self.state_size, self.action_size), dtype=np.int8)
        self.reset_policy()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def reset_qtable(self, initial_qtable=None):
        """Reset the Q-table."""
        if initial_qtable is None:
            self.qtable = np.zeros((self.state_size, self.action_size))
        else:
            self.qtable = initial_qtable

    def reset_policy(self):
        self.policy = np.zeros(self.state_size, dtype=np.int32)
        for state in range(self.state_size):
            self.policy[state] = int(self.env.action_space.sample(self.mask[state, :]))
        return self.policy

    def update_policy(self, seed=None):
        action_selector = Greedy(seed)
        policy = np.zeros(self.state_size, dtype=np.int32)
        for state in range(self.state_size):
            self.policy[state] = action_selector.choose_action(self.env.action_space, state, self.qtable, self.mask)


    def get_policy(self, random=False, seed=None):
        return self.policy

    def get_state_available_actions(self, state):
        actions = []
        for action in range(self.action_size):
            if self.mask[state, action]:
                actions.append(action)
        return actions

    def reset_learning_rate(self):
        self.learning_rate = self.initial_learning_rate

    def decay_learning_rate(self, episode):
        self.learning_rate = self.initial_learning_rate / (1.0 + self.learning_rate_dacay * episode)