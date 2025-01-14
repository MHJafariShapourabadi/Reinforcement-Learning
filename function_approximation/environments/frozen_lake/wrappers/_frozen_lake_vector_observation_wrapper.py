import numpy as np

import gymnasium as gym
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from gymnasium.envs.toy_text.utils import categorical_sample





LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class FrozenLakeVectorObservationWrapper(Wrapper):
    action_to_dir = {
        0: 'LEFT',
        1: 'DOWN',
        2: 'RIGHT',
        3: 'UP'
    }
    def __init__(self, env, step_reward=-1, hole_reward=0, goal_reward=0, episode_auto_restart=True,
                 active_neighbour=2, seed=None):
        super(FrozenLakeVectorObservationWrapper, self).__init__(env)

        is_slippery = self.env.unwrapped.spec.kwargs["is_slippery"]
        # s_slippery = self.env.get_wrapper_attr('spec').kwargs["is_slippery"]

        self.desc = desc = self.env.unwrapped.desc
        # self.desc = desc = self.env.get_wrapper_attr('desc')

        self.active_neighbour = active_neighbour

        self.n_active_features = 2 * active_neighbour + 1

        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.padded_nrow = nrow + 2 * active_neighbour

        self.padded_ncol = ncol + 2 * active_neighbour

        self.n_actions = nA = 4
        self.n_states = nS = nrow * ncol
        self.n_observations = self.padded_nrow * self.padded_ncol
        self.observation_shape = (self.n_observations,)

        # self.observation_space = Discrete(nS, seed=seed)

        self.state_to_vector = np.zeros((self.n_states, self.n_observations))
        for s in range(self.n_states):
          self.state_to_vector[s] = self.vectorize(s)

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = desc[new_row, new_col]
            if new_letter == b"G":
                terminated = True
                reward = float(goal_reward)
            elif new_letter == b"H":
                if episode_auto_restart:
                    terminated = False
                    new_state = to_s(0, 0)
                else:
                    terminated = True
                reward = float(hole_reward)
            else:
                terminated = False
                reward = float(step_reward)
            return new_state, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(nA):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

    def step(self, a):
        transitions = self.P[self.env.unwrapped.s][a]
        i = categorical_sample([t[0] for t in transitions], self.env.unwrapped.np_random)
        p, s, r, t = transitions[i]
        self.env.unwrapped.s = s
        self.env.unwrapped.lastaction = a

        if self.env.unwrapped.render_mode == "human":
            self.env.unwrapped.render()

        # TimeLimit handling
        self.env._elapsed_steps += 1

        if self.env._elapsed_steps >= self.env._max_episode_steps:
            truncated = True
        else:
            truncated = False

        observation = self.state_to_vector[s]

        s, reward, terminated, truncated, info = int(s), r, t, truncated, {"prob": p, "observation":observation}

        return s, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        super().reset(seed=seed)
        s = categorical_sample(self.env.unwrapped.initial_state_distrib, self.env.unwrapped.np_random)
        self.env.unwrapped.s = s
        self.env.unwrapped.lastaction = None

        if self.env.unwrapped.render_mode == "human":
            self.env.unwrapped.render()

        observation = self.state_to_vector[s]

        s, info = int(s), {"prob": 1, "observation":observation}
        return  s, info

    # Helper function to create the state vector
    def vectorize(self, state):
        assert state < self.observation_space.n, "state is out of state space"
        observation = np.zeros((self.padded_nrow, self.padded_ncol), dtype=self.observation_space.dtype)
        row, col = self.state_to_pos(state)
        observation[row - self.active_neighbour : row + self.active_neighbour + 1, col - self.active_neighbour : col + self.active_neighbour + 1] = 1
        observation = observation.ravel()
        return observation

    def state_to_pos(self, state):
        row = state // self.ncol
        col = state % self.ncol
        row += self.active_neighbour
        col += self.active_neighbour
        return row, col