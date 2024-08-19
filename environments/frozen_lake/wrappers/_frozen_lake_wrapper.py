import numpy as np

import gymnasium as gym
from gymnasium.envs.toy_text.utils import categorical_sample

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class FrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env, step_reward=-1, hole_reward=0, goal_reward=0, episode_auto_restart=True):
        super(FrozenLakeWrapper, self).__init__(env)
        
        is_slippery = self.env.unwrapped.spec.kwargs["is_slippery"]
        # s_slippery = self.env.get_wrapper_attr('spec').kwargs["is_slippery"]

        self.desc = desc = self.env.unwrapped.desc
        # self.desc = desc = self.env.get_wrapper_attr('desc')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

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
                for a in range(4):
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
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.env.unwrapped.s = s
        self.env.unwrapped.lastaction = a

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p}

