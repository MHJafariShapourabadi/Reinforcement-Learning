import numpy as np
import cv2  # For image processing

from os import path

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from gymnasium.envs.toy_text.utils import categorical_sample




LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3




class FrozenLakeImageFrameObservationWrapper(Wrapper):
    action_to_dir = {
        0: 'LEFT',
        1: 'DOWN',
        2: 'RIGHT',
        3: 'UP'
    }
    def __init__(self, env, slip_epsilon=0.3, step_reward=-1, hole_reward=0, goal_reward=0, episode_auto_restart=True, frame_size=64, crop_size=None, gray_scale=True, seed=None):
        super(FrozenLakeImageFrameObservationWrapper, self).__init__(env)

        self.is_slippery = is_slippery = self.env.unwrapped.spec.kwargs["is_slippery"]
        # s_slippery = self.env.get_wrapper_attr('spec').kwargs["is_slippery"]

        self.slip_epsilon = slip_epsilon

        self.desc = desc = self.env.unwrapped.desc
        # self.desc = desc = self.env.get_wrapper_attr('desc')

        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.n_actions = nA = 4
        self.n_states = nS = nrow * ncol

        self.env.unwrapped.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

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
                    li = self.env.unwrapped.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for action in [(a - 1) % 4, a, (a + 1) % 4]:
                                if action == a:
                                    prob = 1 - self.slip_epsilon + (self.slip_epsilon / 3.0)
                                else:
                                    prob = self.slip_epsilon / 3.0
                                li.append(
                                    (prob, *update_probability_matrix(row, col, action))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.reward_range = (min(step_reward, hole_reward, goal_reward), max(step_reward, hole_reward, goal_reward))

        self.action_space = Discrete(nA, seed=seed)

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

        self.frame_size = frame_size
        self.crop_size = self.window_size[0] if crop_size is None else crop_size
        self.gray_scale = gray_scale
        if self.gray_scale:
            self.observation_space = Box(low=0, high=255, shape=(1, frame_size, frame_size), dtype=np.uint8, seed=seed)
        else:
            self.observation_space = Box(low=0, high=255, shape=(3, frame_size, frame_size), dtype=np.uint8, seed=seed)

        self.state_to_observation = np.zeros((self.n_states, *self.observation_space.shape), dtype=np.float32)
        for s in range(self.n_states):
            frame = self._render_frame(s)
            observation = self.preprocess_image(frame)
            self.state_to_observation[s] = observation

    def step(self, a):
        transitions = self.env.unwrapped.P[self.env.unwrapped.s][a]
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

        observation = self.state_to_observation[s]

        observation, reward, terminated, truncated, info = observation, r, t, truncated, {"prob": p, "state":s}

        return observation, reward, terminated, truncated, info

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

        observation = self.state_to_observation[s]

        observation, info = observation, {"prob": 1, "state":s}
        return  observation, info

    # Helper function to preprocess the state image
    def preprocess_image(self, frame):
        # Center crop and resize (adjust based on FrozenLake)
        h, w, c = frame.shape
        crop_x = (w - self.crop_size) // 2
        crop_y = (h - self.crop_size) // 2
        cropped = frame[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size, :]
        resized = cv2.resize(cropped, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)

        if self.gray_scale:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            normalized = gray / 255.0  # Normalize pixel values to [0, 1]
            return normalized[np.newaxis, :, :]  # Add channel dimension (1, frame_size, frame_size)
        else:
            normalized = resized / 255.0  # Normalize pixel values to [0, 1]
            return normalized.transpose(2, 0, 1) # transpose channel dimension (3, frame_size, frame_size)

    def _render_frame(self, state, action=None):
        assert state < self.n_states, "state is out of state space"
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window_surface is None:
            pygame.init()

            self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = state // self.ncol, state % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = action if action is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
