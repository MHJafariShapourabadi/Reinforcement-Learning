import numpy as np
import cv2  # For image processing

import gymnasium as gym
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from gymnasium.envs.toy_text.utils import categorical_sample




class FrozenLakeFrameObservationWrapper(ObservationWrapper):
    action_to_dir = {
        0: 'LEFT',
        1: 'DOWN',
        2: 'RIGHT',
        3: 'UP'
    }
    def __init__(self, env, frame_size=64, crop_size=497, gray_scale=True):
        super().__init__(env)
        self.frame_size = frame_size
        self.crop_size = crop_size
        self.gray_scale = gray_scale
        if self.gray_scale:
            self.observation_space = Box(low=0, high=255, shape=(1, frame_size, frame_size), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=255, shape=(3, frame_size, frame_size), dtype=np.uint8)

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

    def observation(self, obs):
        frame = self.env.render()
        return self.preprocess_image(frame)