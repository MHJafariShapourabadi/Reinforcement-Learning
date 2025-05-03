import gymnasium as gym

class BipedalWalkerEnvClass:
    def __init__(self, hardcore=False, render_mode="rgb_array", max_episode_steps=None):
        self.hardcore = hardcore
        self.render_mode=render_mode
        self.max_episode_steps = max_episode_steps
    def create_env(self,):
        env = gym.make("BipedalWalker-v3", hardcore=self.hardcore, render_mode=self.render_mode, max_episode_steps=self.max_episode_steps)
        return env
