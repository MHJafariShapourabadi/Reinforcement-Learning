import gymnasium as gym

class PendulumEnvClass:
    def __init__(self, render_mode="rgb_array", max_episode_steps=None):
        self.render_mode=render_mode
        self.max_episode_steps = max_episode_steps
    def create_env(self,):
        env = gym.make("Pendulum-v1", render_mode=self.render_mode, g=9.81, max_episode_steps=self.max_episode_steps)  # default g=10.0
        return env
