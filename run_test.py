import gymnasium as gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from advanced_continuous_function_approximation.environments.bipedal_walker.custom_bipedal_walker import BipedalWalkerEnvClass, BipedalWalkerModifiedEnvClass
from advanced_continuous_function_approximation.environments.pendulum.custom_pendulum import PendulumEnvClass
from advanced_continuous_function_approximation.environments.bipedal_walker.utils import run_and_display_env, run_and_record_env, play_videos, remove_videos

# multiprocess environment
vec_env = make_vec_env("BipedalWalker-v3", n_envs=4)
# vec_env = make_vec_env("Pendulum-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
model.learn(total_timesteps=400000)

# Set the directory for saving models
model_dir = "./models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, "ppo_bipedal_walker"))
# model.save(os.path.join(model_dir, "ppo_pendulum"))

model = PPO.load(os.path.join(model_dir, "ppo_bipedal_walker"), device='cpu')
# model = PPO.load(os.path.join(model_dir, "ppo_pendulum"), device='cpu')

class  Agent:
    def __init__(self, model):
        self.model = model
    
    def select_action(self, obs, info):
        action, _states = model.predict(obs)
        return action

agent = Agent(model)

max_episode_steps = None

env_class = BipedalWalkerEnvClass(hardcore=False, render_mode="rgb_array", max_episode_steps=max_episode_steps)
# env_class = BipedalWalkerModifiedEnvClass(hardcore=False, render_mode="rgb_array", max_episode_steps=max_episode_steps)
# env_class = PendulumEnvClass(render_mode="rgb_array", max_episode_steps=max_episode_steps)

env = env_class.create_env()
# env = env_class.create_env(
#             LEG_W_R_U=8*0.5, LEG_W_R_D=8*0.5, LEG_W_L_U=8*0.5, LEG_W_L_D=8*0.5, 
#             LEG_H_R_U=34*0.5, LEG_H_R_D=68*0.5, LEG_H_L_U=34*0.5, LEG_H_L_D=68*0.5
#             )

# Set the directory for saving videos
video_dir = "./videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

run_and_record_env(env, agent, video_dir=video_dir, num_episodes = 10, max_steps=None)

# Display the videos
play_videos(video_dir)

# Remove the videos
# remove_videos(video_dir)