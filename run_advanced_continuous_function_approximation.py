# %%
import gymnasium as gym

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import torch

from advanced_continuous_function_approximation.environments.custom_envs import AsyncVectorEnv
from advanced_continuous_function_approximation.environments.bipedal_walker.custom_bipedal_walker import BipedalWalkerEnvClass, BipedalWalkerModifiedEnvClass
from advanced_continuous_function_approximation.environments.pendulum.custom_pendulum import PendulumEnvClass

from advanced_continuous_function_approximation.rl_algorithms.agents import A2C, NStepA2C, A2CGAE, \
    PPO, NStepPPO, PPOGAE, PPOPER, NStepPPOPER, PPOGAEPER, PPOGAEROLL, PPOGAEROLLNStep

from advanced_continuous_function_approximation.environments.bipedal_walker.utils import run_and_display_env, run_and_record_env, play_videos, remove_videos
from advanced_continuous_function_approximation.environments.bipedal_walker.utils import plot_with_matplotlib, plot_with_seaborn, plot_q_values_map

from importlib import reload


# %%


class DummyAgent:
    def __init__(self, env):
        self.env =env

    def select_action(self, state, info):
        return self.env.action_space.sample()


if __name__ == "__main__":
    num_envs = 8
    seed = 7
    max_episode_steps = None

    env_class = BipedalWalkerEnvClass(hardcore=False, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    # env_class = BipedalWalkerModifiedEnvClass(hardcore=False, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    # env_class = PendulumEnvClass( render_mode="rgb_array", max_episode_steps=max_episode_steps)

    env = env_class.create_env()
    # env = env_class.create_env(
    #             LEG_W_R_U=8*0.5, LEG_W_R_D=8*0.5, LEG_W_L_U=8*0.5, LEG_W_L_D=8*0.5, 
    #             LEG_H_R_U=34*0.5, LEG_H_R_D=68*0.5, LEG_H_L_U=34*0.5, LEG_H_L_D=68*0.5
    #             )

    print(type(env.unwrapped).__name__)
    print(env)
    print(env.observation_space)
    print(env.observation_space.shape)
    print(env.action_space)
    print(env.action_space.shape)

    # %%

    # agent = DummyAgent(env)

    # run_and_display_env(env, agent, num_episodes=1, max_steps=6)


    # %%

    # agent = DummyAgent(env)

    # # Set the directory for saving videos
    # video_dir = "./videos"
    # if not os.path.exists(video_dir):
    #     os.makedirs(video_dir)

    # run_and_record_env(env, agent, video_dir=video_dir, num_episodes = 1, max_steps=None)

    # # Display the videos
    # play_videos(video_dir)

    # # Remove the videos
    # remove_videos(video_dir)

    # %%

    env_fns_args = [() for _ in range(num_envs)]
    env_fns = [env_class.create_env for _ in range(num_envs)]
    vector_env = AsyncVectorEnv(env_fns, env_fns_args)

    # %%

    # Choose the agent

    # agent = A2C(env=vector_env, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = NStepA2C(env=vector_env, n_step=5, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = A2CGAE(env=vector_env, n_step=5, lambd=0.6, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)

    # agent = PPO(env=vector_env, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99, seed=None, verbose=False)
    # agent = NStepPPO(env=vector_env, n_step=5, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99, seed=None, verbose=False)
    # agent = PPOPER(env=vector_env, actor_lr_start=1e-2, actor_lr_end=1e-3, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-1, critic_lr_end=1e-2, critic_lr_decay=0.000005, critic_decay="exponential",buffer_size=2048, batch_size=128,alpha_start=0.8, alpha_end=1.0, alpha_increment=1e-3, beta_start=0.4, beta_end=1.0, beta_increment=1e-4, r_clip_epsilon=0.2, v_clip_epsilon=0.2, entropy_coef=0.01, gamma=0.99,seed=None, verbose=False)
    # agent = NStepPPOPER(env=vector_env, n_step=5, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3, beta_start=0.4, beta_end=1.0, beta_increment=1e-4, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99, seed=None, verbose=False)
    
    # agent = PPOGAE(env=vector_env, n_step=128, lambd = 0.5, actor_weight_decay = 0.1, actor_lr_start=0.0001, actor_lr_end=0.00005, actor_lr_decay=0.00001, actor_decay="exponential", critic_weight_decay = 0.01, critic_lr_start=0.0005, critic_lr_end=0.0001, critic_lr_decay=0.00001, critic_decay="exponential", n_updates_per_iter=5, buffer_size=2048, batch_size=128, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, std=1.0, gamma=0.99, max_grad_norm=0.1, target_kl = 0.02, seed=None, verbose=False)
    # agent = PPOGAEPER(env=vector_env, n_step=512, lambd = 0.5, actor_weight_decay = 0.01, actor_lr_start=0.0001, actor_lr_end=0.00005, actor_lr_decay=0.00001, actor_decay="exponential", critic_weight_decay = 0.01, critic_lr_start=0.0005, critic_lr_end=0.0001, critic_lr_decay=0.00001, critic_decay="exponential", n_updates_per_iter=5, buffer_size=2048, batch_size=128, alpha_start=0.8, alpha_end=1.0, alpha_increment=1e-3, beta_start=0.6, beta_end=1.0, beta_increment=1e-4, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, std=1.0, gamma=0.99, max_grad_norm=0.1, target_kl = 0.02, seed=None, verbose=False)
    agent = PPOGAEROLL(env=vector_env, n_rollout=1600, lambd = 0.5, actor_weight_decay = 0.01, actor_lr_start=0.0001, actor_lr_end=0.00005, actor_lr_decay=0.00001, actor_decay="exponential", critic_weight_decay = 0.01, critic_lr_start=0.0005, critic_lr_end=0.0001, critic_lr_decay=0.00001, critic_decay="exponential", n_updates_per_iter=5, batch_size=128, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, std=0.5, gamma=0.99, max_grad_norm=0.1, target_kl = 1.0, seed=None, verbose=False)
    # agent = PPOGAEROLLNStep(env=vector_env, n_rollout=1600, n_step=128, lambd = 0.5, actor_weight_decay = 0.01, actor_lr_start=0.0001, actor_lr_end=0.00005, actor_lr_decay=0.00001, actor_decay="exponential", critic_weight_decay = 0.01, critic_lr_start=0.0005, critic_lr_end=0.0001, critic_lr_decay=0.00001, critic_decay="exponential", n_updates_per_iter=5, batch_size=128, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, std=1.1, gamma=0.99, max_grad_norm=0.1, target_kl = 0.02, seed=None, verbose=False)

    # %%

    max_episodes = 1000
    agent.verbose = False
    agent.training()

    # Set the directory for saving models
    model_dir = "./models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    agent_steps = 'last'
    agent_path = os.path.join(model_dir, f"ppo_{type(agent.env.dummy_env.unwrapped).__name__}_{agent_steps}_steps.pth")
    if os.path.exists(agent_path):
        agent.load_agent(agent_path)

    tic = time.time()

    episode_rewards, episode_steps = agent.train(max_episodes=max_episodes)

    toc = time.time()
    elapsed = toc - tic

    print(f"Time elapsed: {elapsed} sec")

    agent.save_agent('last')

    plot_with_seaborn(episode_rewards, episode_steps)

    vector_env.close()

    # %%

    # Set the directory for saving models
    model_dir = "./models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    agent_steps = 'last'
    agent_path = os.path.join(model_dir, f"ppo_{type(agent.env.dummy_env.unwrapped).__name__}_{agent_steps}_steps.pth")
    agent.load_agent(agent_path)

    # %%
    
    # # Set the directory for saving models
    # model_dir = "./models"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    
    # # Saving models
    # print(f"... Saving model to {model_dir} ...")
    # # actor_file_path = os.path.join(model_dir, "ppo_pendulum_actor.pth")
    # actor_file_path = os.path.join(model_dir, "ppo_bipedal_walker_actor.pth")
    # torch.save(agent.actor.state_dict(), actor_file_path)
    # # critic_file_path = os.path.join(model_dir, "ppo_pendulum_critic.pth")
    # critic_file_path = os.path.join(model_dir, "ppo_bipedal_walker_critic.pth")
    # torch.save(agent.critic.state_dict(), critic_file_path)

    # %%

    # Set the directory for saving models
    # model_dir = "./models"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    # # actor_file_path = os.path.join(model_dir, "ppo_pendulum_actor.pth")
    # actor_file_path = os.path.join(model_dir, "ppo_bipedal_walker_actor.pth")
    # # critic_file_path = os.path.join(model_dir, "ppo_pendulum_critic.pth")
    # critic_file_path = os.path.join(model_dir, "ppo_bipedal_walker_critic.pth")

    # # Loading models
    # print(f"... Loading model from {model_dir} ...")
    # agent.actor.load_state_dict(torch.load(actor_file_path, weights_only=True))
    # agent.critic.load_state_dict(torch.load(critic_file_path, weights_only=True))



    # %%

    agent.verbose = False
    # agent.training()
    agent.evaluating(seed=None)
    agent.greedy_actions(True)

    # Set the directory for saving videos
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    run_and_record_env(env, agent, video_dir=video_dir, num_episodes = 10, max_steps=None)

    # %%

    # Set the directory for saving videos
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Display the videos
    play_videos(video_dir)

    # Remove the videos
    # remove_videos(video_dir)

    # %%
