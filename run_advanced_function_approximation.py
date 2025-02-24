# %%
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import  generate_random_map

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import torch

from advanced_function_approximation.environments.frozen_lake import wrappers
from advanced_function_approximation.rl_algorithms import agents
from advanced_function_approximation.environments.frozen_lake import utils

from advanced_function_approximation.environments.custom_envs import AsyncVectorEnv
from advanced_function_approximation.environments.frozen_lake.wrappers import FrozenLakeVectorObservationWrapper, FrozenLakeImageFrameObservationWrapper, FrozenLakeArrayFrameObservationWrapper
from advanced_function_approximation.environments.frozen_lake.custom_frozen_lake import FrozenLakeVectorObsevationEnvClass, FrozenLakeImageFrameObsevationEnvClass, FrozenLakeArrayFrameObsevationEnvClass

from advanced_function_approximation.rl_algorithms.exploration import GreedyExploration, EpsilonGreedyExploration, SoftmaxExploration
from advanced_function_approximation.rl_algorithms.agents import A2C, NStepA2C, A2CGAE, \
A3C, NStepA3C, A3CGAE, ActorCritic, NStepActorCritic, ActorCriticGAE, \
    PPO, NStepPPO, PPOGAE, PPOPER, NStepPPOPER, PPOGAEPER

from advanced_function_approximation.environments.frozen_lake.utils import run_and_display_env, run_and_record_env, play_videos, remove_videos
from advanced_function_approximation.environments.frozen_lake.utils import plot_with_matplotlib, plot_with_seaborn, plot_q_values_map

from importlib import reload

reload(wrappers)
reload(agents)
reload(utils)

# %%


class DummyAgent:
    def __init__(self, env):
        self.env =env

    def select_action(self, state, info):
        return self.env.action_space.sample(), False


if __name__ == "__main__":
    num_envs = 8
    map_size = 8
    proba_frozen = 0.94
    seed = 7
    max_episode_steps = 300
    is_slippery = True
    episode_auto_restart = True


    env_class = FrozenLakeVectorObsevationEnvClass(map_size, proba_frozen, max_episode_steps, is_slippery, episode_auto_restart, seed)

    modified_env = env_class.create_env()

    print(modified_env)
    print(modified_env.observation_space)
    print(modified_env.observation_space.shape)
    print(modified_env.action_space.n)


    vector = modified_env.state_to_vector[62]
    print(vector.shape)
    print(vector.dtype)

    # %%

    # agent = DummyAgent(modified_env)

    # run_and_display_env(modified_env, agent, num_episodes=1, max_steps=6)


    # %%

    env_fns_args = [() for _ in range(num_envs)]
    env_fns = [env_class.create_env for _ in range(num_envs)]
    vector_env = AsyncVectorEnv(env_fns, env_fns_args)

    # %%

    # Choose the agent

    # agent = ActorCritic(env_class=env_class, input_dim=modified_env.observation_shape[0], action_dim=modified_env.action_space.n, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = NStepActorCritic(env_class=env_class, input_dim=modified_env.observation_shape[0], action_dim=modified_env.action_space.n, n_step=5, actor_lr_start=1e-2, actor_lr_end=1e-3, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = ActorCriticGAE(env_class=env_class, input_dim=modified_env.observation_shape[0], action_dim=modified_env.action_space.n, n_step=5, lambd = 0.95, actor_lr_start=1e-2, actor_lr_end=1e-3, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)

    # agent = A3C(env_class=env_class, input_dim=modified_env.observation_shape[0], action_dim=modified_env.action_space.n, actor_lr_start=1e-3, actor_lr_end=1e-4, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-3, critic_lr_end=1e-4, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = NStepA3C(env_class=env_class, input_dim=modified_env.observation_shape[0], action_dim=modified_env.action_space.n, n_step=5, actor_lr_start=1e-3, actor_lr_end=1e-4, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-3, critic_lr_end=1e-4, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = A3CGAE(env_class=env_class, input_dim=modified_env.observation_shape[0], action_dim=modified_env.action_space.n, n_step=5, lambd = 0.5, actor_lr_start=1e-3, actor_lr_end=1e-4, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-3, critic_lr_end=1e-4, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)

    # agent = A2C(env=vector_env, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = NStepA2C(env=vector_env, n_step=5, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)
    # agent = A2CGAE(env=vector_env, n_step=5, lambd=0.6, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0,  seed=None, verbose=False)

    # agent = PPO(env=vector_env, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99, seed=None, verbose=False)
    # agent = NStepPPO(env=vector_env, n_step=5, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99, seed=None, verbose=False)
    # agent = PPOGAE(env=vector_env, n_step=5, lambd = 0.1, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99, seed=None, verbose=False)
    # agent = PPOPER(env=vector_env,actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential",critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential",buffer_size=256, batch_size=64,alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3,beta_start=0.4, beta_end=1.0, beta_increment=1e-4,r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99,seed=None, verbose=False)
    # agent = NStepPPOPER(env=vector_env, n_step=5, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3, beta_start=0.4, beta_end=1.0, beta_increment=1e-4, r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, gamma=0.99, seed=None, verbose=False)
    agent = PPOGAEPER(env=vector_env, n_step=5, lambd = 0.5, actor_lr_start=1e-2, actor_lr_end=1e-2, actor_lr_decay=0.00005, actor_decay="exponential", critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential", buffer_size=256, batch_size=64, alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3, beta_start=0.4, beta_end=1.0, beta_increment=1e-4, r_clip_epsilon=0.2, v_clip_epsilon=0.2, entropy_coef=0.01, gamma=0.99, seed=None, verbose=False)


    # %%

    max_episodes = 250
    agent.verbose = False
    agent.training()

    tic = time.time()

    # For other methods except A3C:
    episode_rewards, episode_steps = agent.train(max_episodes=max_episodes)

    # For A3C:
    # episode_rewards, episode_steps = agent.train(max_episodes=max_episodes, num_workers=8)

    toc = time.time()
    elapsed = toc - tic

    vector_env.close()

    # %%

    all_states_vectors = torch.tensor(modified_env.state_to_vector, dtype=torch.float32, device=agent.device)

    # For other Actor-Critic and A2C methods:
    qtable = agent.actor(all_states_vectors).detach().cpu().numpy()

    # For A3C:
    # qtable = agent.global_actor(all_states_vectors).detach().cpu().numpy()

    plot_q_values_map(
            qtable=qtable, 
            map_size=map_size, 
            agent_name=type(agent).__name__, 
            time=elapsed,
            savefig_folder=None
            )


    # %%

    plot_with_seaborn(episode_rewards, episode_steps)

    # %%

    agent.verbose = False
    # agent.evaluating(seed=None)

    # Set the directory for saving videos
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    run_and_record_env(modified_env, agent, video_dir=video_dir, num_episodes = 1, max_steps=None)

    # Display the videos
    play_videos(video_dir)

    # Remove the videos
    # remove_videos(video_dir)
