# %%
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import  generate_random_map

import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import torch


from function_approximation.environments.frozen_lake.wrappers import FrozenLakeVectorObservationWrapper
from function_approximation.rl_algorithms.exploration import GreedyExploration, EpsilonGreedyExploration, SoftmaxExploration
from function_approximation.rl_algorithms.agents import PERDuelingDoubleDeepNStepTreeBackup, DuelingDoubleDeepQNetwork,\
    PERDuelingDoubleDeepQNetwork, PERDuelingDoubleDeepSarsa, PERDuelingDoubleDeepNStepSarsa, REINFORCE
from function_approximation.environments.frozen_lake.utils import run_and_display_env
from function_approximation.environments.frozen_lake.utils import run_and_display_env, run_and_record_env, play_videos, remove_videos
from function_approximation.environments.frozen_lake.utils import plot_with_matplotlib, plot_with_seaborn, plot_q_values_map


# %%

map_size = 8
proba_frozen = 0.94
seed = 7
max_episode_steps = 300
is_slippery = True

env = gym.make(
      "FrozenLake-v1",
      is_slippery=is_slippery,
      render_mode="rgb_array",
      desc=generate_random_map(size=map_size, p=proba_frozen, seed=seed),
      max_episode_steps= max_episode_steps,)

modified_env = FrozenLakeVectorObservationWrapper(
    env=env,
    step_reward=-1,
    hole_reward=-10,
    goal_reward=10,
    active_neighbour=2,
    seed=seed)

print(modified_env)
print(modified_env.env)
print(modified_env.env.env)
print(modified_env.observation_space.n)
print(modified_env.observation_shape)
print(modified_env.action_space.n)

# %%

# exploration = SoftmaxExploration(
#     temperature_start=1.0,
#     temperature_end=0.1,
#     temperature_decay=None,
#     decay="linear",
#     seed=None
# )

exploration = EpsilonGreedyExploration(
    epsilon_start=0.5,
    epsilon_end=0.1,
    epsilon_decay=0.001,
    decay="linear",
    seed=None
)

# agent = PERDuelingDoubleDeepNStepTreeBackup(
#     env=modified_env,
#     exploration=exploration,
#     n_step=5,
#     lr_start=0.5, lr_end=0.001, lr_decay=0.0001, decay="linear",
#     gamma=0.99, Huberbeta=1.0, buffer_size=2048, batch_size=512, polyak_tau=0.05,
#     alpha_start=0.6, alpha_end=0.9, alpha_increment=1e-3, 
#     beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
#     seed=None, verbose=True
# )

# agent = DuelingDoubleDeepQNetwork(
#     env=modified_env,
#     exploration=exploration, 
#     lr_start=0.5, lr_end=0.005, lr_decay=0.001, decay="linear", 
#     gamma=0.99, Huberbeta=1.0, buffer_size=5000, batch_size=512, polyak_tau=1.0, 
#     seed=None, verbose=False
# )

# agent = PERDuelingDoubleDeepQNetwork(
#     env=modified_env,
#     exploration=exploration,
#     lr_start=0.5, lr_end=0.001, lr_decay=0.0001, decay="linear",
#     gamma=0.99, Huberbeta=1.0, buffer_size=2048, batch_size=512, polyak_tau=0.01,
#     alpha_start=0.6, alpha_end=0.9, alpha_increment=1e-3, 
#     beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
#     seed=None, verbose=True
# )

# agent = PERDuelingDoubleDeepSarsa(
#     env=modified_env,
#     exploration=exploration,
#     lr_start=0.5, lr_end=0.01, lr_decay=0.0001, decay="linear",
#     gamma=0.99, Huberbeta=1.0, buffer_size=5000, batch_size=512, polyak_tau=0.5,
#     alpha_start=0.6, alpha_end=0.9, alpha_increment=1e-3, 
#     beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
#     seed=None, verbose=True
# )

# agent = PERDuelingDoubleDeepNStepSarsa(
#     env=modified_env,
#     exploration=exploration,
#     n_step=5,
#     lr_start=0.5, lr_end=0.01, lr_decay=0.0001, decay="linear",
#     gamma=0.99, Huberbeta=1.0, buffer_size=5000, batch_size=512, polyak_tau=0.5,
#     alpha_start=0.6, alpha_end=0.9, alpha_increment=1e-3, 
#     beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
#     seed=None, verbose=True
# )

agent = REINFORCE(
    env=modified_env,
    lr_start=0.5, lr_end=0.001, lr_decay=0.0001, decay="linear",
    gamma=0.99,
    seed=None, verbose=True
)

# %%

num_episodes = 500
agent.verbose = False
agent.training()

tic = time.time()

episode_rewards, episode_steps = agent.train(num_episodes)

toc = time.time()
elapsed = toc - tic

# %%

all_states_vectors = torch.tensor(modified_env.state_to_vector, dtype=torch.float32, device=agent.device)
qtable = agent.policy_net(all_states_vectors).detach().cpu().numpy()

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