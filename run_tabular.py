from tabular.rl_algorithms.exploration import Greedy, EpsilonGreedy, SoftMax
from tabular.rl_algorithms.agents import PolicyIteration, ValueIteration, RTDP, QPlanning, MonteCarloOnPolicy, \
    Sarsa, ExpectedSarsa, QLearning, DoubleQLearning, NStepSarsa, NStepExpectedSarsa, NStepTreeBackup, \
    DynaQ, DynaQPlus, PrioritizedSweeping

from tabular.environments.frozen_lake.utils import *
from tabular.environments.frozen_lake.wrappers import FrozenLakeWrapper

from pathlib import Path
from typing import NamedTuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


sns.set_theme()


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    max_episode_steps: int # Maximum episode steps
    learning_rate: float  # Learning rate
    learning_rate_decay: float # Learning rate decay
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    epsilon_decay: float # Exploration probability decay
    step_reward: float # Reward in each step
    hole_reward: float # Reward of reaching goal
    goal_reward: float # Reward of stepping into hole
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


params = Params(
    total_episodes = 1000,
    max_episode_steps = 300,
    learning_rate = 0.5,
    learning_rate_decay = 0.0,
    gamma = 0.99,
    epsilon = 0.1,
    epsilon_decay = 0.0,
    step_reward = -1,
    hole_reward = -10,
    goal_reward = 10,
    map_size = 8,
    seed = 7,
    is_slippery = True,
    n_runs = 1,
    action_size = None,
    state_size = None,
    proba_frozen = 0.9,
    savefig_folder = Path("../../_static/img/"),
)

# Set the seed
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exists
params.savefig_folder.mkdir(parents=True, exist_ok=True)


env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
    max_episode_steps= params.max_episode_steps,
)

env.action_space.seed(
        params.seed
    )  # Set the seed to get reproducible results when sampling the action space


params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")

env = FrozenLakeWrapper(env, params.step_reward, params.hole_reward, params.goal_reward)

# print(env.get_wrapper_attr('spec'))

res_all = pd.DataFrame()
st_all = pd.DataFrame()

agents = [
    # PolicyIteration(env, gamma=params.gamma, threshold1=1e-4, threshold2=1e-15), # gamma should never be 1 for Policy Iteration because it may not converge
    # ValueIteration(env, gamma=params.gamma, threshold=1e-4),
    # RTDP(env, gamma=params.gamma),
    # QPlanning(env, learning_rate=params.learning_rate, gamma=params.gamma, threshold=1e-4, n_planning=100),
    # MonteCarloOnPolicy(env, learning_rate=params.learning_rate, gamma=params.gamma, first_vist=False),
    # Sarsa(env, learning_rate=params.learning_rate, gamma=params.gamma),
    # ExpectedSarsa(env, learning_rate=params.learning_rate, gamma=params.gamma),
    QLearning(env, learning_rate=params.learning_rate, gamma=params.gamma),
    # DoubleQLearning(env, learning_rate=params.learning_rate, gamma=params.gamma),
    # NStepSarsa(env, learning_rate=params.learning_rate, gamma=params.gamma, n_step=1),
    # NStepExpectedSarsa(env, learning_rate=params.learning_rate, gamma=params.gamma, n_step=1),
    # NStepTreeBackup(env, learning_rate=params.learning_rate, gamma=params.gamma, n_step=1),
    # DynaQ(env, learning_rate=params.learning_rate, gamma=params.gamma, n_planning=30),
    # DynaQPlus(env, learning_rate=params.learning_rate, gamma=params.gamma, n_planning=30, reward_kappa=1e-1),
    # PrioritizedSweeping(env, learning_rate=params.learning_rate, gamma=params.gamma, n_planning=30, priority_threshold=1e-8, expected_update=True),
]

explorer = EpsilonGreedy(
        epsilon=params.epsilon,
        seed=params.seed
        )

for agent in agents:

    print(f"Map size: {params.map_size}x{params.map_size}")
    print(f"Agent: {type(agent).__name__}")

    tic = time.time()
    rewards, steps, episodes, qtables, all_states, all_actions = agent.train(
        explorer=explorer, total_episodes=params.total_episodes, n_runs=params.n_runs, seed=params.seed
        )
    toc = time.time()
    elapsed = toc - tic

    print(f"{type(agent).__name__} agent time: {elapsed:0.2f} sec")

    # Save the results in dataframes
    res, st = postprocess(params.n_runs, episodes, rewards, steps, params.map_size, type(agent).__name__)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)  # Average the Q-table between runs

    plot_states_actions_distribution(
        states=all_states, 
        actions=all_actions, 
        map_size=params.map_size, 
        agent_name=type(agent).__name__,
        savefig_folder=params.savefig_folder
    )  # Sanity check

    plot_q_values_map(
        qtable=qtable, 
        env=env, 
        map_size=params.map_size, 
        agent_name=type(agent).__name__, 
        time=elapsed,
        savefig_folder=params.savefig_folder
        )

env.close()

plot_steps_and_rewards(res_all, st_all, params.savefig_folder)


############################################################
n_episodes = 2

env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="human",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
    max_episode_steps= params.max_episode_steps,
)

env.action_space.seed(
        params.seed
    )  # Set the seed to get reproducible results when sampling the action space

env = FrozenLakeWrapper(env, params.step_reward, params.hole_reward, params.goal_reward)

explorer = Greedy(seed=params.seed)

for episode in range(n_episodes):
    state, info = env.reset()
    done = False

    while not done:
        action = explorer.choose_action(env.action_space, state, agent.qtable)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()