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
    PERDuelingDoubleDeepQNetwork, PERDuelingDoubleDeepSarsa, PERDuelingDoubleDeepNStepSarsa, REINFORCE,\
    REINFORCEWithBaseline, ActorCritic, NStepActorCritic, ActorCriticGAE, A3C, NStepA3C, A3CGAE
from function_approximation.environments.frozen_lake.utils import run_and_display_env
from function_approximation.environments.frozen_lake.utils import run_and_display_env, run_and_record_env, play_videos, remove_videos
from function_approximation.environments.frozen_lake.utils import plot_with_matplotlib, plot_with_seaborn, plot_q_values_map


# %%

class EnvClass:
    def __init__(self, map_size, proba_frozen, max_episode_steps, is_slippery, seed):
        self.map_size = map_size
        self.proba_frozen = proba_frozen
        self.max_episode_steps = max_episode_steps
        self.is_slippery = is_slippery
        self.seed = seed

    def create_env(self, slip_epsilon=0.6, step_reward=-1, hole_reward=-10, goal_reward=10, active_neighbour=2,):
        import gymnasium as gym
        env = gym.make(
            "FrozenLake-v1",
            is_slippery=self.is_slippery,
            render_mode="rgb_array",
            desc=generate_random_map(size=self.map_size, p=self.proba_frozen, seed=self.seed),
            max_episode_steps= self.max_episode_steps,)

        modified_env = FrozenLakeVectorObservationWrapper(
            env=env,
            slip_epsilon=slip_epsilon,
            step_reward=step_reward,
            hole_reward=hole_reward,
            goal_reward=goal_reward,
            active_neighbour=active_neighbour,
            seed=self.seed)

        return modified_env

if __name__ == "__main__":

    map_size = 8
    proba_frozen = 0.94
    seed = 7
    max_episode_steps = 300
    is_slippery = True


    env_class = EnvClass(map_size, proba_frozen, max_episode_steps, is_slippery, seed)

    modified_env = env_class.create_env()

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

    # exploration = EpsilonGreedyExploration(
    #     epsilon_start=0.5,
    #     epsilon_end=0.1,
    #     epsilon_decay=0.001,
    #     decay="linear",
    #     seed=None
    # )

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
    #     lr_start=0.1, lr_end=0.0005, lr_decay=0.00001, decay="linear",
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

    # agent = REINFORCE(
    #     env=modified_env,
    #     lr_start=0.1, lr_end=0.01, lr_decay=0.005, decay="exponential",
    #     gamma=0.99,
    #     seed=None, verbose=False
    # )

    # agent = REINFORCEWithBaseline(
    #     env=modified_env,
    #     policy_lr_start=1e-1, policy_lr_end=1e-2, policy_lr_decay=0.005, policy_decay="exponential",
    #     value_lr_start=1e-1, value_lr_end=1e-2, value_lr_decay=0.005, value_decay="exponential",
    #     entropy_coef=0.00001, Huberbeta=1.0,
    #     gamma=0.99,
    #     seed=None, verbose=False
    # )

    # agent = ActorCritic(
    #     env_class=env_class,
    #     input_dim=modified_env.observation_shape[0], 
    #     action_dim=modified_env.action_space.n,
    #     actor_lr_start=1e-2, actor_lr_end=1e-3, actor_lr_decay=0.00005, actor_decay="exponential",
    #     critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential",
    #     gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0, 
    #     seed=None, verbose=False
    # )

    agent = NStepActorCritic(
        env_class=env_class,
        input_dim=modified_env.observation_shape[0], 
        action_dim=modified_env.action_space.n,
        n_step=5,
        actor_lr_start=1e-2, actor_lr_end=1e-3, actor_lr_decay=0.00005, actor_decay="exponential",
        critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential",
        gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0, 
        seed=None, verbose=False
    )

    # agent = ActorCriticGAE(
    #     env_class=env_class,
    #     input_dim=modified_env.observation_shape[0], 
    #     action_dim=modified_env.action_space.n,
    #     n_step=5, lambd = 0.95,
    #     actor_lr_start=1e-2, actor_lr_end=1e-3, actor_lr_decay=0.00005, actor_decay="exponential",
    #     critic_lr_start=1e-2, critic_lr_end=1e-3, critic_lr_decay=0.00005, critic_decay="exponential",
    #     gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0, 
    #     seed=None, verbose=False
    # )

    # agent = A3C(
    #     env_class=env_class,
    #     input_dim=modified_env.observation_shape[0], 
    #     action_dim=modified_env.action_space.n,
    #     actor_lr_start=1e-3, actor_lr_end=1e-4, actor_lr_decay=0.00005, actor_decay="exponential",
    #     critic_lr_start=1e-3, critic_lr_end=1e-4, critic_lr_decay=0.00005, critic_decay="exponential",
    #     gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0, 
    #     seed=None, verbose=False
    # )

    # agent = NStepA3C(
    #     env_class=env_class,
    #     input_dim=modified_env.observation_shape[0], 
    #     action_dim=modified_env.action_space.n,
    #     n_step=5,
    #     actor_lr_start=1e-3, actor_lr_end=1e-4, actor_lr_decay=0.00005, actor_decay="exponential",
    #     critic_lr_start=1e-3, critic_lr_end=1e-4, critic_lr_decay=0.00005, critic_decay="exponential",
    #     gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0, 
    #     seed=None, verbose=False
    # )

    # agent = A3CGAE(
    #     env_class=env_class,
    #     input_dim=modified_env.observation_shape[0], 
    #     action_dim=modified_env.action_space.n,
    #     n_step=5, lambd = 0.95,
    #     actor_lr_start=1e-3, actor_lr_end=1e-4, actor_lr_decay=0.00005, actor_decay="exponential",
    #     critic_lr_start=1e-3, critic_lr_end=1e-4, critic_lr_decay=0.00005, critic_decay="exponential",
    #     gamma=0.99, entropy_coef=0.0001, Huberbeta=1.0, 
    #     seed=None, verbose=False
    # )



    # %%

    max_episodes = 150
    agent.verbose = False
    agent.training()

    tic = time.time()

    # For other methods except A3C:
    episode_rewards, episode_steps = agent.train(max_episodes=max_episodes)

    # For A3C:
    # episode_rewards, episode_steps = agent.train(max_episodes=max_episodes, num_workers=8)

    toc = time.time()
    elapsed = toc - tic

    # %%

    all_states_vectors = torch.tensor(modified_env.state_to_vector, dtype=torch.float32, device=agent.device)

    # For other methods except Actor-Critic and A3C:
    # qtable = agent.policy_net(all_states_vectors).detach().cpu().numpy()

    # For other Actor-Critic methods:
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