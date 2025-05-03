import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from ..wrappers import FrozenLakeVectorObservationWrapper

class FrozenLakeVectorObsevationEnvClass:
    def __init__(self, map_size, proba_frozen, max_episode_steps, is_slippery, episode_auto_restart, seed):
        self.map_size = map_size
        self.proba_frozen = proba_frozen
        self.max_episode_steps = max_episode_steps
        self.is_slippery = is_slippery
        self.episode_auto_restart = episode_auto_restart
        self.seed = seed

    def create_env(self, slip_epsilon=0.6, step_reward=-1, hole_reward=-10, goal_reward=10, active_neighbour=2):
        env = gym.make(
            "FrozenLake-v1",
            is_slippery=self.is_slippery,
            render_mode="rgb_array",
            desc=generate_random_map(size=self.map_size, p=self.proba_frozen, seed=self.seed),
            max_episode_steps= self.max_episode_steps,)

        modified_env = FrozenLakeVectorObservationWrapper(
            env=env,
            slip_epsilon=slip_epsilon, step_reward=step_reward, hole_reward=hole_reward, goal_reward=goal_reward,
            active_neighbour=active_neighbour,
            episode_auto_restart = self.episode_auto_restart,
            seed=self.seed)

        return modified_env
