from . import Agent

import numpy as np

from tqdm import tqdm



class MonteCarloOnPolicy(Agent):
    def __init__(self, env, learning_rate, gamma, first_vist=False, initial_qtable=None):
        super(MonteCarloOnPolicy, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, initial_qtable=initial_qtable)
        self.first_visit = first_vist
        

    def update(self, state, action, Return):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            Return
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        self.qtable[state, action] = q_update
        return q_update


    def train(self, explorer, total_episodes, n_runs, seed=None):
        rewards = np.zeros((total_episodes, n_runs))
        steps = np.zeros((total_episodes, n_runs))
        episodes = np.arange(total_episodes)
        qtables = np.zeros((n_runs, self.state_size, self.action_size))
        all_states = []
        all_actions = []

        for run in range(n_runs):  # Run several times to account for stochasticity
            self.reset_qtable(self.initial_qtable)  # Reset the Q-table between runs
            self.reset_policy()
            self.reset_learning_rate()
            explorer.reset_epsilon()

            for episode in tqdm(
                episodes, desc=f"Run {run}/{n_runs} - Episodes", leave=False
            ):
                self.decay_learning_rate(episode)
                explorer.decay_epsilon(episode)

                state = self.env.reset(seed=seed)[0]  # Reset the environment
                step = 0
                done = False
                total_rewards = 0

                episode_buffer = []
                rewards_buffer = []

                while not done:
                    action = explorer.choose_action(
                        action_space=self.env.action_space, state=state, qtable=self.qtable, mask=self.mask
                    )

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = self.env.step(action)

                    episode_buffer.append((state, action))
                    rewards_buffer.append(reward)

                    done = terminated or truncated

                    total_rewards += reward
                    step += 1

                    # Our new state is state
                    state = new_state

                Return = 0
                for t in range(len(episode_buffer)-1, -1, -1):
                    state, action = episode_buffer[t]
                    reward = rewards_buffer[t]
                    Return = self.gamma * Return + reward
                    if self.first_visit and ((state, action) in episode_buffer[:t]):
                        continue
                    self.update(
                        state, action, Return
                    )


                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step

            qtables[run, :, :] = self.qtable

            self.update_policy(seed=seed)

        return rewards, steps, episodes, qtables, all_states, all_actions