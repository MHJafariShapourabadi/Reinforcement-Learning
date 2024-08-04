from . import Agent

import numpy as np

from tqdm import tqdm



class Sarsa(Agent):
    def __init__(self, env, learning_rate, gamma, initial_qtable=None):
        super(Sarsa, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, initial_qtable=initial_qtable)

    def update(self, state, action, reward, new_state, new_action):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * self.qtable[new_state, new_action]
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
                
                action = explorer.choose_action(
                        action_space=self.env.action_space, state=state, qtable=self.qtable, mask=self.mask
                    )

                while not done:
                    
                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = self.env.step(action)

                    done = terminated or truncated

                    new_action = explorer.choose_action(
                        action_space=self.env.action_space, state=new_state, qtable=self.qtable, mask=self.mask
                    )

                    self.update(
                        state, action, reward, new_state, new_action
                    )

                    total_rewards += reward
                    step += 1

                    # Our new state is state
                    state = new_state
                    # Our new action is action
                    action = new_action

                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step

            qtables[run, :, :] = self.qtable

            self.update_policy(seed=seed)

        return rewards, steps, episodes, qtables, all_states, all_actions