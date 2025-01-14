from . import Agent

import numpy as np

from tqdm import tqdm





class RTDP(Agent):
    def __init__(self, env, gamma, initial_qtable=None):
        super(RTDP, self).__init__(env=env, learning_rate=None, gamma=gamma, initial_qtable=initial_qtable)

    def update(self, state, action, P):
        """Update Q(s,a)"""
        q_update = 0
        for prob, next_state, reward, terminated in P[state][action]:
             q_update = q_update + prob * (reward + self.gamma * np.max(self.qtable[next_state, self.mask[next_state, :].astype("bool")])) 
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
            explorer.reset_epsilon()

            for episode in tqdm(
                episodes, desc=f"Run {run}/{n_runs} - Episodes", leave=False
            ):
                explorer.decay_epsilon(episode)
                
                state = self.env.reset(seed=seed)[0]  # Reset the environment
                step = 0
                done = False
                total_rewards = 0

                while not done:
                    action = explorer.choose_action(
                        action_space=self.env.action_space, state=state, qtable=self.qtable
                    )

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = self.env.step(action)

                    done = terminated or truncated

                    self.update(
                        state, action, self.env.get_wrapper_attr('P') # env.unwrapped.P or env.get_wrapper_attr('P')
                    )

                    total_rewards += reward
                    step += 1

                    # Our new state is state
                    state = new_state

                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step
            
            qtables[run, :, :] = self.qtable

            self.update_policy(seed=seed)

        return rewards, steps, episodes, qtables, all_states, all_actions