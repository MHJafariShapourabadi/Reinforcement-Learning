from . import Agent

import numpy as np

from tqdm import tqdm



class PolicyIteration(Agent):
    def __init__(self, env, gamma, threshold1, threshold2, initial_qtable=None):
        super(PolicyIteration, self).__init__(env=env, learning_rate=None, gamma=gamma, initial_qtable=initial_qtable)
        if gamma == 1:
            raise ValueError("gamma should never be 1 for Policy Iteration because it may not converge, there may be policies that continue for ever")
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def update(self, state, action, P):
        """Update Q(s,a)"""
        q_update = 0
        for prob, next_state, reward, terminated in P[state][action]:
            q_update = q_update + prob * (reward + self.gamma * self.qtable[next_state, self.policy[next_state]]) 
        self.qtable[state, action] = q_update
        return q_update

    def train(self, total_episodes, n_runs=1, seed=None, **kwargs):
        rewards = np.zeros((total_episodes, n_runs))
        steps = np.zeros((total_episodes, n_runs))
        episodes = np.arange(total_episodes)
        qtables = np.zeros((n_runs, self.state_size, self.action_size))
        all_states = []
        all_actions = []

        for run in range(n_runs):  # Run several times to account for stochasticity
            self.reset_qtable(self.initial_qtable)  # Reset the Q-table between runs
            self.reset_policy()
            
            for episode in tqdm(
                episodes, desc=f"Run {run}/{n_runs} - Episodes", leave=False
            ):
                new_state = self.env.reset(seed=seed)[0]  # Reset the environment
                step = 0
                done = False
                total_rewards = 0

                old_qtable = self.qtable.copy()
                
                while True:
                    delta = 0

                    for state in range(self.state_size):
                        for action in range(self.action_size):

                            # Log all states and actions
                            all_states.append(state)
                            all_actions.append(action)

                            # Take the action (a) and observe the outcome state(s') and reward (r)
                            new_state, reward, terminated, truncated, info = self.env.step(self.policy[new_state])

                            old_qvalue = self.qtable[state, action]

                            self.update(
                                state, action, self.env.get_wrapper_attr('P') # env.unwrapped.P or env.get_wrapper_attr('P')
                            )

                            total_rewards += reward
                            step += 1

                            delta = max(delta, abs(old_qvalue - self.qtable[state, action]))


                    if delta < self.threshold1:
                        break

                if np.max(np.abs(self.qtable - old_qtable)) < self.threshold2:
                    break

                self.update_policy(seed=seed)


                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step
                                
            qtables[run, :, :] = self.qtable

        return rewards, steps, episodes, qtables, all_states, all_actions
    
