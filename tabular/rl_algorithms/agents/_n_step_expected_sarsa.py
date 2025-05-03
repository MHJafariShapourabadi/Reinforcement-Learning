from . import Agent

import numpy as np

from tqdm import tqdm



class NStepExpectedSarsa(Agent):
    def __init__(self, env, learning_rate, gamma, n_step=1, learning_rate_dacay=0.0, initial_qtable=None, seed=None):
        super(NStepExpectedSarsa, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, learning_rate_dacay=learning_rate_dacay, initial_qtable=initial_qtable)
        self.n_step = n_step
        self.storage = [{'s':None, 'a':None, 'r':None} for _ in range(n_step + 1)]

    
    def update(self, step, T, epsilon):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        t = step + 1 - self.n_step
        if t >= 0:
            G = 0
            for i in range(min(t + self.n_step, T), t, -1):
                idx = i % (self.n_step + 1)
                r = self.storage[idx]['r']
                G = r + self.gamma * G

            if t + self.n_step < T:
                idx = (t + self.n_step) % (self.n_step + 1)
                s = self.storage[idx]['s']
                a = self.storage[idx]['a']
                action_size = int(np.sum(self.mask[s, :]).item())
                policy_probs = np.zeros(action_size)
                policy_probs.fill(epsilon / action_size)
                policy_probs[np.argmax(self.qtable[s, self.mask[s, :].astype("bool")])] = 1 - epsilon + (epsilon / action_size)
                G = G + (self.gamma ** self.n_step) * np.dot(policy_probs, self.qtable[s, self.mask[s, :].astype("bool")])

            idx = t % (self.n_step + 1)
            s = self.storage[idx]['s']
            a = self.storage[idx]['a']
            delta = G - self.qtable[s, a]

            q_update = self.qtable[s, a] + self.learning_rate * delta
            self.qtable[s, a] = q_update
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
                action = explorer.choose_action(
                        action_space=self.env.action_space, state=state, qtable=self.qtable, mask=self.mask
                    )
                step = 0
                idx = step % (self.n_step + 1)
                self.storage[idx]['s'] = state
                self.storage[idx]['a'] = action
                done = False
                T = float('inf')
                total_rewards = 0

                while True:
                    if step < T:

                        # Log all states and actions
                        all_states.append(state)
                        all_actions.append(action)

                        # Take the action (a) and observe the outcome state(s') and reward (r)
                        new_state, reward, terminated, truncated, info = self.env.step(action)

                        idx = (step + 1) % (self.n_step + 1)
                        self.storage[idx]['s'] = new_state
                        self.storage[idx]['r'] = reward

                        done = terminated or truncated
                        if done:
                            T = step + 1
                        else:
                            action = explorer.choose_action(
                            action_space=self.env.action_space, state=new_state, qtable=self.qtable, mask=self.mask
                            )
                            self.storage[idx]['a'] = action

                    self.update(step, T, explorer.epsilon)

                    total_rewards += reward
                    step += 1

                    # Our new state is state
                    state = new_state

                    if step == T + self.n_step - 1:
                        break

                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step

            qtables[run, :, :] = self.qtable

            self.update_policy(seed=seed)

        return rewards, steps, episodes, qtables, all_states, all_actions