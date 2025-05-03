from . import QLearning

import numpy as np

from tqdm import tqdm



class DynaQEnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.reset()
        

    def reset(self):
        self.P = {s: {a: {"n_all": 0} for a in range(self.n_actions)} for s in range(self.n_states)}
        self.visited_states = []
        self.state_visited_actions = {s: [] for s in range(self.n_states)}
        self.next_states_rewards = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        self.next_states_rewards_probs = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
    
    def update(self, state, action, new_state, reward):

        if state not in self.visited_states:
            self.visited_states.append(state)

        if action not in self.state_visited_actions[state]:
            self.state_visited_actions[state].append(action)

        key = (new_state, reward)
        if key in self.P[state][action]:
            self.P[state][action][key] = self.P[state][action][key] + 1
        else:
            self.P[state][action][key] = 1
        
        self.P[state][action]["n_all"] = self.P[state][action]["n_all"] + 1

        n_all = self.P[state][action]["n_all"]
        self.next_states_rewards[state][action] = []
        self.next_states_rewards_probs[state][action] = []
        for key, n in self.P[state][action].items():
            if key == "n_all":
                continue
            (s, r) = key
            self.next_states_rewards[state][action].append((s, r))
            self.next_states_rewards_probs[state][action].append(n / n_all)

    def __call__(self, state, action):

        next_state, reward = self.rng.choice(self.next_states_rewards[state][action], p=self.next_states_rewards_probs[state][action])

        return int(next_state), reward








class DynaQPlusEnvironmentModel(DynaQEnvironmentModel):

    def __call__(self, state, action):
        if action not in self.state_visited_actions[state]:
            next_state, reward = state, 0
        else:
            next_state, reward = self.rng.choice(self.next_states_rewards[state][action], p=self.next_states_rewards_probs[state][action])
        
        return int(next_state), reward
        










class DynaQ(QLearning):
    def __init__(self, env, learning_rate, gamma, n_planning, learning_rate_dacay=0.0, initial_qtable=None, seed=None):
        super(DynaQ, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, learning_rate_dacay=learning_rate_dacay, initial_qtable=initial_qtable)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n_planning= n_planning
        self.model = DynaQEnvironmentModel(self.state_size, self.action_size, seed)       


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

                while not done:
                    action = explorer.choose_action(
                        action_space=self.env.action_space, state=state, qtable=self.qtable, mask=self.mask
                    )

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = self.env.step(action)

                    done = terminated or truncated

                    self.update(
                        state, action, new_state, reward
                    )

                    self.model.update(state, action, new_state, reward)

                    for plan_step in range(self.n_planning):
                        s = self.rng.choice(self.model.visited_states)
                        a = self.rng.choice(self.model.state_visited_actions[s])
                        n_s, r = self.model(s, a)
                        self.update(s, a, n_s, r)

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










class DynaQPlus(DynaQ):
    def __init__(self, env, learning_rate, gamma, n_planning, reward_kappa, learning_rate_dacay=0.0, initial_qtable=None, seed=None):
        super(DynaQPlus, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, n_planning=n_planning, learning_rate_dacay=learning_rate_dacay, initial_qtable=initial_qtable)
        self.k = reward_kappa
        self.model = DynaQPlusEnvironmentModel(self.state_size, self.action_size, seed)
        self.reset_tau()

    
    def reset_tau(self):
        self.tau = np.zeros((self.state_size, self.action_size))

    def update_tau(self, state, action):
        self.tau = self.tau + 1
        self.tau[state, action] = 0

    def update_reward(self, state, action, reward):
        reward = reward + self.k * (self.tau[state, action] ** 0.5)
        return reward


    def train(self, explorer, total_episodes, n_runs, seed=None):
        rewards = np.zeros((total_episodes, n_runs))
        steps = np.zeros((total_episodes, n_runs))
        episodes = np.arange(total_episodes)
        qtables = np.zeros((n_runs, self.state_size, self.action_size))
        all_states = []
        all_actions = []

        for run in range(n_runs):  # Run several times to account for stochasticity
            self.reset_qtable(self.initial_qtable) # Reset the Q-table between runs
            self.reset_tau()
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

                while not done:
                    action = explorer.choose_action(
                        action_space=self.env.action_space, state=state, qtable=self.qtable, mask=self.mask
                    )

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    self.update_tau(state, action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = self.env.step(action)

                    done = terminated or truncated

                    self.update(
                        state, action, new_state, reward
                    )

                    self.model.update(state, action, new_state, reward)

                    for plan_step in range(self.n_planning):
                        s = self.rng.choice(self.model.visited_states)
                        a = self.env.action_space.sample(self.mask[s, :])
                        n_s, r = self.model(s, a)
                        r = self.update_reward(s, a, r)
                        self.update(s, a, n_s, r)


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