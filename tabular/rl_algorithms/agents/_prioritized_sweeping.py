from ..utils import MaxPriorityQueue
from . import QLearning

import numpy as np


from tqdm import tqdm



class PrioritizedSweepingEnvironmentModel:
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
        self.previous_states_actions = {ns: [] for ns in range(self.n_states)}

    

    def update(self, state, action, next_state, reward):

        if state not in self.visited_states:
            self.visited_states.append(state)

        if action not in self.state_visited_actions[state]:
            self.state_visited_actions[state].append(action)

        if (state, action) not in self.previous_states_actions[next_state]:
            self.previous_states_actions[next_state].append((state, action))

        key = (next_state, reward)
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
            prob = n / n_all
            self.next_states_rewards[state][action].append((s, r, prob))
            self.next_states_rewards_probs[state][action].append(prob)


    def __call__(self, state, action):

        next_state, reward, prob = self.rng.choice(self.next_states_rewards[state][action], p=self.next_states_rewards_probs[state][action])

        return int(next_state), reward, prob

    
    def get_next_state_reward_pairs(self, state, action):

        return self.next_states_rewards[state][action]

    
    def get_transition_reward_probability(self, state, action ,next_state):
        reward = 0
        prob = 0
        for ns, r, p in self.get_next_state_reward_pairs(state, action):
            if ns == next_state:
                reward = reward + p * r
                prob = prob + p
        reward = reward / prob

        return reward, prob

    def get_previous_state_action_pairs(self, next_state):
        return self.previous_states_actions[next_state]









class PrioritizedSweeping(QLearning): # # It doesn't work well with negative rewards
    def __init__(self, env, learning_rate, gamma, n_planning, priority_threshold=0.001, expected_update=False, learning_rate_dacay=0.0, initial_qtable=None, seed=None):
        super(PrioritizedSweeping, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, learning_rate_dacay=learning_rate_dacay, initial_qtable=initial_qtable)
        self.seed = seed
        self.n_planning= n_planning
        self.priority_threshold = priority_threshold
        self.expected_update_flag = expected_update
        self.model = PrioritizedSweepingEnvironmentModel(self.state_size, self.action_size, seed)
        self.priority_queue = MaxPriorityQueue()
        

    def sample_update(self, state, action, new_state, reward):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, self.mask[new_state, :].astype("bool")])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        self.qtable[state, action] = q_update
        return q_update

    def expected_update(self, state, action):
        """Update Q(s,a)"""
        q_update = 0
        for new_state, reward, prob in self.model.get_next_state_reward_pairs(state, action):
             q_update = q_update + prob * (reward + self.gamma * np.max(self.qtable[new_state, self.mask[new_state, :].astype("bool")])) 
        self.qtable[state, action] = q_update
        return q_update

    def compute_sample_priority(self, state, action, new_state, reward, prob):
        priority = np.abs(
            reward 
            + self.gamma * np.max(self.qtable[new_state, self.mask[new_state, :].astype("bool")])
            - self.qtable[state, action]
            ) * prob
        return priority

    def compute_expected_priority(self, state, action):
        q_update = 0
        for new_state, reward, prob in self.model.get_next_state_reward_pairs(state, action):
            q_update = q_update + prob * (reward + self.gamma * np.max(self.qtable[new_state, self.mask[new_state, :].astype("bool")]))
        priority = np.abs(self.qtable[state, action] - q_update)
        return priority
        

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

                    self.model.update(state, action, new_state, reward)

                    if self.expected_update_flag:
                        
                        priority = self.compute_expected_priority(state, action)

                    else:

                        average_reward, prob = self.model.get_transition_reward_probability(state, action, new_state)

                        priority = self.compute_sample_priority(state, action, new_state, average_reward, prob)

                    if priority > self.priority_threshold:

                        self.priority_queue.put(priority, (state, action))

                    for plan_step in range(self.n_planning):
                        
                        if self.priority_queue.empty():
                            break

                        prior, (s, a) = self.priority_queue.get()

                        if self.expected_update_flag:
                            
                            self.expected_update(s, a)

                        else:

                            n_s, r, prob = self.model(s, a)

                            self.sample_update(s, a, n_s, r)
                        
                        for s_, a_ in self.model.get_previous_state_action_pairs(s):

                            if self.expected_update_flag:
                                
                                prior_ = self.compute_expected_priority(s_, a_)

                            else:

                                avg_r_, prob_ = self.model.get_transition_reward_probability(s_, a_, s)

                                prior_ = self.compute_sample_priority(s_, a_, s, avg_r_, prob_)

                            if prior_ > self.priority_threshold:

                                self.priority_queue.put(prior_, (s_, a_))

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

    