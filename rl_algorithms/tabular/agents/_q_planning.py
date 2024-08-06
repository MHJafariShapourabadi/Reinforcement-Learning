from . import Agent
from ..exploration import Greedy

import numpy as np

from tqdm import tqdm


class QPlanningModel:
    def __init__(self, P, seed=None):
        self.rng = np.random.default_rng(seed)
        self.P = P
    
    def __call__(self, state, action):
        transitions = self.P[state][action] 
        probs = [t[0] for t in transitions]
        next_states_rewards = [(t[1], t[2]) for t in transitions]
        new_state, reward = self.rng.choice(next_states_rewards, p=probs)
        new_state = int(new_state)
        return new_state, reward


class QPlanning(Agent):
    def __init__(self, env, learning_rate, gamma, learning_rate_dacay=0.0, initial_qtable=None):
        super(QPlanning, self).__init__(env=env, learning_rate=learning_rate, gamma=gamma, learning_rate_dacay=learning_rate_dacay, initial_qtable=initial_qtable)
        self.threshold = threshold
        self.n_planning = n_planning
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model = QPlanningModel(P=self.env.get_wrapper_attr('P'), seed=seed) # env.unwrapped.P or env.get_wrapper_attr('P')


    def update(self, state, action, new_state, reward):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, self.mask[new_state, :].astype("bool")])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
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
            self.reset_learning_rate()
            explorer = Greedy(seed=seed)

            for episode in tqdm(
                episodes, desc=f"Run {run}/{n_runs} - Episodes", leave=False
            ):
                self.decay_learning_rate(episode)
                
                new_state = self.env.reset(seed=seed)[0]  # Reset the environment
                step = 0
                done = False
                total_rewards = 0
                
                Delta = 0

                for n in range(self.n_planning):
                    state = self.rng.choice(self.state_size)
                    action = self.rng.choice(self.action_size)

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)
                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, new_reward, terminated, truncated, info = self.env.step(explorer.choose_action(action_space=self.env.action_space, state=new_state, qtable=self.qtable, mask=self.mask))
                    
                    old_qvalue = self.qtable[state, action]
                    
                    next_state, reward = self.model(state, action)

                    self.update(
                    state, action, next_state, reward
                    )

                    total_rewards += reward
                    step += 1

                    Delta = max(Delta, abs(old_qvalue - self.qtable[state, action]))

                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step
                
                if Delta < self.threshold:
                    break

            qtables[run, :, :] = self.qtable

            self.update_policy(seed=seed)

        return rewards, steps, episodes, qtables, all_states, all_actions