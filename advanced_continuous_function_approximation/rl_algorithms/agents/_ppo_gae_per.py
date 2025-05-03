import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
import numpy as np
import os
import time
import gc
# from collections import deque
from itertools import count








class PrioritizedReplayBuffer:
    def __init__(self, capacity, n_envs, state_dim, action_dim,
                alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3,
                beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
                epsilon=1e-5, seed=None):
        self.capacity = ((capacity // n_envs) + 1) * n_envs
        self.n_envs = n_envs
        self.size = 0
        self.pos = 0  # Pointer to the next position to overwrite
        self.alpha = alpha_start  # Controls how much prioritization is used
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_increment = alpha_increment
        self.alpha_steps = 0
        self.beta = beta_start  # Controls how much importance sampling is used
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_increment = beta_increment
        self.beta_steps = 0
        self.epsilon = epsilon # Epsilon added to priorities for stability
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        # Pre-allocate memory for transitions
        self.I = torch.zeros((self.capacity,), dtype=torch.float32)
        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32)
        self.values = torch.zeros((self.capacity,), dtype=torch.float32)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.int64)
        self.log_probs = torch.zeros((self.capacity,), dtype=torch.float32)
        self.returns = torch.zeros((self.capacity,), dtype=torch.float32)

        # Initialize priorities (start with max priority for new transitions)
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float32)

    def reset(self):
      self.alpha = self.alpha_start
      self.beta = self.beta_start
      self.alpha_steps = 0
      self.beta_steps = 0

    def update_alpha(self, steps=None):
      self.alpha_steps += 1
      steps = self.alpha_steps if steps is None else steps
      self.alpha = min(self.alpha_end, self.alpha_start + self.alpha_increment * steps)

    def update_beta(self, steps=None):
      self.beta_steps += 1
      steps = self.beta_steps if steps is None else steps
      self.beta = min(self.beta_end, self.beta_start + self.beta_increment * steps)

    def push(self, I, states, values, actions, log_probs, returns):
        """Store a transition in the buffer."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        self.I[self.pos : self.pos + self.n_envs] = I
        self.states[self.pos : self.pos + self.n_envs] = states
        self.values[self.pos : self.pos + self.n_envs] = values
        self.actions[self.pos : self.pos + self.n_envs] = actions
        self.log_probs[self.pos : self.pos + self.n_envs] = log_probs
        self.returns[self.pos : self.pos + self.n_envs] = returns
        self.priorities[self.pos : self.pos + self.n_envs] = max_priority

        self.pos = (self.pos + self.n_envs) % self.capacity
        self.size = min(self.size + self.n_envs, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of transitions based on priorities."""
        if self.size == 0:
            raise ValueError("Replay buffer is empty!")

        # Calculate probabilities using priorities
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices based on probabilities
        indices = self.rng.choice(self.size, size=batch_size, p=sampling_probabilities.cpu().numpy(), replace=False,)

        # Compute importance sampling weights
        total_prob = sampling_probabilities[indices]
        weights = (self.size * total_prob) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        return (
            self.I[indices],
            self.states[indices],
            self.values[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.returns[indices],
            indices,  # Return sampled indices to update priorities later
            weights,
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities for the sampled transitions."""
        # New priorities are proportional to the absolute TD error
        self.priorities[indices] = torch.abs(td_errors) + self.epsilon  # Add epsilon for stability

    def __len__(self):
        return self.size










class SpecialDeque:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.list = [None] * maxlen
        # self.last_item = None
        self.start_idx = 0
        self.end_idx = 0
        self.size = 0
        self.full = False
        self.empty = True

    def append(self, value):
        self.list[self.end_idx] = value
        self.size = min(self.size + 1, self.maxlen)
        self.end_idx = (self.end_idx + 1) % self.maxlen
        self.empty = False
        if self.end_idx == self.start_idx:
            self.full = True
        if self.full:
            self.start_idx = self.end_idx

    # def append_last(self, value):
    #     self.last_item = value

    def popleft(self):
        if self.empty:
            raise Exception("Poping empty deque.")
        value = self.list[self.start_idx]
        self.size = max(self.size - 1, 0)
        self.start_idx = (self.start_idx + 1) % self.maxlen
        self.full = False
        if self.start_idx == self.end_idx:
            self.empty = True
        return value

    def __getitem__(self, index):
        if index >= self.maxlen or index < - self.maxlen:
            raise IndexError("Index out of deque range.")
        index = (index + self.start_idx) % self.maxlen
        return self.list[index]
        # if index == self.maxlen:
        #     return self.last_item
        # else:
        #     if index > self.maxlen or index < - self.maxlen:
        #         raise IndexError("Index out of deque range.")
        #     index = (index + self.start_idx) % self.maxlen
        #     return self.list[index]

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{self.list}"









class Actor(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(Actor, self).__init__()

        self.negative_slope = 0.01

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, num_actions)
        # self.fc_std = nn.Linear(128, num_actions)

        # Initialize weights using He Initialization
        self._initialize_weights()
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope) + x
        x = F.leaky_relu(self.fc3(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc4(x), negative_slope=self.negative_slope) + x
        means = self.fc_mean(x)
        # means = torch.tanh(means)
        # stds = self.fc_std(x)
        # stds = torch.sigmoid(stds)
        return means

    def _initialize_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                if 'fc_mean' in name:
                    nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                    # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                    # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('linear'))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                # elif 'fc_std' in name:
                #     nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                #     # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
                #     # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('linear'))
                #     if layer.bias is not None:
                #         nn.init.zeros_(layer.bias)
                else:
                    # nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                    nn.init.kaiming_normal_(layer.weight, a=self.negative_slope, nonlinearity="leaky_relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # def _initialize_weights(self):
    #     """Initialize weights of the layers using He initialization."""
    #     for layer in self.modules():
    #         if isinstance(layer, nn.Linear):
    #             nn.init.kaiming_normal_(layer.weight, a=self.negative_slope)
    #             # nn.init.zeros_(layer.weight)
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)








class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.negative_slope = 0.01

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_value = nn.Linear(128, 1)

        # Initialize weights using He Initialization
        self._initialize_weights()
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope) + x
        x = F.leaky_relu(self.fc3(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc4(x), negative_slope=self.negative_slope) + x
        value = self.fc_value(x).squeeze(-1)  # Ensure scalar output
        return value

    def _initialize_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                if 'fc_value' in name:
                    # nn.init.xavier_uniform_(layer.weight, gain=1.0)  # gain = 1.0 for linear activation
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('linear'))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:
                    nn.init.kaiming_normal_(layer.weight, a=self.negative_slope, nonlinearity="leaky_relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # def _initialize_weights(self):
    #     """Initialize weights of the layers using He initialization."""
    #     for layer in self.modules():
    #         if isinstance(layer, nn.Linear):
    #             nn.init.kaiming_normal_(layer.weight, a=self.negative_slope)
    #             # nn.init.zeros_(layer.weight)
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)








class PPOGAEPER:
    def __init__(self, env, n_step, lambd = 0.9,
    actor_weight_decay = 0.1, actor_lr_start=1e-3, actor_lr_end=1e-5, actor_lr_decay=0.001, actor_decay="linear",
    critic_weight_decay = 0.01, critic_lr_start=1e-3, critic_lr_end=1e-5, critic_lr_decay=0.001, critic_decay="linear",
    n_updates_per_iter=5, buffer_size=256, batch_size=64,
    alpha_start=0.8, alpha_end=1.0, alpha_increment=1e-3,
    beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
    r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.0001, std=1.0,
    gamma=0.99, max_grad_norm=0.5, target_kl = 0.02, seed=None, verbose=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_lower_bound = torch.tensor(self.env.action_space.low).to(self.device)
        self.action_upper_bound = torch.tensor(self.env.action_space.high).to(self.device)
        self.n_step = n_step
        self.lambd = lambd

        # self.std_min = std_min
        # self.std_bound = std_bound
        self.std = std

        # Validate the actor decay type
        if actor_decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.actor_lr_start = actor_lr_start 
        self.actor_lr_end = actor_lr_end 
        self.actor_lr_decay = actor_lr_decay 
        self.actor_decay = actor_decay

        # Validate the critic decay type
        if critic_decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.critic_lr_start = critic_lr_start 
        self.critic_lr_end = critic_lr_end 
        self.critic_lr_decay = critic_lr_decay 
        self.critic_decay = critic_decay

        self.n_updates_per_iter = n_updates_per_iter
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.r_clip_epsilon = r_clip_epsilon
        self.v_clip_epsilon = v_clip_epsilon

        self.entropy_coef = entropy_coef
        self.gamma = gamma

        self.seed = seed if seed is not None else int(time.time())
        torch.manual_seed(seed=self.seed)
        self.rng = np.random.default_rng(seed=self.seed)

        self.verbose = verbose
        self._is_training = True

        # actor network
        self.actor= Actor(self.state_dim, self.action_dim).to(self.device)
        # critic network
        self.critic = Critic(self.state_dim).to(self.device)
        
        # Optimizer for actor network
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_start)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr_start, weight_decay=actor_weight_decay, amsgrad=True)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=self.update_actor_learning_rate)

        # Optimizer for critic network
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_start)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr_start, weight_decay=critic_weight_decay, amsgrad=True)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=self.update_critic_learning_rate)

        self.max_grad_norm = max_grad_norm

        self.target_kl = target_kl                           # KL Divergence threshold

        # Prioritized Replay Buffer
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, n_envs=self.env.n_envs,
                                   state_dim=self.state_dim, action_dim=self.action_dim,
                                   alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3,
                                   beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
                                   seed=self.seed)

        self.n_updates = None
        self.steps = None

        self.training()

        self.greedy_actions(False)

    @property
    def is_training(self):
        return self._is_training

    def training(self):
        self._is_training = True
        torch.manual_seed(seed=self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        self.actor.train()
        self.critic.train()

    def evaluating(self, seed=None):
        self._is_training = False
        seed = seed if seed is not None else self.seed
        torch.manual_seed(seed=seed)
        self.rng = np.random.default_rng(seed=seed)
        self.actor.eval()
        self.critic.eval()

    def greedy_actions(self, greedy=True):
        self._greedy_actions = greedy


    @torch.no_grad()
    def select_action(self, state, info):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.actor.eval()
            mean = self.actor(state)
            mean = mean.squeeze(0)
            mean, std = self.bound(mean)
            if not self._greedy_actions:
                action_dist = Independent(Normal(mean, std), 1)
                action = action_dist.sample()
                # action = torch.clamp(action, self.action_lower_bound, self.action_upper_bound)
            else:
                action = mean
        # print("================================")
        # print(f"action dist batch shape: {action_dist.batch_shape}")
        # print(f"action dist eveny shape: {action_dist.event_shape}")
        # print(f"mean: {mean}")
        # print(f"std: {std}")
        # print(f"action: {action}")
        return action.detach().cpu().numpy()

    def train(self, max_episodes):
        self.actor.train()
        self.critic.train()
        episode_rewards = []
        episode_steps = []
        avg_means = 0
        avg_stds = 0
        actor_loss_avg = 0
        critic_loss_avg = 0
        avg_beta = 0.2
        approx_kl = 0
        self.n_updates = 0 if self.n_updates is None else self.n_updates
        self.steps = 0 if self.steps is None else self.steps

        n_states = SpecialDeque(maxlen=self.n_step)
        n_values = SpecialDeque(maxlen=self.n_step)
        n_next_values = SpecialDeque(maxlen=self.n_step)
        n_actions = SpecialDeque(maxlen=self.n_step)
        n_log_probs = SpecialDeque(maxlen=self.n_step)
        n_rewards = SpecialDeque(maxlen=self.n_step)
        n_I = SpecialDeque(maxlen=self.n_step)
        n_terminateds = SpecialDeque(maxlen=self.n_step)
        n_dones = SpecialDeque(maxlen=self.n_step)

        episodes = 0
        I, states, infos = self.env.reset()

        I = torch.tensor(I, dtype=torch.float32).to(self.device)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        for t in count():
            with torch.no_grad():
                means = self.actor(states)
                means, stds = self.bound(means)
                states_value = self.critic(states)
                # print("=====================")
                # print(f"states: {states}")
                # print(f"means: {means}")
                # print(f"stds: {stds}")
                # print(f"states value: {states_value}")
                actions_dist = Independent(Normal(means, stds), 1)
                actions = actions_dist.sample().detach()
                # actions = torch.clamp(actions, self.action_lower_bound, self.action_upper_bound)
                log_probs = actions_dist.log_prob(actions)
                # print(f"action lower bound: {self.action_lower_bound}")
                # print(f"action upper bound: {self.action_upper_bound}")
                # print(f"actions: {actions}")
                # print(f"log probs: {log_probs}")

                next_I, next_states, rewards, terminateds, truncateds, next_infos = self.env.step(actions.cpu().numpy())

                dones = np.logical_or(terminateds, truncateds)

                next_I = torch.tensor(next_I, dtype=torch.float32).to(self.device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                terminateds = torch.tensor(terminateds, dtype=torch.float32).to(self.device)
                dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    next_states_value = self.critic(next_states).detach()

                n_states.append(states)
                n_values.append(states_value.detach())
                n_next_values.append(next_states_value.detach())
                n_actions.append(actions.detach())
                n_log_probs.append(log_probs.detach())
                n_rewards.append(rewards)
                n_I.append(I)
                n_terminateds.append(terminateds)
                n_dones.append(dones)

                I = next_I
                states = next_states
                infos = next_infos
                self.steps += 1

            if t >= self.n_step - 1:
                advantages = 0.0
                for i in reversed(range(self.n_step)):
                    td_errors = n_rewards[i] + self.gamma * n_next_values[i].detach() * (1.0 - n_terminateds[i]) - n_values[i].detach()
                    advantages = td_errors + (1.0 - n_dones[i]) * (self.gamma * self.lambd) * advantages
                targets = advantages + n_values[0].detach().clone()

                self.buffer.push(n_I[0].detach().cpu(), n_states[0].detach().cpu(), n_values[0].detach().cpu(), n_actions[0].detach().cpu(), n_log_probs[0].detach().cpu(), targets.detach().cpu())
                
                if len(self.buffer) > self.batch_size:
                    for _ in range(self.n_updates_per_iter):
                        self.n_updates += 1
                        # Sample from replay buffer with PER
                        mb_I, mb_states, mb_states_value_old, mb_actions, mb_log_probs_old, mb_targets, indices, weights = self.buffer.sample(self.batch_size)
                        mb_I, mb_states, mb_states_value_old, mb_actions, mb_log_probs_old, mb_targets, weights = mb_I.to(self.device), mb_states.to(self.device), mb_states_value_old.to(self.device), mb_actions.to(self.device), mb_log_probs_old.to(self.device), mb_targets.to(self.device), weights.to(self.device)

                        mb_states_value = self.critic(mb_states)
                        mb_means = self.actor(mb_states)
                        mb_means, mb_stds = self.bound(mb_means)
                        mb_actions_dist = Independent(Normal(mb_means, mb_stds), 1)
                        mb_log_probs = mb_actions_dist.log_prob(mb_actions)
                        mb_entropies = mb_actions_dist.entropy()

                        # Unclipped value loss:
                        vf_loss_unclipped = (mb_targets - mb_states_value) ** 2
                        # Clipped value prediction:
                        clipped_values = mb_states_value_old + torch.clamp(mb_states_value - mb_states_value_old, -self.v_clip_epsilon, self.v_clip_epsilon)
                        vf_loss_clipped = (mb_targets - clipped_values) ** 2
                        # Final value function loss:
                        critic_loss = 0.5 * torch.mean(weights * torch.max(vf_loss_unclipped, vf_loss_clipped))
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()
                        self.critic_scheduler.step()

                        # Compute ratios.
                        mb_advantages = mb_targets - mb_states_value_old.detach()
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-10)
                        logratios = mb_log_probs - mb_log_probs_old
                        ratios = torch.exp(logratios)
                        approx_kl = ((ratios - 1) - logratios).mean() # Approximating KL Divergence
                        surr1 = ratios * mb_advantages
                        surr2 = torch.clamp(ratios, 1 - self.r_clip_epsilon, 1 + self.r_clip_epsilon) * mb_advantages
                        surr = torch.min(surr1, surr2)
                        actor_loss = (- torch.pow(self.gamma, mb_I) * weights * surr - self.entropy_coef * mb_entropies).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()
                        self.actor_scheduler.step()

                        # Update priorities in replay buffer
                        self.buffer.update_priorities(indices, mb_advantages.detach().cpu().squeeze())

                        # Update beta for PER
                        self.buffer.update_alpha()
                        self.buffer.update_beta()

                        critic_loss_avg = (1 - avg_beta) * critic_loss_avg + avg_beta * critic_loss.item()
                        actor_loss_avg = (1 - avg_beta) * actor_loss_avg + avg_beta * actor_loss.item()

                        if approx_kl > self.target_kl:
                            break # if kl aboves threshold

            with torch.no_grad():
                avg_means = (1 - avg_beta) * avg_means + avg_beta * means.mean(dim=0).detach().cpu().numpy()
                avg_stds = (1 - avg_beta) * avg_stds + avg_beta * stds.mean(dim=0).detach().cpu().numpy()

            for worker_id in range(len(infos)):
                info = infos[worker_id]
                if info['done']:
                    episodes += 1
                    I, states, new_info = self.env.reset(worker_id)
                    I = torch.tensor(I, dtype=torch.float32).to(self.device)
                    states = torch.tensor(states, dtype=torch.float32).to(self.device)
                    infos[worker_id] = new_info
                    episode_reward = info['episode_reward']
                    steps_in_episode = info['episode_steps']
                    episode_rewards.append(episode_reward)
                    episode_steps.append(steps_in_episode)
                    print("===========================================")
                    print(f"Episodes {episodes}, Steps {self.steps}: N-updates: {self.n_updates}, Approx kl: {approx_kl : .5f}")
                    print(f"actor avg loss: {actor_loss_avg : .5f}, critic avg loss:{critic_loss_avg : .5f}")
                    print(f"Reward: {episode_reward : .5f}, Steps: {steps_in_episode}")
                    print(f"avg means: {avg_means}")
                    print(f"avg stds: {avg_stds}")
                    print(f"actor lr: {self.actor_scheduler.get_last_lr()[0] : .5f}, critic lr: {self.critic_scheduler.get_last_lr()[0] : .5f}")

            if self.steps % 10000 == 0:
                self.save_agent(steps=self.steps)

            if episodes >= max_episodes:
                gc.collect()
                break

        return episode_rewards, episode_steps

    def bound(self, means):
        # return means * self.action_upper_bound, torch.full_like(means, self.std).to(self.device)
        return means, torch.full_like(means, self.std).to(self.device)

    def save_agent(self, steps='last', model_dir="./models"):
        # Set the directory for saving models
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Saving models
        print(f"... Saving model to {model_dir} after {steps} steps ...")

        agent = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'steps': self.steps,
            'n_updates': self.n_updates
        }

        file_path = os.path.join(model_dir, f"ppo_{type(self.env.dummy_env.unwrapped).__name__}_{steps}_steps.pth")
        torch.save(agent, file_path)

    def load_agent(self, path, reset_lr=False):
        # Loading models
        print(f"... Loading model from {path} ...")
        agent = torch.load(path, weights_only=False)

        self.actor.load_state_dict(agent['actor_state_dict'])
        self.actor_optimizer.load_state_dict(agent['actor_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(agent['actor_scheduler_state_dict'])

        self.critic.load_state_dict(agent['critic_state_dict'])
        self.critic_optimizer.load_state_dict(agent['critic_optimizer_state_dict'])
        self.critic_scheduler.load_state_dict(agent['critic_scheduler_state_dict'])

        if reset_lr:
            # For the actor optimizer:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.actor_lr_start
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=self.update_actor_learning_rate)

            # For the critic optimizer:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.critic_lr_start
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=self.update_critic_learning_rate)

        self.steps = agent['steps']
        self.n_updates = agent['n_updates']


    def update_actor_learning_rate(self, step):
        """
        Updates the learning rate value using the specified decay type.

        Args:
            steps (int): Current step count.

        Behavior:
            - Linear Decay: Epsilon decreases linearly with each step.
            - Exponential Decay: Epsilon decreases exponentially with each step.
            - Reciprocal Decay: Epsilon decreases inversely with each step.
        """
        if self.actor_decay == "linear":
            # Linearly decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.actor_lr_end,
                self.actor_lr_start - self.actor_lr_decay * step 
            )
        elif self.actor_decay == "exponential":
            # Exponentially decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.actor_lr_end,
                self.actor_lr_start * np.exp(-self.actor_lr_decay * step)
            )
        elif self.actor_decay == "reciprocal":
            # Inversely decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.actor_lr_end,
                self.actor_lr_start / (1 + self.actor_lr_decay * step)
            )
        return self.lr / self.actor_lr_start

    def update_critic_learning_rate(self, step):
        """
        Updates the learning rate value using the specified decay type.

        Args:
            steps (int): Current step count.

        Behavior:
            - Linear Decay: Epsilon decreases linearly with each step.
            - Exponential Decay: Epsilon decreases exponentially with each step.
            - Reciprocal Decay: Epsilon decreases inversely with each step.
        """
        if self.critic_decay == "linear":
            # Linearly decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.critic_lr_end,
                self.critic_lr_start - self.critic_lr_decay * step 
            )
        elif self.critic_decay == "exponential":
            # Exponentially decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.critic_lr_end,
                self.critic_lr_start * np.exp(-self.critic_lr_decay * step)
            )
        elif self.critic_decay == "reciprocal":
            # Inversely decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.critic_lr_end,
                self.critic_lr_start / (1 + self.critic_lr_decay * step)
            )
        return self.lr / self.critic_lr_start

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal, Independent

# # mean = torch.tensor([0,-1,2,0]).type(torch.float32)
# # std = torch.tensor([2,3,1,5]).type(torch.float32)

# mean = torch.tensor([[0,-1,2,0],[1,2,-3,0]]).type(torch.float32)
# std = torch.tensor([[2,3,1,5],[1,2,0.2,0.1]]).type(torch.float32)

# dist1 = Normal(mean, std)

# print(dist1.batch_shape)
# print(dist1.event_shape)

# raw_action = dist1.rsample()  # for reparameterization trick (PPO needs this)

# print(dist1.log_prob(raw_action))
# print(raw_action)

# dist2 = Independent(Normal(mean, std), 1)

# print(dist2.batch_shape)
# print(dist2.event_shape)

# raw_action = dist2.rsample()  # for reparameterization trick (PPO needs this)

# print(dist2.log_prob(raw_action))
# print(raw_action)

# import torch
# import numpy as np
# import gymnasium as gym
# from gymnasium.envs.classic_control import PendulumEnv
# low = torch.tensor(PendulumEnv().action_space.low)
# high = torch.tensor(PendulumEnv().action_space.high)
# print(low)
# print(high)
# print(torch.clamp(torch.tensor(5), low, high))
# from gymnasium.envs.box2d import BipedalWalker
# low = torch.tensor(BipedalWalker().action_space.low)
# high = torch.tensor(BipedalWalker().action_space.high)
# print(low)
# print(high)
# print(torch.clamp(torch.tensor([0.2,5,-3,-0.1]), low, high))


# import torch
# from torch.distributions import Normal, Independent

# means = torch.tensor([
#     [1,],
#     [0,]
# ], dtype=torch.float32)

# # std = torch.tensor(0.5, dtype=torch.float32)
# std = 0.5

# dist = Independent(Normal(means, std), 1)

# print(dist.sample())
# print(dist.batch_shape)
# print(dist.event_shape)