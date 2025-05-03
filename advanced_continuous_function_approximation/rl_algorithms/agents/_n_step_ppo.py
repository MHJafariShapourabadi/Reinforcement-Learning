import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import time
import gc
# from collections import deque
from itertools import count








class ReplayBuffer:
    def __init__(self, capacity, n_envs, state_dim, seed=None):
        self.capacity = ((capacity // n_envs) + 1) * n_envs
        self.n_envs = n_envs
        self.size = 0
        self.pos = 0  # Pointer to the next position to overwrite
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        # Pre-allocate memory for transitions
        self.I = torch.zeros((self.capacity,), dtype=torch.float32)
        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32)
        self.values = torch.zeros((self.capacity,), dtype=torch.float32)
        self.actions = torch.zeros((self.capacity,), dtype=torch.int64)
        self.log_probs = torch.zeros((self.capacity,), dtype=torch.float32)
        self.returns = torch.zeros((self.capacity,), dtype=torch.float32)

    def push(self, I, states, values, actions, log_probs, returns):
        """Store a transition in the buffer."""
        self.I[self.pos : self.pos + self.n_envs] = I
        self.states[self.pos : self.pos + self.n_envs] = states
        self.values[self.pos : self.pos + self.n_envs] = values
        self.actions[self.pos : self.pos + self.n_envs] = actions
        self.log_probs[self.pos : self.pos + self.n_envs] = log_probs
        self.returns[self.pos : self.pos + self.n_envs] = returns

        self.pos = (self.pos + self.n_envs) % self.capacity
        self.size = min(self.size + self.n_envs, self.capacity)
    
    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        indices = self.rng.choice(self.size, size=batch_size, replace=False, shuffle=True)
        return (
            self.I[indices],
            self.states[indices],
            self.values[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.returns[indices],
        )

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

        # self.negative_slope = 0.01

        # self.fc1 = nn.Linear(state_dim, 64)
        # self.fc2 = nn.Linear(64, num_actions)

        self.fc = nn.Linear(state_dim, num_actions)

        # Initialize weights using He Initialization
        self._initialize_weights()
    
    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        # logits = self.fc2(x)
        logits = self.fc(x)
        return logits

    def _initialize_weights(self):
        """Initialize weights of the layers using He initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # nn.init.kaiming_normal_(layer.weight, a=self.negative_slope)
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)







class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        # self.negative_slope = 0.01

        # self.fc1 = nn.Linear(state_dim, 64)
        # self.fc2 = nn.Linear(64, 1)

        self.fc = nn.Linear(state_dim, 1)

        # Initialize weights using He Initialization
        self._initialize_weights()
    
    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        # value = self.fc2(x).squeeze(-1)  # Ensure scalar output
        value = self.fc(x).squeeze(-1)  # Ensure scalar output
        return value

    def _initialize_weights(self):
        """Initialize weights of the layers using He initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # nn.init.kaiming_normal_(layer.weight, a=self.negative_slope)
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)








class NStepPPO:
    def __init__(self, env, n_step,
    actor_lr_start=1e-3, actor_lr_end=1e-5, actor_lr_decay=0.001, actor_decay="linear",
    critic_lr_start=1e-3, critic_lr_end=1e-5, critic_lr_decay=0.001, critic_decay="linear",
    buffer_size=256, batch_size=64,
    r_clip_epsilon=0.1, v_clip_epsilon=0.1, entropy_coef=0.01,
    gamma=0.99, seed=None, verbose=False):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.n_step = n_step

        # Validate the actor decay type
        if actor_decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.actor_lr_start = actor_lr_start / self.env.dummy_env.n_active_features
        self.actor_lr_end = actor_lr_end / self.env.dummy_env.n_active_features
        self.actor_lr_decay = actor_lr_decay 
        self.actor_decay = actor_decay

        # Validate the critic decay type
        if critic_decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.critic_lr_start = critic_lr_start / self.env.dummy_env.n_active_features
        self.critic_lr_end = critic_lr_end / self.env.dummy_env.n_active_features
        self.critic_lr_decay = critic_lr_decay 
        self.critic_decay = critic_decay
        
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # actor network
        self.actor= Actor(self.state_dim, self.action_dim).to(self.device)
        # critic network
        self.critic = Critic(self.state_dim).to(self.device)
        
        # Optimizer for actor network
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_start)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr_start, amsgrad=True)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=self.update_actor_learning_rate)

        # Optimizer for critic network
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_start)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr_start, amsgrad=True)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=self.update_critic_learning_rate)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=buffer_size, n_envs=self.env.n_envs,
                                   state_dim=self.state_dim, seed=self.seed)

        self.steps = 0

        self.training()

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


    @torch.no_grad()
    def select_action(self, state, info):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.actor.eval()
            logits = self.actor(state).squeeze(0).detach()
        if self.is_training:
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        else:
            action = self.random_argmax(logits)
        greedy = torch.amax(logits) == logits[action]
        return action, greedy

    def random_argmax(self, tensor):
        """
        Returns the index of the maximum value in the tensor, breaking ties randomly.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            int: The index of the maximum value with ties broken randomly.
        """
        max_value = torch.max(tensor)  # Get the maximum value
        max_indices = torch.nonzero(tensor == max_value, as_tuple=False).squeeze(1)  # Indices of maximum values
        random_index = self.rng.integers(len(max_indices)) # Choose one index randomly
        return max_indices[random_index].item()

    def train(self, max_episodes):
        self.actor.train()
        self.critic.train()
        episode_rewards = []
        episode_steps = []
        self.steps = 0

        n_states = SpecialDeque(maxlen=self.n_step)
        n_values = SpecialDeque(maxlen=self.n_step)
        n_actions = SpecialDeque(maxlen=self.n_step)
        n_log_probs = SpecialDeque(maxlen=self.n_step)
        n_entropies = SpecialDeque(maxlen=self.n_step)
        n_rewards = SpecialDeque(maxlen=self.n_step)
        n_I = SpecialDeque(maxlen=self.n_step)
        n_terminateds = SpecialDeque(maxlen=self.n_step)

        episodes = 0
        I, states, infos = self.env.reset()

        I = torch.tensor(I, dtype=torch.float32).to(self.device)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        for t in count():
            actions_logits = self.actor(states)
            states_value = self.critic(states)
            actions_dist = torch.distributions.Categorical(logits=actions_logits)
            actions = actions_dist.sample()
            log_probs = actions_dist.log_prob(actions)

            next_I, next_states, rewards, terminateds, truncateds, next_infos = self.env.step(actions.cpu().numpy())

            next_I = torch.tensor(next_I, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            terminateds = torch.tensor(terminateds, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                next_states_value = self.critic(next_states).detach()

            n_states.append(states)
            n_values.append(states_value.detach())
            n_actions.append(actions.detach())
            n_log_probs.append(log_probs.detach())
            n_rewards.append(rewards)
            n_I.append(I)
            n_terminateds.append(terminateds)

            if t >= self.n_step - 1:
                targets = next_states_value
                for i in reversed(range(self.n_step)):
                    targets = n_rewards[i] + self.gamma * targets * (1.0 - n_terminateds[i])

                self.buffer.push(n_I[0].detach().cpu(), n_states[0].detach().cpu(), n_values[0].detach().cpu(), n_actions[0].detach().cpu(), n_log_probs[0].detach().cpu(), targets.detach().cpu())
                
                if len(self.buffer) > self.batch_size:
                    # Sample from replay buffer with PER
                    mb_I, mb_states, mb_states_value_old, mb_actions, mb_log_probs_old, mb_targets = self.buffer.sample(self.batch_size)
                    mb_I, mb_states, mb_states_value_old, mb_actions, mb_log_probs_old, mb_targets = mb_I.to(self.device), mb_states.to(self.device), mb_states_value_old.to(self.device), mb_actions.to(self.device), mb_log_probs_old.to(self.device), mb_targets.to(self.device)

                    mb_states_value = self.critic(mb_states)
                    mb_actions_logits = self.actor(mb_states)
                    mb_actions_dist = torch.distributions.Categorical(logits=mb_actions_logits)
                    mb_log_probs = mb_actions_dist.log_prob(mb_actions)
                    mb_entropies = mb_actions_dist.entropy()


                    # Unclipped value loss:
                    vf_loss_unclipped = (mb_targets - mb_states_value) ** 2
                    # Clipped value prediction:
                    clipped_values = mb_states_value_old + torch.clamp(mb_states_value - mb_states_value_old, -self.v_clip_epsilon, self.v_clip_epsilon)
                    vf_loss_clipped = (mb_targets - clipped_values) ** 2
                    # Final value function loss:
                    critic_loss = 0.5 * torch.mean(torch.max(vf_loss_unclipped, vf_loss_clipped))
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    self.critic_scheduler.step()

                    # Compute ratio.
                    mb_advantages = mb_targets - mb_states_value_old.detach()
                    ratio = torch.exp(mb_log_probs - mb_log_probs_old)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.r_clip_epsilon, 1 + self.r_clip_epsilon) * mb_advantages
                    surr = torch.min(surr1, surr2)
                    actor_loss = (- torch.pow(self.gamma, mb_I) * surr - self.entropy_coef * mb_entropies).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.actor_scheduler.step()

            I = next_I
            states = next_states
            infos = next_infos
            self.steps += 1

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
                    print(f"Episodes {episodes}, Reward: {episode_reward}, Steps: {steps_in_episode}, actor lr: {self.actor_scheduler.get_last_lr()[0] : .5f}, critic lr: {self.critic_scheduler.get_last_lr()[0] : .5f}")

            if episodes >= max_episodes:
                gc.collect()
                break

        return episode_rewards, episode_steps

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

