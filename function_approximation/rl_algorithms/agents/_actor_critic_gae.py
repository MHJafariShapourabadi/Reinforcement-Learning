import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import time
# from collections import deque
from itertools import count
import gc








class SpecialDeque:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.list = [None] * maxlen
        self.last_item = None
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

    def append_last(self, value):
        self.last_item = value

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
        if index == self.maxlen:
            return self.last_item
        else:
            if index > self.maxlen or index < - self.maxlen:
                raise IndexError("Index out of deque range.")
            index = (index + self.start_idx) % self.maxlen
            return self.list[index]

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{self.list}"









class Actor(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(Actor, self).__init__()

        # self.negative_slope = 0.01

        # self.fc1 = nn.Linear(input_dim, 64)
        # self.fc2 = nn.Linear(64, num_actions)

        self.fc = nn.Linear(input_dim, num_actions)

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
    def __init__(self, input_dim):
        super(Critic, self).__init__()

        # self.negative_slope = 0.01

        # self.fc1 = nn.Linear(input_dim, 64)
        # self.fc2 = nn.Linear(64, 1)

        self.fc = nn.Linear(input_dim, 1)

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








class ActorCriticGAE:
    def __init__(self, env_class,
    input_dim, action_dim, n_step, lambd = 0.9,
    actor_lr_start=1e-3, actor_lr_end=1e-5, actor_lr_decay=0.001, actor_decay="linear",
    critic_lr_start=1e-3, critic_lr_end=1e-5, critic_lr_decay=0.001, critic_decay="linear",
    entropy_coef=0.01, Huberbeta=1.0,
    gamma=0.99, seed=None, verbose=False):
        self.env_class = env_class
        self.env = env_class.create_env()
        self.observation_shape = self.env.observation_shape
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.lambd = lambd

        # Validate the actor decay type
        if actor_decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.actor_lr_start = actor_lr_start / self.env.n_active_features
        self.actor_lr_end = actor_lr_end / self.env.n_active_features
        self.actor_lr_decay = actor_lr_decay 
        self.actor_decay = actor_decay

        # Validate the critic decay type
        if critic_decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.critic_lr_start = critic_lr_start / self.env.n_active_features
        self.critic_lr_end = critic_lr_end / self.env.n_active_features
        self.critic_lr_decay = critic_lr_decay 
        self.critic_decay = critic_decay

        self.entropy_coef = entropy_coef
        self.Huberbeta = Huberbeta
        self.gamma = gamma

        self.seed = seed if seed is not None else int(time.time())
        torch.manual_seed(seed=self.seed)
        self.rng = np.random.default_rng(seed=self.seed)

        self.verbose = verbose
        self._is_training = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # actor network
        self.actor= Actor(input_dim, action_dim).to(self.device)
        # critic network
        self.critic = Critic(input_dim).to(self.device)
        
        # Optimizer for actor network
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_start)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr_start, amsgrad=True)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=self.update_actor_learning_rate)

        # Optimizer for critic network
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_start)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr_start, amsgrad=True)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=self.update_critic_learning_rate)
        self.critic_criterion = nn.SmoothL1Loss(beta=Huberbeta)

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
        observation = info['observation']
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.actor.eval()
            logits = self.actor(observation).squeeze(0).detach()
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

        for episode in range(max_episodes):
            n_values = SpecialDeque(maxlen=self.n_step)
            n_log_probs = SpecialDeque(maxlen=self.n_step)
            n_entropies = SpecialDeque(maxlen=self.n_step)
            n_rewards = SpecialDeque(maxlen=self.n_step)
            n_discounts = SpecialDeque(maxlen=self.n_step)

            T = float('inf')
        
            state, info = self.env.reset()
            done = False

            I = torch.tensor(1.0).to(self.device)

            episode_reward = 0
            steps_in_episode = 0

            for t in count():
                if t < T:
                    observation = info['observation']
                    observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action_logits = self.actor(observation)
                    state_value = self.critic(observation)
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    entropy = action_dist.entropy()

                    next_state, reward, terminated, truncated, next_info = self.env.step(action.item())
                    done = terminated or truncated

                    if done:
                      T = t + 1

                    next_observation = next_info['observation']
                    next_observation = torch.tensor(next_observation, dtype=torch.float32).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        if terminated:
                            next_state_value = torch.zeros(1, dtype=torch.float32).to(self.device)
                        else:
                            next_state_value = self.critic(next_observation).detach()

                    n_values.append(state_value)
                    n_values.append_last(next_state_value)
                    n_log_probs.append(log_prob)
                    n_entropies.append(entropy)
                    n_rewards.append(reward)
                    n_discounts.append(I)

                    I *= self.gamma
                    state = next_state
                    info = next_info

                    episode_reward += reward
                    self.steps += 1
                    steps_in_episode += 1
                
                tau = t + 1 - self.n_step
                if tau >= 0:
                    target = n_values[0].detach().clone()
                    for i in range(len(n_values)):
                        td_error = n_rewards[i] + self.gamma * n_values[i + 1].detach() - n_values[i].detach()
                        target += ((self.gamma * self.lambd) ** i) * td_error
                    
                    n_rewards.popleft()
                    state_value_ = n_values.popleft()
                    critic_loss = self.critic_criterion(state_value_, target.detach())

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    self.critic_scheduler.step()

                    advantage_ = target.detach() - state_value_.detach()
                    log_prob_ = n_log_probs.popleft()
                    I_ = n_discounts.popleft()
                    entropy_ = n_entropies.popleft()
                    actor_loss = -(I_ * advantage_ * log_prob_) - self.entropy_coef * entropy_

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.actor_scheduler.step()

                if tau == T - 1: 
                    gc.collect()
                    break

            episode_rewards.append(episode_reward)
            episode_steps.append(steps_in_episode)
            print(f"Episode {episode + 1}, Reward: {episode_reward}, Steps: {steps_in_episode}, actor lr: {self.actor_scheduler.get_last_lr()[0] : .5f}, critic lr: {self.critic_scheduler.get_last_lr()[0] : .5f}")

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
