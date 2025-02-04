import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import time
import gc









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








class A2C:
    def __init__(self, env,
    actor_lr_start=1e-3, actor_lr_end=1e-5, actor_lr_decay=0.001, actor_decay="linear",
    critic_lr_start=1e-3, critic_lr_end=1e-5, critic_lr_decay=0.001, critic_decay="linear",
    entropy_coef=0.01, Huberbeta=1.0,
    gamma=0.99, seed=None, verbose=False):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

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

        episodes = 0
        I, states, infos = self.env.reset()

        I = torch.tensor(I, dtype=torch.float32).to(self.device)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        while True:
            actions_logits = self.actor(states)
            states_value = self.critic(states)
            actions_dist = torch.distributions.Categorical(logits=actions_logits)
            actions = actions_dist.sample()
            log_probs = actions_dist.log_prob(actions)
            entropys = actions_dist.entropy()

            next_I, next_states, rewards, terminateds, truncateds, next_infos = self.env.step(actions.cpu().numpy())
            dones = np.logical_or(terminateds, truncateds)
            episodes += np.sum(dones, keepdims=False)

            next_I = torch.tensor(next_I, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                next_states_value = self.critic(next_states).detach()
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            terminateds = torch.tensor(terminateds, dtype=torch.float32).to(self.device)
            
            targets = rewards + self.gamma * next_states_value * (1.0 - terminateds)
            # print("=================================")
            # print(f"states: {states.shape}")
            # print(f"next states: {next_states.shape}")
            # print(f"actions_logits: {actions_logits.shape}")
            # print(f"rewards: {rewards.shape}")
            # print(f"states value: {states_value.shape}")
            # print(f"next states value: {next_states_value.shape}")
            # print(f"terminateds: {terminateds.shape}")
            # print(f"targets: {targets.shape}")
            critic_loss = self.critic_criterion(states_value, targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_scheduler.step()

            advantages = targets - states_value.detach()
            # print(f"log_probs: {log_probs.shape}")
            # print(f"entropys: {entropys.shape}")
            # print(f"advantages: {advantages.shape}")
            # print(f"I: {I.shape}")
            actor_loss = (-(torch.pow(self.gamma, I) * advantages * log_probs) - self.entropy_coef * entropys).mean()
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

