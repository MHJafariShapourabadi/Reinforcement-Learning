import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import time









class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()

        # self.negative_slope = 0.01

        # self.fc1 = nn.Linear(input_shape[0], 64)
        # self.fc2 = nn.Linear(64, num_actions)

        self.fc = nn.Linear(input_shape[0], num_actions)

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







class ValueNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()

        # self.negative_slope = 0.01

        # self.fc1 = nn.Linear(input_shape[0], 64)
        # self.fc2 = nn.Linear(64, 1)

        self.fc = nn.Linear(input_shape[0], 1)

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








class REINFORCEWithBaseline:
    def __init__(self, env,
    lr_start=1e-3, lr_end=1e-5, lr_decay=0.001, decay="linear",
    entropy_coef=0.01, Huberbeta=1.0,
    gamma=0.99, seed=None, verbose=False):
        self.env = env
        self.observation_shape = env.observation_shape
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n

        # Validate the decay type
        if decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.lr = lr_start / env.n_active_features
        self.lr_start = lr_start / env.n_active_features
        self.lr_end = lr_end / env.n_active_features
        self.lr_decay = lr_decay 
        self.decay = decay

        self.entropy_coef = entropy_coef
        self.Huberbeta = Huberbeta
        self.gamma = gamma
        self.seed = seed if seed is not None else int(time.time())
        torch.manual_seed(seed=self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        self.verbose = verbose
        self._is_training = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network
        self.policy_net = PolicyNetwork(self.observation_shape, self.num_actions).to(self.device)
        # Value network
        self.value_net = ValueNetwork(self.observation_shape).to(self.device)
        
        # Optimizer for policy network
        # self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr_start)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr_start, amsgrad=True)
        self.policy_scheduler = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda=self.update_learning_rate)

        # Optimizer for value network
        # self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr_start)
        self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=self.lr_start, amsgrad=True)
        self.value_scheduler = optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda=self.update_learning_rate)
        self.value_criterion = nn.SmoothL1Loss(beta=Huberbeta)

        self.steps = 0

        self.training()

    @property
    def is_training(self):
        return self._is_training

    def training(self):
        self._is_training = True
        torch.manual_seed(seed=self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        self.policy_net.train()
        self.value_net.train()

    def evaluating(self, seed=None):
        self._is_training = False
        seed = seed if seed is not None else self.seed
        torch.manual_seed(seed=seed)
        self.rng = np.random.default_rng(seed=seed)
        self.policy_net.eval()
        self.value_net.eval()

    def _select_action(self, state, info):
        self.policy_net.train()
        observation = info['observation']
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(observation).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        ########################################################################
        if self.verbose:
            print("==========================")
            # print(f"observation: {info['observation']}")
            print(f"state: {state}")
            print(f"logits: {logits}")
            print(f"action: {action} - {self.env.action_to_dir[action.item()]}")
            print("==========================")
        ########################################################################
        return action.item(), log_prob, entropy

    @torch.no_grad()
    def select_action(self, state, info):
        observation = info['observation']
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.policy_net.eval()
            logits = self.policy_net(observation).squeeze(0).detach()
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

    def compute_returns(self, rewards):
        returns = np.zeros((len(rewards),))
        G = 0
        for i in reversed(range(len(rewards))):
            r = rewards[i]
            G = r + self.gamma * G
            returns[i] = G
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns

    def compute_values(self, states):
        self.value_net.train()
        # Compute value estimates
        observations = self.env.state_to_vector[states]
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        values = self.value_net(observations)
        return values

    def update_policy(self, log_probs, entropies, returns, values):
        # Compute advantages
        advantages = returns - values.detach()  # Detach values to prevent backprop through value network

        # Policy loss
        policy_losses = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_losses.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_losses).sum()

        # Entropy loss
        entropy_loss = torch.stack(entropies).mean()

        print(f"entropy loss: {entropy_loss}")

        # Combine policy loss and entropy loss
        total_policy_loss = policy_loss - self.entropy_coef * entropy_loss


        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_scheduler.step()

    def update_value_function(self, returns, values):
        # Value loss (Mean Squared Error)
        value_loss = self.value_criterion(values, returns)

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        self.value_scheduler.step()

    def train(self, num_episodes):
        self.policy_net.train()
        self.value_net.train()
        episode_rewards = []
        episode_steps = []
        self.steps = 0

        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False

            episode_reward = 0
            steps_in_episode = 0

            states = []
            rewards = []
            log_probs = []
            entropies = []

            while not done:
                self.steps += 1
                steps_in_episode += 1

                states.append(state)

                action, log_prob, entropy = self._select_action(state, info)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                log_probs.append(log_prob)
                entropies.append(entropy)

                state = next_state
                episode_reward += reward

            returns = self.compute_returns(rewards)
            values = self.compute_values(states)
            self.update_policy(log_probs, entropies, returns, values)
            self.update_value_function(returns, values)

            episode_rewards.append(episode_reward)
            episode_steps.append(steps_in_episode)
            print(f"Episode {episode + 1}, Reward: {episode_reward}, Steps: {steps_in_episode}")

        return episode_rewards, episode_steps

    def update_learning_rate(self, step):
        """
        Updates the learning rate value using the specified decay type.

        Args:
            steps (int): Current step count.

        Behavior:
            - Linear Decay: Epsilon decreases linearly with each step.
            - Exponential Decay: Epsilon decreases exponentially with each step.
            - Reciprocal Decay: Epsilon decreases inversely with each step.
        """
        if self.decay == "linear":
            # Linearly decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.lr_end,
                self.lr_start - self.lr_decay * step 
            )
        elif self.decay == "exponential":
            # Exponentially decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.lr_end,
                self.lr_start * np.exp(-self.lr_decay * step)
            )
        elif self.decay == "reciprocal":
            # Inversely decay epsilon, ensuring it does not drop below epsilon_end
            self.lr = max(
                self.lr_end,
                self.lr_start / (1 + self.lr_decay * step)
            )
        return self.lr / self.lr_start

