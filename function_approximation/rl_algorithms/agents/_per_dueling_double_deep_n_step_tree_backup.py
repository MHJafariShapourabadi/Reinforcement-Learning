import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from ..exploration import GreedyExploration






# Define the Dueling Q-Network
class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Args:
            input_shape (int): Dimension of the input state.
            num_actions (int): Number of actions (output dimension).
        """
        super(DuelingQNetwork, self).__init__()

        # Value stream
        self.value_fc = nn.Linear(input_shape[0], 1)         # Output single value for the state

        # Advantage stream
        self.advantage_fc = nn.Linear(input_shape[0], num_actions)  # Output advantage for each action

        # Initialize weights using He Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of the layers using He initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
              nn.init.zeros_(layer.weight)
              if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass of the dueling network.

        Args:
            x (Tensor): Input tensor (state representation).

        Returns:
            Tensor: Q-values for each action.
        """

        # Value stream
        value = self.value_fc(x)  # Output the state-value V(s)

        # Advantage stream
        advantage = self.advantage_fc(x)  # Output the advantage A(s, a)

        # Combine value and advantage to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values








class PrioritizedReplayBuffer:
    def __init__(self, capacity,
                 alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3,
                 beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
                 epsilon=1e-5, seed=None):
        self.capacity = capacity
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
        self.states = np.zeros((capacity,), dtype=np.int64)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity,), dtype=np.int64)
        self.ns = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Initialize priorities (start with max priority for new transitions)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def reset(self):
      self.alpha = self.alpha_start
      self.beta = self.beta_start
      self.alpha_steps = 0
      self.beta_steps = 0

    def update_alpha(self, steps=None):
      self.alpha_steps += 1
      steps = self.alpha_steps if steps is None else steps
      self.beta = min(self.alpha_end, self.alpha_start + self.alpha_increment * steps)

    def update_beta(self, steps=None):
      self.beta_steps += 1
      steps = self.beta_steps if steps is None else steps
      self.beta = min(self.beta_end, self.beta_start + self.beta_increment * steps)

    def push(self, state, action, reward, next_state, n, done):
        """Store a transition in the buffer with maximum priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.ns[self.pos] = n
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions based on priorities."""
        if self.size == 0:
            raise ValueError("Replay buffer is empty!")

        # Calculate probabilities using priorities
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices based on probabilities
        indices = self.rng.choice(self.size, size=batch_size, p=sampling_probabilities)

        # Compute importance sampling weights
        total_prob = sampling_probabilities[indices]
        weights = (self.size * total_prob) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.ns[indices],
            self.dones[indices],
            indices,  # Return sampled indices to update priorities later
            weights,
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities for the sampled transitions."""
        # New priorities are proportional to the absolute TD error
        self.priorities[indices] = np.abs(td_errors) + self.epsilon  # Add epsilon for stability

    def __len__(self):
        return self.size







# PERDuelingDoubleDeepNStepTreeBackup Agent
class PERDuelingDoubleDeepNStepTreeBackup:
    def __init__(self, env, exploration, n_step, 
                lr_start=1e-3, lr_end=1e-5, lr_decay=0.001, decay="linear",
                gamma=0.99,
                Huberbeta=1.0, buffer_size=10000, batch_size=64, polyak_tau=0.05,
                alpha_start=0.6, alpha_end=0.8, alpha_increment=1e-3, 
                beta_start=0.4, beta_end=1.0, beta_increment=1e-4,
                seed=None, verbose=False):
        self.env = env
        self.observation_shape = env.observation_shape
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n

        self.n_step = n_step
        self.n_states = np.zeros((n_step,), dtype=np.int64)
        self.n_actions = np.zeros((n_step,), dtype=np.int64)
        self.n_rewards = np.zeros((n_step,), dtype=np.float32)
        self.n_next_states = np.zeros((n_step,), dtype=np.int64)
        self.n_next_greedys = np.zeros((n_step,), dtype=np.float32)
        self.n_dones = np.zeros((n_step,), dtype=np.float32)

        # Validate the decay type
        if decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.lr = lr_start / env.n_active_features
        self.lr_start = lr_start / env.n_active_features
        self.lr_end = lr_end / env.n_active_features
        self.lr_decay = lr_decay 
        self.decay = decay

        self.Huberbeta = Huberbeta
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.polyak_tau = polyak_tau
        self.seed = seed
        self.verbose = verbose
        self._is_training = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main and Target networks
        self.policy_net = DuelingQNetwork(self.observation_shape, self.num_actions).to(self.device)
        self.target_net = DuelingQNetwork(self.observation_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr_start)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr_start, amsgrad=True)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.update_learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none', beta=Huberbeta)  # For PER, reduction='none' is needed

        # Prioritized Replay Buffer
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size,
                                              alpha_start=alpha_start, alpha_end=alpha_end, alpha_increment=alpha_increment,
                                              beta_start=beta_start, beta_end=beta_end, beta_increment=beta_increment,
                                              seed=self.seed)

        self.steps = 0

        # Exploration strategy
        self.train_exploration = exploration
        self.training()

    @property
    def is_training(self):
      return self._is_training

    def training(self):
      self._is_training = True
      self.exploration = self.train_exploration

    def evaluating(self, seed=None):
      self._is_training = False
      self.exploration = GreedyExploration(seed=seed)

    @torch.no_grad()
    def select_action(self, state, info):
        observation = info['observation']
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(observation).squeeze(0).detach()
        action = self.exploration.select_action(q_values, self.steps)
        greedy = torch.amax(q_values) == q_values[action]
        ########################################################################
        if self.verbose:
            print("==========================")
            print(f"observation: {info['observation']}")
            print(f"state: {state}")
            print(f"q_values: {q_values}")
            print(f"action: {action} - {self.env.action_to_dir[action]}")
            print("==========================")
        ########################################################################
        return action, greedy

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return

        self.policy_net.train()

        # Sample from replay buffer with PER
        states, actions, rewards, next_states, ns, dones, indices, weights = self.buffer.sample(self.batch_size)

        observations = self.env.state_to_vector[states]
        next_observations = self.env.state_to_vector[next_states]

        # Convert to tensors
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(1) # power of the gamma for next observation in target computation
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute current Q values
        q_values = self.policy_net(observations).gather(1, actions)

       # Compute Double Q-Learning target Q values
        with torch.no_grad():
            # Use policy network to select action with max Q-value
            next_action_indices = self.policy_net(next_observations).argmax(1, keepdim=True)

            # Use target network to evaluate the Q-value of the selected action
            next_q_values = self.target_net(next_observations).gather(1, next_action_indices)

            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * (self.gamma ** ns) * next_q_values

        # Compute TD errors and loss
        td_errors = target_q_values - q_values
        loss = (weights * self.criterion(q_values, target_q_values)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        # Update priorities in replay buffer
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())

        # Update beta for PER
        self.buffer.update_alpha(self.steps)
        self.buffer.update_beta(self.steps)

        # Update target network using Polyak averaging
        self.update_target_network()
        
    @torch.no_grad()
    def update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.polyak_tau * policy_param.data + (1.0 - self.polyak_tau) * target_param.data)

    def train(self, num_episodes):
        episode_rewards = []
        episode_steps = []
        self.steps = 0

        for episode in range(num_episodes):
            state, info = self.env.reset()
            # Select action
            action, greedy = self.select_action(state, info)

            t = 0
            idx = t % self.n_step

            T = float('inf')

            episode_reward = 0
            steps_in_episode = 0

            while True:
                if t < T:
                    self.steps += 1
                    steps_in_episode += 1

                    self.n_states[idx] = state
                    self.n_actions[idx] = action

                    # Step environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)

                    # Select action
                    next_action, greedy = self.select_action(next_state, info)

                    self.n_rewards[idx] = reward
                    self.n_next_states[idx] = next_state
                    self.n_next_greedys[idx] = greedy
                    self.n_dones[idx] = terminated

                    if terminated:
                      T = t + 1

                    state = next_state
                    action = next_action
                    episode_reward += reward

                if truncated:
                  break

                tau = t + 1 - self.n_step
                if tau >= 0:
                    index = tau % self.n_step
                    s = self.n_states[index]
                    a = self.n_actions[index]
                    sum_rs = 0
                    last_i = min(tau + self.n_step, T)
                    for n, i in enumerate(range(tau, last_i)):
                      index = i % self.n_step
                      sum_rs += (self.gamma ** n) * self.n_rewards[index]
                      if not self.n_next_greedys[index]:
                        break
                    next_s = self.n_next_states[index]
                    d = self.n_dones[index]
                    # Store in replay buffer
                    self.buffer.push(s, a, sum_rs, next_s, n + 1, d)

                # Optimize the model
                self.optimize()

                if tau == T - 1:
                  break

                t += 1
                idx = t % self.n_step

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