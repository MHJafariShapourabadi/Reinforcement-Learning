import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import time
import random
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




class SahredLambdaLRScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, *args, **kwargs):
        super(SahredLambdaLRScheduler, self).__init__(optimizer=optimizer, lr_lambda=lr_lambda, *args, **kwargs)
        self.last_epoch = torch.tensor(self.last_epoch).share_memory_()

    def _initial_step(self):
        """Initialize step counts and perform a step."""
        self._step_count = torch.zeros(1).share_memory_()
        self.step()


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)






class SharedAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdamW, self).__init__(
            params, lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)










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








class NStepA3C:
    def __init__(self, env_class,
    input_dim, action_dim, n_step,
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
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        random.seed(self.seed)

        self.verbose = verbose
        self._is_training = True

        self.device = self._get_device(0)

        # actor network
        self.global_actor = Actor(input_dim, action_dim).to(self.device).share_memory()
        # critic network
        self.global_critic = Critic(input_dim).to(self.device).share_memory()
        
        # Optimizer for actor network
        # self.actor_optimizer = optim.Adam(self.global_actor.parameters(), lr=self.lr_start)
        # self.actor_optimizer = optim.AdamW(self.global_actor.parameters(), lr=self.actor_lr_start, amsgrad=True)
        # self.actor_optimizer = SharedAdam(self.global_actor.parameters(), lr=actor_lr_start)
        self.actor_optimizer = SharedAdamW(self.global_actor.parameters(), lr=actor_lr_start, amsgrad=True)
        # self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=self.update_actor_learning_rate)
        self.actor_scheduler = SahredLambdaLRScheduler(self.actor_optimizer, lr_lambda=self.update_actor_learning_rate)

        # Optimizer for critic network
        # self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=self.lr_start)
        # self.critic_optimizer = optim.AdamW(self.global_critic.parameters(), lr=self.critic_lr_start, amsgrad=True)
        # self.critic_optimizer = SharedAdam(self.global_critic.parameters(), lr=critic_lr_start)
        self.critic_optimizer = SharedAdamW(self.global_critic.parameters(), lr=critic_lr_start, amsgrad=True)
        # self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=self.update_critic_learning_rate)
        self.critic_scheduler = SahredLambdaLRScheduler(self.critic_optimizer, lr_lambda=self.update_critic_learning_rate)
        self.critic_criterion = nn.SmoothL1Loss(beta=Huberbeta)

        self.steps = 0

        self.training()

    @staticmethod
    def _get_device(worker_id):
        """Get the available device (CPU or GPU)."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # Assign GPU based on worker ID
            device_id = worker_id % num_gpus
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        return device


    @property
    def is_training(self):
        return self._is_training

    def training(self):
        self._is_training = True
        torch.manual_seed(seed=self.seed)
        np.random.seed(seed=self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        random.seed(self.seed)
        self.global_actor.train()
        self.global_critic.train()

    def evaluating(self, seed=None):
        self._is_training = False
        seed = seed if seed is not None else self.seed
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)
        self.rng = np.random.default_rng(seed=seed)
        random.seed(seed)
        self.global_actor.eval()
        self.global_critic.eval()


    @torch.no_grad()
    def select_action(self, state, info):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.global_actor.eval()
            logits = self.global_actor(state).squeeze(0).detach()
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

    def worker(self, worker_id, *args):
        """
        Worker function to interact with the environment and update the global networks.
        :param worker_id: ID of the worker.
        """
        device = self._get_device(worker_id)

        local_seed = self.seed + worker_id
        env = env = self.env_class.create_env(*args)
        torch.manual_seed(local_seed) ; np.random.seed(local_seed) ; random.seed(local_seed)

        local_actor = Actor(self.input_dim, self.action_dim).to(device)
        local_critic = Critic(self.input_dim).to(device)
        local_actor.load_state_dict(self.global_actor.state_dict())
        local_critic.load_state_dict(self.global_critic.state_dict())

        while True:
            n_values = SpecialDeque(maxlen=self.n_step)
            n_log_probs = SpecialDeque(maxlen=self.n_step)
            n_entropies = SpecialDeque(maxlen=self.n_step)
            n_rewards = SpecialDeque(maxlen=self.n_step)
            n_discounts = SpecialDeque(maxlen=self.n_step)

            T = float('inf')

            state, info = self.env.reset()
            done = False

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            I = torch.tensor(1.0).to(device)

            episode_reward = 0
            steps_in_episode = 0

            for t in count():
                if t < T:
                    action_logits = local_actor(state)
                    state_value = local_critic(state)
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    entropy = action_dist.entropy()

                    next_state, reward, terminated, truncated, next_info = self.env.step(action.item())
                    done = terminated or truncated

                    if done:
                      T = t + 1

                    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        if terminated:
                            next_state_value = torch.zeros(1, dtype=torch.float32).to(device)
                        else:
                            next_state_value = local_critic(next_state).detach()

                    n_values.append(state_value)
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
                    target = next_state_value
                    for r in reversed(n_rewards):
                        target = self.gamma * target + r

                    n_rewards.popleft()
                    state_value_ = n_values.popleft()
                    critic_loss = self.critic_criterion(state_value_, target)

                    local_critic.zero_grad()
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    for global_param, local_param in zip(self.global_critic.parameters(), local_critic.parameters()):
                                if global_param.grad is None:
                                    global_param._grad = local_param.grad
                    self.critic_optimizer.step()
                    self.critic_scheduler.step()
                    local_critic.load_state_dict(self.global_critic.state_dict())

                    advantage_ = target - state_value_.detach()
                    log_prob_ = n_log_probs.popleft()
                    I_ = n_discounts.popleft()
                    entropy_ = n_entropies.popleft()
                    actor_loss = -(I_ * advantage_ * log_prob_) - self.entropy_coef * entropy_

                    local_actor.zero_grad()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    for global_param, local_param in zip(self.global_actor.parameters(), local_actor.parameters()):
                                if global_param.grad is None:
                                    global_param._grad = local_param.grad
                    self.actor_optimizer.step()
                    self.actor_scheduler.step()
                    local_actor.load_state_dict(self.global_actor.state_dict())

                if tau == T - 1: 
                    gc.collect()
                    break

            if self.get_out_signal: 
                gc.collect()
                break

            with self.get_out_lock:
                self.episode_rewards[self.episode.item()] = episode_reward
                self.episode_steps[self.episode.item()] = steps_in_episode
                print(f"Episode {self.episode.item() + 1}, Reward: {episode_reward}, Steps: {steps_in_episode}, actor lr: {self.actor_scheduler.get_last_lr()[0] : .5f}, critic lr: {self.critic_scheduler.get_last_lr()[0] : .5f}")
                self.episode.add_(1)
                if self.episode.item() >= self.max_episodes:
                    self.get_out_signal.add_(1)

    def train(self, *, max_episodes=1000, num_workers=4, envs_args=None):
        """
        Train the A3C algorithm using multiple worker processes.
        :param num_workers: Number of worker processes.
        :param max_episodes: Maximum number of episodes for each worker.
        """
        self.global_actor.train()
        self.global_critic.train()

        self.episode = torch.zeros(1, dtype=torch.int).share_memory_()
        self.max_episodes = max_episodes

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()

        self.episode_rewards = torch.zeros([max_episodes]).share_memory_()
        self.episode_steps = torch.zeros([max_episodes], dtype=torch.int).share_memory_()

        processes = []
        if envs_args is not None:
            num_workers = len(envs_args)
            for worker_id in range(num_workers):
                args = envs_args[worker_id]
                process = mp.Process(target=self.worker, args=(worker_id, *args))
                process.start()
                processes.append(process)
        else:
            for worker_id in range(num_workers):
                process = mp.Process(target=self.worker, args=(worker_id,))
                process.start()
                processes.append(process)


        for process in processes:
            process.join()

        return self.episode_rewards, self.episode_steps

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

