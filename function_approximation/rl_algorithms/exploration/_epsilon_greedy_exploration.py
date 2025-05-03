import torch
import numpy as np







# Epsilon-Greedy Exploration Strategy
class EpsilonGreedyExploration:
    """
    Implements the Epsilon-Greedy exploration strategy, where actions are selected 
    either randomly (exploration) or greedily (exploitation) based on a decaying epsilon value.

    Attributes:
        epsilon (float): Current epsilon value.
        epsilon_start (float): Initial epsilon value.
        epsilon_end (float): Minimum epsilon value (lower bound).
        epsilon_decay (float): Rate of decay for epsilon.
        decay (str): Type of decay ('linear', 'exponential' or 'reciprocal').
        seed (int): Seed for reproducibility.
        rng (np.random.Generator): Random number generator for reproducibility.
        steps (int): Counter for the number of steps taken.
    """

    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.001, decay="linear", seed=None):
        """
        Initializes the EpsilonGreedyExploration strategy.

        Args:
            epsilon_start (float): Initial epsilon value for exploration (default: 1.0).
            epsilon_end (float): Minimum epsilon value for exploitation (default: 0.1).
            epsilon_decay (float, optional): Decay rate for epsilon (default: 0.001).
            decay (str): Type of epsilon decay ('linear', 'exponential' or 'reciprocal') (default: 'linear').
            seed (int, optional): Seed for random number generation (default: None).
        """
        # Validate the decay type
        if decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.epsilon = epsilon_start  # Current epsilon value
        self.epsilon_start = epsilon_start  # Initial epsilon value
        self.epsilon_end = epsilon_end  # Minimum epsilon value
        self.epsilon_decay = epsilon_decay # Set decay rate based
        self.decay = decay  # Type of decay (linear, exponential or reciprocal)

        self.seed = seed  # Random seed
        self.rng = np.random.default_rng(seed=seed)  # Initialize random number generator
        self.steps = 0  # Step counter

    def reset(self):
        """
        Resets the exploration strategy to its initial state.
        """
        self.epsilon = self.epsilon_start  # Reset epsilon to its starting value
        self.steps = 0  # Reset step counter

    @torch.no_grad()
    def select_action(self, q_values, steps=None):
        """
        Selects an action using the epsilon-greedy exploration strategy.

        Args:
            q_values (torch.Tensor): Q-values predicted by the model for each action.
            steps (int, optional): Current step count. If not provided, internal counter is used.

        Returns:
            int: Index of the selected action.
        """
        self.steps += 1  # Increment the step counter
        steps = self.steps if steps is None else steps  # Use provided step value if given

        # Update epsilon based on the current step count
        self.update_epsilon(steps)

        # Decide between exploration and exploitation
        if self.rng.random() < self.epsilon:
            # Exploration: choose a random action
            return self.rng.integers(len(q_values))
        else:
            # Exploitation: choose the action with the highest Q-value
            return self.random_argmax(q_values)

    def update_epsilon(self, steps):
        """
        Updates the epsilon value using the specified decay type.

        Args:
            steps (int): Current step count.

        Behavior:
            - Linear Decay: Epsilon decreases linearly with each step.
            - Exponential Decay: Epsilon decreases exponentially with each step.
            - Reciprocal Decay: Epsilon decreases inversely with each step.
        """
        if self.decay == "linear":
            # Linearly decay epsilon, ensuring it does not drop below epsilon_end
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - self.epsilon_decay * steps 
            )
        elif self.decay == "exponential":
            # Exponentially decay epsilon, ensuring it does not drop below epsilon_end
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start * np.exp(-self.epsilon_decay * steps)
            )
        elif self.decay == "reciprocal":
            # Inversely decay epsilon, ensuring it does not drop below epsilon_end
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start / (1 + self.epsilon_decay * steps)
            )

    def random_argmax(self, tensor):
        """
        Returns the index of the maximum value in the tensor, breaking ties randomly.

        Args:
            tensor (torch.Tensor): A tensor containing Q-values for each action.

        Returns:
            int: Index of the maximum value, with ties broken randomly.
        """
        max_value = torch.max(tensor)  # Get the maximum Q-value
        max_indices = torch.nonzero(tensor == max_value, as_tuple=False).squeeze(1)  # Get indices of all max values
        # Select one index randomly from the maximum indices
        random_index = self.rng.integers(len(max_indices))
        return max_indices[random_index].item()
