import torch
import numpy as np




# Softmax Exploration Strategy
class SoftmaxExploration:
    """
    Implements the Softmax Exploration strategy, where actions are selected based on a softmax probability
    distribution of Q-values. The temperature parameter controls the degree of exploration.

    Attributes:
        temperature (float): Current temperature value.
        temperature_start (float): Initial temperature value.
        temperature_end (float): Minimum temperature value (lower bound).
        temperature_decay (float): Rate of decay for the temperature parameter.
        decay (str): Type of decay to apply to the temperature ('linear', 'exponential' or 'reciprocal').
        seed (int): Seed for random number generation (optional).
        rng (np.random.Generator): Random number generator.
        steps (int): Counter for the number of steps taken.
    """

    def __init__(self, temperature_start=1.0, temperature_end=0.1, temperature_decay=0.001, decay="linear", seed=None):
        """
        Initializes the SoftmaxExploration class.

        Args:
            temperature_start (float): Initial temperature value (default: 1.0).
            temperature_end (float): Minimum temperature value (default: 0.1).
            temperature_decay (float or None): Rate of decay for the temperature (default: 0.001).
            decay (str): Type of decay for the temperature ('linear', 'exponential' or 'reciprocal') (default: 'linear').
            seed (int or None): Seed for random number generation (optional).
        """
        # Validate the decay type
        if decay not in {"linear", "exponential", "reciprocal"}:
            raise ValueError("Invalid value for decay. Must be 'linear', 'exponential' or 'reciprocal'.")

        self.temperature = temperature_start  # Initialize temperature to start value.
        self.temperature_start = temperature_start  # Store the starting temperature.
        self.temperature_end = temperature_end  # Set the minimum temperature value.
        self.temperature_decay = temperature_decay # Determine the default decay rate 
        self.decay = decay  # Set the type of decay (linear or exponential).

        self.seed = seed  # Set the random seed for reproducibility.
        self.rng = np.random.default_rng(seed=seed)  # Initialize random number generator.
        self.steps = 0  # Initialize step counter.

    def reset(self):
        """
        Resets the exploration strategy to its initial state.
        """
        self.temperature = self.temperature_start  # Reset temperature to the start value.
        self.steps = 0  # Reset step counter.

    @torch.no_grad()
    def select_action(self, q_values, steps=None):
        """
        Selects an action based on the softmax probability distribution of Q-values.

        Args:
            q_values (Tensor): Tensor of Q-values for each action.
            steps (int or None): Optional step value to update the temperature manually (default: None).

        Returns:
            int: Selected action index.
        """
        self.steps += 1  # Increment step counter.
        steps = self.steps if steps is None else steps  # Use the provided step value if given.

        # Update the temperature value based on the current step.
        self.update_temperature(steps)

        # Normalize Q-values by the temperature to control exploration.
        q_values = q_values / self.temperature

        # Compute softmax probabilities over Q-values.
        probabilities = torch.softmax(q_values, dim=0).cpu().numpy()

        # Select an action randomly based on the computed probabilities.
        return self.rng.choice(len(probabilities), p=probabilities)

    def update_temperature(self, steps):
        """
        Updates the temperature value based on the current step using the specified decay type.

        Args:
            steps (int): Current step count.

        Behavior:
            - Linear Decay: Temperature decreases linearly at each step.
            - Exponential Decay: Temperature decreases exponentially at each step.
            - Reciprocal Decay: Temperature decreases inversely with each step.
        """
        if self.decay == "linear":
            # Linearly decay temperature, ensuring it does not drop below the minimum temperature.
            self.temperature = max(
                self.temperature_end, 
                self.temperature_start - self.temperature_decay * steps 
            )
        elif self.decay == "exponential":
            # Exponentially decay temperature, ensuring it does not drop below the minimum temperature.
            self.temperature = max(
                self.temperature_end, 
                self.temperature_start * np.exp(-self.temperature_decay * steps)
            )
        elif self.decay == "reciprocal":
            # Inversely decay temperature, ensuring it does not drop below the minimum temperature.
            self.temperature = max(
                self.temperature_end,
                self.temperature_start / (1 + self.temperature_decay * steps)
            )
