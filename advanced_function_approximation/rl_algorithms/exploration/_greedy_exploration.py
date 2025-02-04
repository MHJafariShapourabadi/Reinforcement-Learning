import torch
import numpy as np





# Greedy Exploration
class GreedyExploration:
    """
    Implements the Greedy exploration strategy, where actions are selected 
    greedily (exploitation) based on a decaying epsilon value.

    Attributes:
        seed (int): Seed for reproducibility.
        rng (np.random.Generator): Random number generator for reproducibility.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the Greedy Exploration strategy.

        Args:
            seed (int, optional): Seed for reproducibility. Default is None.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    @torch.no_grad()
    def select_action(self, q_values, steps=None):
        """
        Select an action using greedy exploration.

        Args:
            q_values (torch.Tensor): Q-values predicted by the network.
            steps (int, optional): Current step count (not used in this strategy but included for compatibility).

        Returns:
            int: Selected action index.
        """
        # Exploitation: choose the action with the highest Q-value
        return self.random_argmax(q_values)

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