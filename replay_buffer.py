"""Fixed-size experience replay buffer for off-policy learning."""

import random
from collections import deque, namedtuple

import numpy as np
import torch

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Stores past transitions and samples random mini-batches for training.

    Using experience replay breaks temporal correlations between consecutive
    samples, which stabilises DQN training.
    """

    def __init__(self, capacity, seed=None):
        """Initialise the buffer.

        Args:
            capacity: Maximum number of transitions to store.
            seed:     Optional RNG seed for reproducible sampling.
        """
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Add a single transition to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size, device="cpu"):
        """Sample a random mini-batch and return it as tensors.

        Args:
            batch_size: Number of transitions to sample.
            device:     Torch device for the returned tensors.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors.
        """
        experiences = self.rng.sample(list(self.buffer), batch_size)

        states = torch.tensor(
            np.array([e.state for e in experiences]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [e.action for e in experiences],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        rewards = torch.tensor(
            [e.reward for e in experiences],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array([e.next_state for e in experiences]),
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            [float(e.done) for e in experiences],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
