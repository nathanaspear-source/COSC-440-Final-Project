"""DQN Agent that learns to land on the moon.

Combines an epsilon-greedy policy, experience replay, a target network,
and gradient-descent updates on the Bellman error.
"""

import os
import random

import numpy as np
import torch
from torch import optim

import config
from dqn_network import create_q_networks, soft_update
from replay_buffer import ReplayBuffer


class DQNAgent:  # pylint: disable=too-many-instance-attributes
    """Deep Q-Network agent for discrete action spaces.

    The agent collects experience into a replay buffer, samples
    mini-batches, and minimises the TD error between the online
    network's Q-values and the bootstrapped target Q-values.
    """

    def __init__(self, state_size, action_size, epsilon_start = None, epsilon_end = None, epsilon_decay = None, seed=None):
        self.state_size = state_size
        self.action_size = action_size

        # Determines whether to use Nvidia Cuda, Apple Metal, or CPU
        # depending on what hardware is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.rng = random.Random(seed)

        self.online_net, self.target_net = create_q_networks(
            state_size, action_size, self.device
        )
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=config.LEARNING_RATE
        )
        self.buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY, seed=seed)

        # Input validation to ensure starting, ending, and decay values
        # of epsilon are not None
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            self.epsilon = config.EPSILON_START

        if epsilon_end is not None:
            self.epsilon_end = epsilon_end
        else:
            self.epsilon_end = config.EPSILON_END

        if epsilon_decay is not None:
            self.epsilon_decay = epsilon_decay
        else:
            self.epsilon_decay = config.EPSILON_DECAY

    def select_action(self, state):
        """Choose an action using epsilon-greedy exploration.

        With probability epsilon a random action is taken; otherwise
        the action with the highest Q-value from the online network.
        """
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_size)

        state_tensor = torch.tensor(
            np.array(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        self.online_net.train()

        return int(q_values.argmax(dim=1).item())

    def step(self, state, action, reward, next_state, done):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Store a transition and, if enough data exists, learn from a batch."""
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) >= config.BATCH_SIZE:
            self._learn()

    def _learn(self):
        """Sample a batch and perform one gradient step on the Bellman error."""
        states, actions, rewards, next_states, dones = self.buffer.sample(
            config.BATCH_SIZE, self.device
        )

        # Q-values for the actions actually taken
        q_values = self.online_net(states).gather(1, actions)

        # Bootstrap target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_targets_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
        q_targets = rewards + config.GAMMA * q_targets_next * (1.0 - dones)

        loss = torch.nn.functional.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.online_net, self.target_net)

    def decay_epsilon(self):
        """Multiplicatively decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Persist the online network weights to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.online_net.state_dict(), filepath)

    def load(self, filepath):
        """Load previously saved network weights."""
        state_dict = torch.load(filepath, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.epsilon = config.EPSILON_END
