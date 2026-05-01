import pickle
import config
import random
import numpy as np
from collections import defaultdict
import os

class SARSAAgent:
    def __init__(self, action_size, gamma, alpha, epsilon_start = None, epsilon_end = None, epsilon_decay = None, seed=None):
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.rng = random.Random(seed)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

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
        """Selects action based on epsilon greedy policy."""
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_size)

        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_action = np.where(q_values == max_q)[0]
        return int(self.rng.choice(best_action))

    def update(self, state, action, reward, next_state, next_action, ep_done):
        """Updates Q-table with new Q-values."""
        cur_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]

        target = reward + self.gamma * next_q * (1.0 - ep_done)
        self.q_table[state][action] += self.alpha * (target - cur_q)

    def decay_epsilon(self):
        """Decay epsilon based on epsilon decay policy."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves Q-table to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(dict(self.q_table), file)

    def load(self, filepath):
        """Reads Q-table from disk."""
        with open(filepath, "rb") as file:
            loaded = pickle.load(file)

        # Resets Q-table and sets epsilon to allow maximum exploitation
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), loaded)
        self.epsilon = self.epsilon_end
