import numpy as np

class StateDiscretizer:
    def __init__(self, num_bins, bin_ranges):
        self.num_bins = num_bins
        self.bin_ranges = bin_ranges

    def discretize(self, state):
        """Discretizes a Lunar Lander environment state into given bin ranges"""
        # Stores bin indices for each dimension in environment state
        indices = []

        # Iterates through each dimension (value) and bins it
        for value, (low, high), num_bins in zip(state, self.bin_ranges, self.num_bins):
            clipped_value = np.clip(value, low, high)
            normalized_value = (clipped_value - low) / (high - low)
            bin_idx = int(normalized_value * num_bins)
            bin_idx = min(bin_idx, num_bins - 1)
            indices.append(bin_idx)

        return tuple(indices)


