"""Q-Network architecture for the DQN agent."""

from torch import nn

import config


class QNetwork(nn.Module):
    """Fully-connected network that maps observation vectors to Q-values.

    Architecture: input(8) -> [hidden layers with ReLU] -> output(4)
    The output is one Q-value per discrete action.
    """

    def __init__(
        self,
        state_size=config.NUM_STATE_VARIABLES,
        action_size=config.NUM_ACTIONS,
        hidden_sizes=None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_LAYER_SIZES

        layers = []
        prev_size = state_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Compute Q-values for every action given a batch of states."""
        return self.network(state)


def create_q_networks(state_size, action_size, device):
    """Create the online and target Q-networks.

    The target network starts as an exact copy of the online network and
    is updated slowly via soft updates (Polyak averaging) to stabilise
    training.

    Returns:
        (online_net, target_net) tuple, both on the specified device.
    """
    online_net = QNetwork(state_size, action_size).to(device)
    target_net = QNetwork(state_size, action_size).to(device)
    target_net.load_state_dict(online_net.state_dict())
    return online_net, target_net


def soft_update(online_net, target_net, tau=config.TAU):
    """Polyak-average the target network towards the online network.

    target_param = tau * online_param + (1 - tau) * target_param
    """
    for target_param, online_param in zip(
        target_net.parameters(), online_net.parameters()
    ):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )
