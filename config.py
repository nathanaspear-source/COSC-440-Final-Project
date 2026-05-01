"""Configuration constants for the Lunar Lander environment and DQN agent."""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_ID = "LunarLander-v3"

RENDER_MODES = {
    "human": "human",
    "rgb_array": "rgb_array",
    "headless": None,
}

GRAVITY = -10.0
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 1.5

CONTINUOUS_ACTION_SPACE = False

NUM_STATE_VARIABLES = 8
STATE_LABELS = [
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
]

DISCRETE_ACTIONS = {
    0: "Do Nothing",
    1: "Fire Left Engine",
    2: "Fire Main Engine",
    3: "Fire Right Engine",
}

NUM_ACTIONS = len(DISCRETE_ACTIONS)

# ---------------------------------------------------------------------------
# DQN Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 100_000

# Epsilon-greedy exploration schedule
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Target network soft-update rate
TAU = 1e-3

# Network architecture (hidden layer sizes)
HIDDEN_LAYER_SIZES = [128, 128]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_TRAINING_EPISODES = 1000
SOLVE_THRESHOLD = 200.0
SOLVE_WINDOW = 100
LOG_INTERVAL = 10
CHECKPOINT_DIR = "checkpoints"

# ---------------------------------------------------------------------------
# State Discretizer
# ---------------------------------------------------------------------------
SARSA_LEARNING_RATE = 0.15
SARSA_GAMMA = 0.99
SARSA_NUM_EPISODES = 5000
SARSA_EPSILON_START = 1.0
SARSA_EPSILON_END = 0.01
SARSA_EPSILON_DECAY = 0.9996
NUM_BINS = [6, 6, 6, 6, 6, 6, 2, 2]
STATE_BIN_RANGES = [
            (-1.0, 1.0),
            (0.0, 1.5),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-1.0, 1.0),
            (-3.0, 3.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ]
