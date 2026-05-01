"""Wrapper module for the Gymnasium Lunar Lander environment.

Provides factory creation, state inspection utilities, and a random-action
demo loop that can run with or without rendering.
"""

import gymnasium as gym
import numpy as np

import config


def create_environment(render_mode="headless"):
    """Create and return a configured LunarLander-v3 environment.

    Args:
        render_mode: One of "human", "rgb_array", or "headless".
            "human"     -- opens a pygame window for live viewing.
            "rgb_array" -- returns pixel arrays (useful for recording).
            "headless"  -- no rendering, fastest for training.

    Returns:
        A Gymnasium environment instance.
    """
    resolved_mode = config.RENDER_MODES[render_mode]

    env = gym.make(
        config.ENV_ID,
        render_mode=resolved_mode,
        gravity=config.GRAVITY,
        enable_wind=config.ENABLE_WIND,
        wind_power=config.WIND_POWER,
        turbulence_power=config.TURBULENCE_POWER,
        continuous=config.CONTINUOUS_ACTION_SPACE,
    )
    return env


def describe_environment(env):
    """Print a summary of the environment's observation and action spaces."""
    print(f"Environment: {config.ENV_ID}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print()
    print("Observation vector breakdown:")
    for i, label in enumerate(config.STATE_LABELS):
        low = env.observation_space.low[i]
        high = env.observation_space.high[i]
        print(f"  [{i}] {label:25s}  range: [{low:.4f}, {high:.4f}]")
    print()
    print("Discrete actions:")
    for action_id, description in config.DISCRETE_ACTIONS.items():
        print(f"  {action_id} -> {description}")


def format_state(state):
    """Return a human-readable dict mapping state labels to values."""
    return {
        label: round(float(value), 4)
        for label, value in zip(config.STATE_LABELS, state)
    }


def _sample_random_action(env):
    """Return an action-selection callable that samples uniformly."""
    def _pick(_state):
        return env.action_space.sample()
    return _pick


def run_episode(env, action_fn=None, seed=None, verbose=True):
    """Run a single episode with a configurable action-selection function.

    Args:
        env:       A Gymnasium environment instance.
        action_fn: Callable(state) -> action.  Defaults to random sampling.
        seed:      Optional RNG seed for reproducibility.
        verbose:   If True, print step-by-step details.

    Returns:
        A dict with episode statistics: total_reward, steps, and
        final_state.
    """
    if action_fn is None:
        action_fn = _sample_random_action(env)

    state, _info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = action_fn(state)
        next_state, reward, terminated, truncated, _info = env.step(action)

        total_reward += reward
        steps += 1

        if verbose:
            print(
                f"Step {steps:4d} | "
                f"Action: {config.DISCRETE_ACTIONS.get(action, str(action)):20s} | "
                f"Reward: {reward:+8.2f} | "
                f"Cumulative: {total_reward:+10.2f}"
            )

        state = next_state

    if verbose:
        print(f"\nEpisode finished: "
              f"{'LANDED' if terminated and total_reward > 0 else 'CRASHED'}")
        print(f"  Total reward: {total_reward:+.2f}")
        print(f"  Steps:        {steps}")
        print(f"  Final state:  {format_state(state)}")

    return {
        "total_reward": total_reward,
        "steps": steps,
        "final_state": format_state(state),
    }


def run_random_episode(env, seed=None, verbose=True):
    """Run a single episode using random actions (convenience wrapper)."""
    return run_episode(env, action_fn=None, seed=seed, verbose=verbose)


def run_random_episodes(env, num_episodes=10, seed=None, verbose=False):
    """Run multiple random-action episodes and report aggregate statistics.

    Args:
        env:          A Gymnasium environment instance.
        num_episodes: Number of episodes to run.
        seed:         Optional base seed (incremented per episode).
        verbose:      If True, print per-step details for every episode.

    Returns:
        A dict with mean_reward, std_reward, mean_steps, min_reward,
        and max_reward.
    """
    rewards = []
    step_counts = []

    for i in range(num_episodes):
        episode_seed = (seed + i) if seed is not None else None
        result = run_random_episode(env, seed=episode_seed, verbose=verbose)
        rewards.append(result["total_reward"])
        step_counts.append(result["steps"])
        print(
            f"Episode {i + 1:3d}/{num_episodes} | "
            f"Reward: {result['total_reward']:+10.2f} | "
            f"Steps: {result['steps']}"
        )

    rewards_arr = np.array(rewards)
    return {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_steps": float(np.mean(step_counts)),
        "min_reward": float(np.min(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
    }
