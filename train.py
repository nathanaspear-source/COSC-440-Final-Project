"""Training loop for the DQN Lunar Lander agent.

Runs episodes, feeds transitions to the agent, logs rolling statistics,
and saves checkpoints when the environment is solved.
"""

import os
from collections import deque

import numpy as np

import config
from dqn_agent import DQNAgent
from lunar_lander_env import create_environment


def train(epsilon_start = None, epsilon_end = None, epsilon_decay = None, num_episodes=None, render_mode="headless", gs_checkpoint = None, seed=42, verbose = True):
    """Train the DQN agent and return episode reward history.

    Args:
        num_episodes: Total training episodes (default from config).
        render_mode:  Render mode passed to the environment factory.
        seed:         Random seed for reproducibility.

    Returns:
        List of total rewards, one per episode.
    """
    if num_episodes is None:
        num_episodes = config.NUM_TRAINING_EPISODES

    env = create_environment(render_mode=render_mode)
    agent = DQNAgent(config.NUM_STATE_VARIABLES, config.NUM_ACTIONS, epsilon_start, epsilon_end, epsilon_decay, seed=seed)

    reward_history = []
    recent_rewards = deque(maxlen=config.SOLVE_WINDOW)
    solved = False

    if verbose:
        print(f"Training DQN agent for up to {num_episodes} episodes ...")
        print(f"  Device:  {agent.device}")
        print(f"  Solve:   mean reward >= {config.SOLVE_THRESHOLD} "
              f"over {config.SOLVE_WINDOW} episodes\n")

    for episode in range(1, num_episodes + 1):
        episode_reward = _run_training_episode(env, agent, seed + episode)

        reward_history.append(episode_reward)
        recent_rewards.append(episode_reward)
        agent.decay_epsilon()

        mean_reward = np.mean(recent_rewards)

        if verbose and episode % config.LOG_INTERVAL == 0:
            _log_progress(episode, num_episodes, episode_reward,
                          mean_reward, agent.epsilon)

        if not solved and mean_reward >= config.SOLVE_THRESHOLD:
            solved = True

            # Adds GridSearch checkpoint number to solved file name
            solved_filename = ""
            if gs_checkpoint is not None:
                solved_filename = f"{gs_checkpoint}_solved.pth"
            else:
                solved_filename = "solved.pth"

            checkpoint_filename = os.path.join(config.CHECKPOINT_DIR, solved_filename)
            agent.save(checkpoint_filename)

            if verbose:
                print(f"\n*** SOLVED at episode {episode}  "
                      f"(mean {mean_reward:+.2f}) ***")
                print(f"    Checkpoint saved to {checkpoint_filename}\n")

    # Creates final file name and saves file
    final_name = ""
    if gs_checkpoint is not None:
        final_name = f"{gs_checkpoint}_final.pth"
    else:
        final_name = "final.pth"

    final_path = os.path.join(config.CHECKPOINT_DIR, final_name)
    agent.save(final_path)

    if verbose:
        print(f"\nTraining complete. Final weights saved to {final_path}")

    env.close()
    return reward_history


def _run_training_episode(env, agent, seed):
    """Execute one episode, feeding every transition to the agent."""
    state, _info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _info = env.step(action)

        agent.step(state, action, reward, next_state,
                   terminated or truncated)

        state = next_state
        total_reward += reward

    return total_reward


def _log_progress(episode, total, reward, mean_reward, epsilon):
    """Print a compact progress line."""
    print(f"Episode {episode:5d}/{total} | "
          f"Reward: {reward:+8.2f} | "
          f"Mean({config.SOLVE_WINDOW}): {mean_reward:+8.2f} | "
          f"Epsilon: {epsilon:.4f}")
