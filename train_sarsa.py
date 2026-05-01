"""Training loop for the Discretized SARSA Lunar Lander agent.

Runs episodes, feeds transitions to the agent, logs rolling statistics,
and saves checkpoints when the environment is solved.
"""

import os
from collections import deque

import numpy as np

import config
from sarsa_agent import SARSAAgent
from state_discretizer import StateDiscretizer
from lunar_lander_env import create_environment


def train(epsilon_start = None, epsilon_end = None, epsilon_decay = None, num_episodes=None, render_mode="headless", gs_checkpoint = None, seed=42, verbose = True):
    """Train the SARSA agent and return episode reward history.

    Args:
        num_episodes: Total training episodes (default from config).
        render_mode:  Render mode passed to the environment factory.
        seed:         Random seed for reproducibility.

    Returns:
        List of total rewards, one per episode.
    """
    if num_episodes is None:
        num_episodes = config.SARSA_TRAINING_EPISODES

    if epsilon_start is None:
        epsilon_start = config.SARSA_EPSILON_START
    if epsilon_end is None:
        epsilon_end = config.SARSA_EPSILON_END
    if epsilon_decay is None:
        epsilon_decay = config.SARSA_EPSILON_DECAY

    env = create_environment(render_mode=render_mode)
    agent = SARSAAgent(config.NUM_ACTIONS,
                       gamma = config.SARSA_GAMMA,
                       alpha = config.SARSA_LEARNING_RATE,
                       epsilon_start = epsilon_start,
                       epsilon_end = epsilon_end,
                       epsilon_decay = epsilon_decay,
                       seed = seed,)

    reward_history = []
    recent_rewards = deque(maxlen=config.SOLVE_WINDOW)
    solved = False

    if verbose:
        print(f"Training SARSA agent for up to {num_episodes} episodes ...")
        print(f"  Solve:   mean reward >= {config.SOLVE_THRESHOLD} "
              f"over {config.SOLVE_WINDOW} episodes\n")

    discretizer = StateDiscretizer(config.NUM_BINS, config.STATE_BIN_RANGES)
    for episode in range(1, num_episodes + 1):
        episode_reward = _run_training_episode(env, agent, discretizer, seed + episode)

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
                solved_filename = f"{gs_checkpoint}_solved.pkl"
            else:
                solved_filename = "solved.pkl"

            checkpoint_filename = os.path.join(config.CHECKPOINT_DIR, solved_filename)
            agent.save(checkpoint_filename)

            if verbose:
                print(f"\n*** SOLVED at episode {episode}  "
                      f"(mean {mean_reward:+.2f}) ***")
                print(f"    Checkpoint saved to {checkpoint_filename}\n")

    # Creates final file name and saves file
    final_name = ""
    if gs_checkpoint is not None:
        final_name = f"{gs_checkpoint}_final.pkl"
    else:
        final_name = "final.pkl"

    final_path = os.path.join(config.CHECKPOINT_DIR, final_name)
    agent.save(final_path)

    if verbose:
        print(f"\nTraining complete. Final weights saved to {final_path}")

    env.close()
    return reward_history


def _run_training_episode(env, agent, discretizer, seed):
    """Execute one episode, feeding every transition to the agent."""
    prev_state, _info = env.reset(seed = seed)
    state = discretizer.discretize(prev_state)
    action = agent.select_action(state)
    total_reward = 0.0

    done = False
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretizer.discretize(next_state)
        next_action = agent.select_action(next_state)
        agent.update(state, action, reward, next_state, next_action, done)
        state, action = next_state, next_action
        total_reward += reward

    return total_reward

def _log_progress(episode, total, reward, mean_reward, epsilon):
    """Print a compact progress line."""
    print(f"Episode {episode:5d}/{total} | "
          f"Reward: {reward:+8.2f} | "
          f"Mean({config.SOLVE_WINDOW}): {mean_reward:+8.2f} | "
          f"Epsilon: {epsilon:.4f}")
