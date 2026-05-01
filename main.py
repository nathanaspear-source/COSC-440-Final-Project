"""Entry point for the Lunar Lander reinforcement learning project.

Supports three modes:
    demo     -- describe the environment and run random-action episodes
    train    -- train the DQN agent from scratch
    evaluate -- load a trained checkpoint and watch the agent perform
"""

import argparse
import os
import matplotlib.pyplot as plt

import config
from lunar_lander_env import (
    create_environment,
    describe_environment,
    run_episode,
    run_random_episode,
    run_random_episodes,
)


def parse_args():
    """Parse command-line arguments for mode, rendering, and training."""
    parser = argparse.ArgumentParser(
        description="Lunar Lander -- DQN Reinforcement Learning"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- demo ----------------------------------------------------------
    demo = subparsers.add_parser("demo", help="Random-agent baseline")
    demo.add_argument("--render", choices=["human", "rgb_array", "headless"],
                      default="headless")
    demo.add_argument("--episodes", type=int, default=10)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--verbose", action="store_true")

    # --- train ---------------------------------------------------------
    tr = subparsers.add_parser("train", help="Train the DQN agent")
    tr.add_argument("--render", choices=["human", "rgb_array", "headless"],
                    default="headless")
    tr.add_argument("--episodes", type=int, default=config.NUM_TRAINING_EPISODES)
    tr.add_argument("--seed", type=int, default=42)

    # --- evaluate DQN ------------------------------------------------------
    ev = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    ev.add_argument("--render", choices=["human", "rgb_array", "headless"],
                    default="human")
    ev.add_argument("--checkpoint", type=str,
                    default=os.path.join(config.CHECKPOINT_DIR, "final.pth"))
    ev.add_argument("--episodes", type=int, default=5)
    ev.add_argument("--seed", type=int, default=0)

    # Sarsa CLI subcommands
    sarsa = subparsers.add_parser("sarsa")
    sarsa.add_argument("--render", choices=["human", "rgb_array", "headless"], default="headless")
    sarsa.add_argument("--episodes", type=int, default = config.SARSA_NUM_EPISODES)
    sarsa.add_argument("--seed", type=int, default=42)

    # GridSearch CLI subcommands
    gs = subparsers.add_parser("gridsearch")
    gs.add_argument("--episodes", type = int, default = config.NUM_TRAINING_EPISODES)
    gs.add_argument("--seed", type = int, default = 0)
    gs.add_argument("--epsilon-starts", type = float, nargs = "+", default = [0.5, 1.0])
    gs.add_argument("--epsilon-ends", type = float, nargs = "+", default = [0.1, 0.01])
    gs.add_argument("--epsilon-decays", type = float, nargs = "+", default = [0.99, 0.995, 0.999])
    gs.add_argument("--agent", choices = ["dqn", "sarsa"], default = "dqn")


    return parser.parse_args()

def run_demo(args):
    """Describe the environment and run random-agent episodes."""
    env = create_environment(render_mode=args.render)

    print("=" * 60)
    print("LUNAR LANDER ENVIRONMENT")
    print("=" * 60)
    describe_environment(env)

    print("\n" + "=" * 60)
    print("RANDOM AGENT BASELINE")
    print("=" * 60)

    if args.episodes == 1:
        run_random_episode(env, seed=args.seed, verbose=True)
    else:
        stats = run_random_episodes(
            env, num_episodes=args.episodes,
            seed=args.seed, verbose=args.verbose,
        )
        print("\n--- Aggregate Statistics ---")
        print(f"  Mean reward: {stats['mean_reward']:+.2f}")
        print(f"  Std reward:  {stats['std_reward']:.2f}")
        print(f"  Min reward:  {stats['min_reward']:+.2f}")
        print(f"  Max reward:  {stats['max_reward']:+.2f}")
        print(f"  Mean steps:  {stats['mean_steps']:.1f}")

    env.close()


def run_train(args):
    """Train the DQN agent."""
    from train import train  # pylint: disable=import-outside-toplevel
    reward_history = train(num_episodes=args.episodes, render_mode=args.render, seed=args.seed)

    print("Plotting Training Reward Convergence")

    plt.figure()
    episodes = range(1, len(reward_history) + 1)
    plt.plot(episodes, reward_history, linestyle="-", color = "blue")
    plt.title("Training Reward Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    path = os.path.abspath("training_reward_convergence.png")
    print("Saving Plot To:", path)

    plt.savefig(path)
    plt.show()

def run_sarsa(args):
    """Train the SARSA agent."""
    from train_sarsa import train
    reward_history = train(num_episodes=args.episodes, render_mode=args.render, seed=args.seed)

    print("Plotting SARSA Training Reward Convergence")
    plt.figure()
    episodes = range(1, len(reward_history) + 1)
    plt.plot(episodes, reward_history, linestyle="-", color = "green")
    plt.title("SARSA Training Reward Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    path = os.path.abspath("sarsa_training_reward_convergence.png")
    print("Saving Plot To:", path)

    plt.savefig(path)
    plt.show()

def run_evaluate(args):
    """Load a checkpoint and run the trained agent."""
    from dqn_agent import DQNAgent  # pylint: disable=import-outside-toplevel

    env = create_environment(render_mode=args.render)
    agent = DQNAgent(config.NUM_STATE_VARIABLES, config.NUM_ACTIONS)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    for ep in range(1, args.episodes + 1):
        result = run_episode(
            env,
            action_fn=agent.select_action,
            seed=args.seed + ep,
            verbose=False,
        )
        outcome = ("LANDED" if result["total_reward"] > 0 else "CRASHED")
        print(f"Episode {ep} | {outcome} | "
              f"Reward: {result['total_reward']:+.2f} | "
              f"Steps: {result['steps']}")

    env.close()
    print("\nEvaluation complete.")

def run_gridsearch(args):
    """Runs GridSearch over epsilon hyperparameters: epsilon start, epsilon end,
    and epsilon decay."""
    from grid_search import GridSearch

    gridsearch = GridSearch(
        epsilon_start = args.epsilon_starts,
        epsilon_end = args.epsilon_ends,
        epsilon_decay = args.epsilon_decays,
        agent_type = args.agent,
    )

    gridsearch.run(
        num_episodes = args.episodes,
        seed = args.seed,
    )

def main():
    """Dispatch to the selected sub-command."""
    args = parse_args()
    dispatch = {
        "demo": run_demo,
        "train": run_train,
        "evaluate": run_evaluate,
        "gridsearch": run_gridsearch,
        "sarsa": run_sarsa,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
