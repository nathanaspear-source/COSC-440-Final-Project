import itertools
import numpy as np

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

class GridSearch:
    def __init__(self, epsilon_start = [0.5, 1], epsilon_end = [0.1, 0.01], epsilon_decay = [0.99, 0.995, 0.999], num_episodes = 1000, seed = None, agent_type = "dqn"):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.agent_type = agent_type

    def run(self, num_episodes = 1000, render_mode = "headless", seed = None):
        """Runs grid search on all combinations of hyperparameters"""
        results = []

        # Creates a list of all possible hyperparameter combinations
        combinations = list(itertools.product(self.epsilon_start, self.epsilon_end, self.epsilon_decay))
        print(f"Running {len(combinations)} combinations of hyperparameters across {os.cpu_count()} CPU cores")

        # Trains agent with each hyperparameter combination using all available cores
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Stores results from each grid search hyperparameter combination in dictionary
            futures = {
                executor.submit(run_single_combination, params, num_episodes, seed, self.agent_type): params for params in combinations
            }

            # Prints ES, EE, and ED
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Completed ES = {result['epsilon_start']}, EE = {result['epsilon_end']}, ED = {result['epsilon_decay']}")

        # Calculates mean finishing reward (last 100 episodes) and overall max reward for each
        # hyperparameter combination
        for result in results:
            rewards = result["reward_history"]
            finishing_rewards = rewards[-100:]
            result["mean_finishing_reward"] = np.mean(finishing_rewards)
            result["max_reward"] = np.max(rewards)

        # Sorting results by mean finishing reward
        results.sort(key=lambda r: r["mean_finishing_reward"], reverse=True)
        print("Rank, Epsilon Start, Epsilon End, Epsilon Decay, Mean Reward, Max Reward")

        # Printing grid search results in descending order of mean finishing reward
        for rank, result in enumerate(results, start = 1):
            print(f"Rank: {rank}, ES: {result['epsilon_start']}, EE: {result['epsilon_end']}, ED: {result['epsilon_decay']}, MeanR: {result['mean_finishing_reward']}, MaxR: {result['max_reward']}")

        output_path = "grid_search_results.csv"
        field_names = ["rank", "epsilon_start", "epsilon_end", "epsilon_decay", "mean_finishing_reward", "max_reward"]

        # Writes ranked grid search results to csv file
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for rank, result in enumerate(results, start = 1):
                writer.writerow({
                    "rank": rank,
                    "epsilon_start": result["epsilon_start"],
                    "epsilon_end": result["epsilon_end"],
                    "epsilon_decay": result["epsilon_decay"],
                    "mean_finishing_reward": result["mean_finishing_reward"],
                    "max_reward": result["max_reward"],
                })

        # Plotting training convergence from grid search models
        num_hyp_combs = len(results)
        num_cols = 5
        num_rows = (num_hyp_combs + num_cols - 1) // num_cols

        # Creates subplots with same x-axis and y-axis scales
        figure, axes = plt.subplots(num_rows, num_cols, figsize = (16, 3 * num_rows), sharex = True, sharey = True)

        # Converts multidimensional axes array to 1D array for iterating
        # through subplots
        axes = axes.flatten()

        # Plotting each hyperparameter combination in subplot
        for i, result in enumerate(results):
            axis = axes[i]
            episodes = range(1, len(result["reward_history"]) + 1)
            axis.plot(episodes, result["reward_history"], linewidth = 1)
            axis.set_title(f"ES={result['epsilon_start']}, "
                           f"EE={result['epsilon_end']}, "
                           f"ED={result['epsilon_decay']}", fontsize = 6)
            axis.grid(True)

        # Hiding unused subplots in plot figure
        for j in range (len(results), len(axes)):
            axes[j].set_visible(False)

        # Setting super labels for plot figure
        if self.agent_type == "sarsa":
            figure.suptitle("Grid Search SARSA Training Reward Convergence Plots", fontsize = 9)
        else:
            figure.suptitle("Grid Search DQN Training Reward Convergence Plots", fontsize = 9)

        figure.suptitle("Grid Search Training Reward Convergence Plots", fontsize = 9)
        figure.supxlabel("Episode")
        figure.supylabel("Reward")
        figure.tight_layout(rect = (0, 0, 1, 0.95))

        # Saving plot
        path = os.path.abspath("grid_search_reward_convergence.png")
        print("Saving Plot To:", path)
        figure.savefig(path, dpi=300)

        print(output_path)
        return results

def run_single_combination(params, num_episodes, seed, agent_type):
    """"Trains single hyperparameter combination (used to run grid search in parallel)"""
    es, ee, ed = params
    gs_checkpoint = f"es{es}_ee{ee}_ed{ed}"

    if agent_type == "sarsa":
        from train_sarsa import train
    else:
        from train import train

    reward_history = train(
        num_episodes = num_episodes,
        render_mode = "headless",
        seed = seed,
        epsilon_start = es,
        epsilon_end = ee,
        epsilon_decay = ed,
        gs_checkpoint = gs_checkpoint,
        verbose = False,
    )

    return {
        "epsilon_start": es,
        "epsilon_end": ee,
        "epsilon_decay": ed,
        "reward_history": reward_history,
    }