# COSC 440 Final Project: Reinforcement Learning Agents in Lunar Lander Environment
In this project, I implemented the Lunar Lander environment from the Gymnasium library,
along with a DQN agent, discretized SARSA agent, and random agent that can navigate the environment.
Grid search can also be used with the SARSA and DQN agent for hyperparameter tuning.

## Features
- Lunar Lander environment (text-based, array-based, and visual simulation)
- DQN agent, discretized SARSA agent, and random agent that can navigate environment
- Grid search can be used to tune over epsilon start, epsilon end, and epsilon decay
    for DQN and discretized SARSA agents
- Matplotlib is used to display convergence plots after training and grid search

## Installing Dependencies
In your IDE's terminal, run: pip install -r requirements.txt

## Running Project
To run this project, CLI commands entered in the IDE's terminal are used.
Here are all of the available CLI commands:

demo: Used to run episodes with random agent
    --render controls how environment is displayed (human for visual simulate, rgb_array for array based, and headless for CLI based)
    --episodes controls number of episodes the agent will navigate
    --seed sets seed for environment
    --verbose outputs updates about agent's performance to CLI interface

train: Used to train DQN agent
    --render controls how environment is displayed (human for visual simulate, rgb_array for array based, and headless for CLI based)
    --episodes controls number of episodes the agent will navigate
    --seed sets seed for environment
    --verbose outputs updates about agent's performance to CLI interface

evaluate: Used to evaluate saved DQN in .pth file after training
    --checkpoint inputs the saved DQN .pth file
    --render controls how environment is displayed (human for visual simulate, rgb_array for array based, and headless for CLI based)
    --episodes controls number of episodes the agent will navigate
    --seed sets seed for environment
    --verbose outputs updates about agent's performance to CLI interface

sarsa: Used to train discretized SARSA agent
    --render controls how environment is displayed (human for visual simulate, rgb_array for array based, and headless for CLI based)
    --episodes controls number of episodes the agent will navigate
    --seed sets seed for environment
    --verbose outputs updates about agent's performance to CLI interface

gridsearch: Used to perform grid search on DQN or discretized SARSA agent
    --episodes controls number of episodes each agent will navigate
    --seed sets seed for environment
    --epsilon-starts sets one or more values of epsilon start the grid search will tune over
    --epsilon-ends sets one or more values of epsilon end the grid search will tune over
    --epsilon-decays sets one or more values of epslion decay the grid seasrch will tune over
    --agent sets agent that grid search will use (dqn or sarsa)