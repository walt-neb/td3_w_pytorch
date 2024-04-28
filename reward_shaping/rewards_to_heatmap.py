import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def rewards_to_heatmap(log_file):
    # Load the data
    data = pd.read_csv(log_file, header=None, names=['reach_reward', 'grasp_reward', 'lift_reward', 'hover_reward'])

    # Assuming each row is a step and you want to reshape this into a 2D structure where each row is an episode
    horizon = 300  # Number of steps per episode - adjust this if different
    num_episodes = len(data) // horizon

    # Reshape the data for the heatmap - one reward type at a time for clarity
    for i, column in enumerate(data.columns):
        reward_data = data[column].values[:num_episodes * horizon].reshape(num_episodes, horizon)

        plt.figure(i, figsize=(10, 6))  # Create a new figure for each reward component
        sns.heatmap(reward_data, cmap='viridis')
        plt.title(f"Heatmap for {column}")
        plt.xlabel("Step")
        plt.ylabel("Episode")
        plt.show(block=False)  # Display the figure without blocking the execution
    input(f'Press Enter to exit')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rewards_to_heatmap.py [rewards_file]")
        sys.exit(1)
    rewards_to_heatmap(sys.argv[1])

