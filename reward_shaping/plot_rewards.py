import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def update(i):
    plt.clf()
    detailed_view = True #False #True  # Set to False for sum of rewards per episode
    log_file = f'{sys.argv[1]}'

    if os.path.isfile(log_file):
        data = pd.read_csv(log_file, header=None, names=['reach_reward', 'grasp_reward', 'lift_reward', 'hover_reward'])
        data = data.iloc[::-1]

        if detailed_view:
            # Plot all the data points without aggregating them by episodes
            # Reverse data and group them by episodes
            for column in ['reach_reward', 'grasp_reward', 'lift_reward', 'hover_reward']:
                plt.plot(data.index, data[column][::-1], label=column)  # Data is reversed
            plt.xlabel('Steps')
        else:
            # Reverse data and group them by episodes
            data = data.iloc[::-1]
            # Define horizon as the number of steps per episode
            horizon = 300

            num_episodes = (len(data) + horizon - 1) // horizon
            episodes = np.repeat(np.arange(num_episodes, -1, -1), horizon)[:len(data)]
            data['episode'] = episodes

            summed_by_episode_data = data.groupby('episode').sum()

            for column in ['reach_reward', 'grasp_reward', 'lift_reward', 'hover_reward']:
                plt.plot(summed_by_episode_data.index, summed_by_episode_data[column], label=column)

            plt.xlabel('Episodes')

        plt.ylabel(f'Reward Components')
        plt.title(f'{log_file}')
        plt.legend()
def main():
    figure = plt.figure(figsize=(10, 7))
    ani = FuncAnimation(figure, update, interval=10000)  # update every minute
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rewards_to_heatmap.py [rewards_file]")
        sys.exit(1)
    main()
