import os
import sys
import numpy as np
import pandas as pd
from PIL import Image


def rewards_to_images(log_file):
    if os.path.isfile(log_file):
        # Load and preprocess the data
        data = pd.read_csv(log_file, header=None, names=['reach_reward', 'grasp_reward', 'lift_reward', 'hover_reward'])

        # Find the maximum values for scaling
        max_values_pixel1 = data[['reach_reward', 'grasp_reward', 'lift_reward']].max().max()
        max_values_pixel2 = data[['grasp_reward', 'lift_reward', 'hover_reward']].max().max()

        # Scale the rewards for each pixel type
        pixel1_data = (data[['reach_reward', 'grasp_reward', 'lift_reward']].values / max_values_pixel1 * 255).astype(
            np.uint8)
        pixel2_data = (data[['grasp_reward', 'lift_reward', 'hover_reward']].values / max_values_pixel2 * 255).astype(
            np.uint8)

        # Reshape the data to create nearly square images
        side_length = int(np.sqrt(len(pixel1_data)))  # Estimate the side length for a nearly square image
        pixel1_data = pixel1_data[:side_length ** 2].reshape((side_length, side_length, 3))
        pixel2_data = pixel2_data[:side_length ** 2].reshape((side_length, side_length, 3))

        # Create and show the images
        image1 = Image.fromarray(pixel1_data, 'RGB')
        image1.show()

        image2 = Image.fromarray(pixel2_data, 'RGB')
        image2.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rewards_to_images.py [rewards_file]")
        sys.exit(1)
    rewards_to_images(sys.argv[1])