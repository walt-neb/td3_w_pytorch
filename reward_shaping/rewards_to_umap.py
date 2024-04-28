import sys
import pandas as pd
import umap
import matplotlib.pyplot as plt


def rewards_to_umap(log_file, n_components=2):
    # Load the data
    data = pd.read_csv(log_file, header=None, names=['reach_reward', 'grasp_reward', 'lift_reward', 'hover_reward'])

    # Scale the data (UMAP works best with scaled data)
    scaled_data = data / data.max().max()

    # Fit UMAP
    reducer = umap.UMAP(n_components=n_components)
    embedding = reducer.fit_transform(scaled_data)

    # Plot the embedding
    plt.figure(figsize=(8, 6))
    if n_components == 2:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
    elif n_components == 3:
        ax = plt.figure(figsize=(8, 6)).add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
    plt.title('UMAP projection of Reward Components')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rewards_to_images.py [rewards_file]")
        sys.exit(1)
    rewards_to_umap(sys.argv[1])