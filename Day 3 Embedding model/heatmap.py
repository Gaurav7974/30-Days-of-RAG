import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_heatmap(sim_matrix, labels, title, save_path=None):
    """
    Renders a cosine similarity heatmap for a given model.
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=False,
        square=True,
        cbar=True
    )

    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
