import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from models.clap_encoder import ClapEncoder

# First 5 should keep the signal, last 5 should remove the signal
EXAMPLE_QUERIES = [
    "Keep the speech",
    "Enhance the speech",
    
    "Remove the noise",
    "Filter out the noise",
    "Reduce the noise",

    "Keep the noise",
    "Remove the speech",
    "Filter out the speech",
    "Enhance the noise",
    "Reduce the speech",
]

EXAMPLE_QUERIES = [
    "Keep the male voice",
    "Enhance the male voice",
    
    "Remove the female voice",
    "Filter out the female voice",
    "Reduce the female voice",

    "Keep the female voice",
    "Remove the male voice",
    "Filter out the male voice",
    "Enhance the female voice",
    "Reduce the male voice",
]


class ClapDistance:
    def __init__(self) -> None:
        self.clap_encoder = ClapEncoder().eval()
        
    def __call__(self, query1, query2):
        query_embeddings = self.clap_encoder(
            modality="text", text=[query1, query2],
        )
        distance = torch.nn.functional.cosine_similarity(
            query_embeddings[0].unsqueeze(0),
            query_embeddings[1].unsqueeze(0)
        )
        return distance.item()


def main(queries):
    # 1. Load the model
    print("Loading the query encoder (CLAP)...")
    query_encoder = ClapEncoder().eval()

    # 2. Get the query embeddings
    print("Getting the query embeddings...")
    query_embeddings = query_encoder(
        modality="text", text=queries,
    )

    # Get the distance matrix between the query embeddings
    print("Computing the distance between each pair of query embeddings...")
    n_queries = len(queries)
    distances = np.zeros((n_queries, n_queries))
    for i in range(n_queries):
        for j in range(i + 1, n_queries):
                distance = torch.nn.functional.cosine_similarity(
                    query_embeddings[i].unsqueeze(0),
                    query_embeddings[j].unsqueeze(0)
                )
                distances[i, j] = distances[j, i] = distance.item()
    
    # 3. Plot the distances
    print("Plotting the distances...")
    sns.set_theme()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pd.DataFrame(distances, index=queries, columns=queries),
        annot=True,
        cmap="coolwarm",
        cbar=False,
    )
    plt.title("Cosine similarity between query embeddings")
    plt.tight_layout()
    plt.savefig("query_embeddings.png")
    print("Plot saved as query_embeddings.png")


if __name__ == "__main__":
    main(EXAMPLE_QUERIES)