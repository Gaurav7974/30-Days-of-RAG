import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from heatmap import plot_heatmap


def load_sample():
    return [
        "Vector search allows us to compare meaning instead of keywords.",
        "FAISS provides efficient similarity search over dense embeddings.",
        "Deep learning models can transform sentences into numerical vectors.",
        "Dogs are loyal animals and make good pets.",
        "Cats are independent animals and often prefer solitude.",
    ]


def embed(model_name, samples):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(samples, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def compare_models(models, samples):
    for name in models:
        print(f"\nModel: {name}")
        print("-" * (8 + len(name)))

        vectors = embed(name, samples)
        sim_matrix = cosine_similarity(vectors)

        avg_sim = np.mean(sim_matrix)
        print(f"Average similarity: {avg_sim:.4f}")

        # Display top related matches
        for i in range(len(samples)):
            sorted_idx = np.argsort(sim_matrix[i])[::-1]
            best_match = sorted_idx[1]
            score = sim_matrix[i][best_match]

            print(
                f"  '{samples[i]}'\n    → Closest: '{samples[best_match]}' (score={score:.4f})"
            )

        # Render similarity heatmap for this model
        plot_heatmap(
            sim_matrix,
            labels=[f"S{i+1}" for i in range(len(samples))],
            title=f"Cosine Similarity Heatmap — {name}",
        )


def main():
    samples = load_sample()

    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "intfloat/e5-small-v2",
    ]

    compare_models(models, samples)


if __name__ == "__main__":
    main()
