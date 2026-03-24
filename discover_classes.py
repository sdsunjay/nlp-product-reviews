"""Discover natural clusters in review data and map them to binary labels.

Uses TF-IDF embeddings + KMeans clustering to find sub-groups,
then maps each cluster to human_tag (0 or 1) via majority vote.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    silhouette_score,
)
from sklearn.preprocessing import normalize

import config


def load_data(file_path: str, sample_n: int = 0) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["clean_text", "human_tag"])
    df["human_tag"] = df["human_tag"].astype(int)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=config.RANDOM_SEED).reset_index(drop=True)
    return df


def generate_tfidf_embeddings(texts: list[str], n_components: int = 100) -> np.ndarray:
    """TF-IDF + SVD dimensionality reduction."""
    print(f"Building TF-IDF matrix ({len(texts)} texts)...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts)

    n_components = min(n_components, tfidf.shape[1] - 1)
    print(f"Reducing to {n_components} dimensions via SVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    reduced = svd.fit_transform(tfidf)
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.1%}")
    return normalize(reduced)


def find_optimal_k(embeddings: np.ndarray, k_range: range) -> dict:
    """Run KMeans for each k, return silhouette scores and inertias."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        results[k] = {"silhouette": sil, "inertia": km.inertia_}
        print(f"  k={k:>2d}  silhouette={sil:.4f}  inertia={km.inertia_:.0f}")
    return results


def map_clusters_to_labels(cluster_ids: np.ndarray, true_labels: np.ndarray) -> dict[int, int]:
    """Map each cluster to a binary label, adjusting for class imbalance.

    A cluster is assigned label 1 if the proportion of label-1 samples in
    that cluster is significantly higher than the global base rate.
    """
    global_rate = true_labels.mean()
    mapping = {}
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        cluster_rate = true_labels[mask].mean()
        # Assign label 1 if the cluster's label-1 rate is at least 1.5x the
        # global rate — i.e. the cluster is enriched for the minority class.
        mapping[int(c)] = 1 if cluster_rate > global_rate * 1.5 else 0
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Discover classes in review data")
    parser.add_argument("--data", default="data/clean_training3.csv", help="Path to training CSV")
    parser.add_argument("--sample", type=int, default=10000, help="Number of rows to sample (0=all)")
    parser.add_argument("--k-min", type=int, default=2, help="Min clusters to try")
    parser.add_argument("--k-max", type=int, default=20, help="Max clusters to try")
    parser.add_argument("--svd-dims", type=int, default=100, help="SVD dimensions for TF-IDF reduction")
    parser.add_argument("--output", default="cluster_mapping.json", help="Output mapping file")
    args = parser.parse_args()

    # 1. Load data
    print(f"Loading data from {args.data}...")
    df = load_data(args.data, sample_n=args.sample)
    print(f"  {len(df)} rows loaded (label distribution: {dict(df['human_tag'].value_counts().sort_index())})")

    texts = df["clean_text"].tolist()
    true_labels = df["human_tag"].values

    # 2. Generate embeddings
    embeddings = generate_tfidf_embeddings(texts, n_components=args.svd_dims)

    # 3. Find optimal k
    k_range = range(args.k_min, args.k_max + 1)
    print(f"\nSearching for optimal k in [{args.k_min}, {args.k_max}]...")
    scores = find_optimal_k(embeddings, k_range)

    best_k = max(scores, key=lambda k: scores[k]["silhouette"])
    print(f"\nBest k = {best_k} (silhouette = {scores[best_k]['silhouette']:.4f})")

    # 5. Final clustering with best k
    km = KMeans(n_clusters=best_k, random_state=config.RANDOM_SEED, n_init=10)
    cluster_ids = km.fit_predict(embeddings)

    # 6. Map clusters to binary labels
    mapping = map_clusters_to_labels(cluster_ids, true_labels)
    predicted = np.array([mapping[c] for c in cluster_ids])

    print(f"\nCluster -> Label mapping:")
    for c in sorted(mapping):
        mask = cluster_ids == c
        size = mask.sum()
        label_dist = np.bincount(true_labels[mask], minlength=2)
        purity = label_dist[mapping[c]] / size
        print(f"  Cluster {c:>2d} -> Label {mapping[c]}  (n={size}, purity={purity:.1%}, dist=[0:{label_dist[0]}, 1:{label_dist[1]}])")

    # 7. Evaluate the mapping as a classifier
    print(f"\nClassification report (cluster-based):")
    print(classification_report(true_labels, predicted))
    acc = accuracy_score(true_labels, predicted)
    print(f"Accuracy: {acc:.4f}")

    # 8. Save results
    output = {
        "best_k": best_k,
        "silhouette_scores": {str(k): v["silhouette"] for k, v in scores.items()},
        "cluster_to_label": {str(k): v for k, v in mapping.items()},
        "accuracy": acc,
        "sample_size": len(df),
        "data_file": args.data,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
