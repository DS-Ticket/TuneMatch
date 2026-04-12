"""
Evaluation metrics for TuneMatch recommendation models.

Metrics:
  - Precision@K    — fraction of top-K recommendations that are relevant
  - Recall@K       — fraction of relevant items found in top-K
  - NDCG@K         — normalized discounted cumulative gain (ranked quality)
  - Coverage        — fraction of the catalog recommended across all users
  - Diversity       — avg pairwise vibe-distance within a playlist (higher = more varied)
  - Novelty         — avg -log(popularity) (higher = more obscure / surprising)
"""

import numpy as np
import pandas as pd
from loguru import logger


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-k recommended items that are relevant."""
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-k recommendations."""
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not recommended or not relevant:
        return 0.0

    def dcg(items, rel, k):
        score = 0.0
        for i, item in enumerate(items[:k]):
            if item in rel:
                score += 1.0 / np.log2(i + 2)
        return score

    actual_dcg  = dcg(recommended, relevant, k)
    ideal_items = list(relevant)[:k]
    ideal_dcg   = dcg(ideal_items, relevant, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def coverage(all_recommendations: list[list], catalog_size: int) -> float:
    """Fraction of catalog items recommended at least once."""
    recommended_set = set(item for recs in all_recommendations for item in recs)
    return len(recommended_set) / catalog_size


def diversity(playlist_df: pd.DataFrame, vibe_features: list[str]) -> float:
    """
    Average pairwise cosine distance within a playlist.
    Higher = more diverse vibe coverage.
    """
    present = [c for c in vibe_features if c in playlist_df.columns]
    if len(present) < 2 or len(playlist_df) < 2:
        return 0.0

    matrix = playlist_df[present].values
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / (norms + 1e-8)
    sim_matrix = normed @ normed.T

    n = len(playlist_df)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += 1 - sim_matrix[i, j]
            count += 1

    return total_dist / count if count > 0 else 0.0


def novelty(playlist_df: pd.DataFrame) -> float:
    """
    Mean -log(popularity) across playlist.
    Popularity is assumed to be 0-100; we normalize to 0-1 first.
    Higher novelty = more obscure songs.
    """
    if "popularity" not in playlist_df.columns:
        return 0.0
    pops = (playlist_df["popularity"] / 100.0).clip(0.01, 1.0)
    return float(-np.log(pops).mean())


def evaluate_model(
    recommended_lists: list[list],
    relevant_lists: list[set],
    k_values: list[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Batch evaluation across multiple users/queries.

    Args:
        recommended_lists:  list of recommendation lists (one per user)
        relevant_lists:     list of ground-truth relevant item sets
        k_values:           list of K values to evaluate at

    Returns:
        DataFrame with mean metrics at each K
    """
    results = []
    for k in k_values:
        precisions = [precision_at_k(rec, rel, k) for rec, rel in zip(recommended_lists, relevant_lists)]
        recalls    = [recall_at_k(rec, rel, k)    for rec, rel in zip(recommended_lists, relevant_lists)]
        ndcgs      = [ndcg_at_k(rec, rel, k)      for rec, rel in zip(recommended_lists, relevant_lists)]

        results.append({
            "k":            k,
            "precision@k":  np.mean(precisions),
            "recall@k":     np.mean(recalls),
            "ndcg@k":       np.mean(ndcgs),
        })
        logger.info(f"@{k:2d}: P={np.mean(precisions):.4f}  R={np.mean(recalls):.4f}  NDCG={np.mean(ndcgs):.4f}")

    return pd.DataFrame(results)


def compare_models(model_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Tabulate results from multiple models side-by-side.
    model_results: {"model_name": metrics_df, ...}
    """
    rows = []
    for model_name, df in model_results.items():
        for _, row in df.iterrows():
            rows.append({"model": model_name, **row.to_dict()})
    return pd.DataFrame(rows)
