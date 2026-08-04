"""
Collaborative Filtering — Alternating Least Squares (ALS) via the
`implicit` library, which is purpose-built for implicit feedback
(play counts, not explicit ratings).

Why ALS over SVD/NMF for this problem:
  - Play counts are implicit (no explicit 1-5 star ratings)
  - ALS with confidence weighting handles this correctly
  - Scales to millions of users × tracks

Saves model to: models/als_model.pkl
"""

import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml
from loguru import logger

try:
    import implicit
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False
    logger.warning("implicit library not installed. Run: pip install implicit")


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_interaction_matrix(
    interactions: pd.DataFrame,
    min_interactions: int = 5,
) -> tuple[sparse.csr_matrix, list, list]:
    """
    Convert interactions DataFrame to a sparse user×item matrix.

    Returns:
        matrix:   scipy CSR sparse matrix (users × tracks)
        user_ids: list of user IDs (row index → user_id)
        track_ids: list of track IDs (col index → track_id)
    """
    # Filter low-interaction users
    user_counts = interactions.groupby("user_id")["play_count"].sum()
    active_users = user_counts[user_counts >= min_interactions].index
    interactions = interactions[interactions["user_id"].isin(active_users)]

    user_ids  = sorted(interactions["user_id"].unique().tolist())
    track_ids = sorted(interactions["track_id"].unique().tolist())

    user_map  = {uid: i for i, uid in enumerate(user_ids)}
    track_map = {tid: i for i, tid in enumerate(track_ids)}

    rows = interactions["user_id"].map(user_map).values
    cols = interactions["track_id"].map(track_map).values
    data = interactions["play_count"].values.astype(np.float32)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(track_ids)),
    )
    logger.info(f"Interaction matrix: {matrix.shape} | density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")
    return matrix, user_ids, track_ids


class ALSRecommender:
    def __init__(self, factors=128, iterations=30, regularization=0.01, alpha=40):
        if not HAS_IMPLICIT:
            raise ImportError("pip install implicit")
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
        )
        self.alpha     = alpha
        self.user_ids  = None
        self.track_ids = None
        self.matrix    = None

    def fit(
        self,
        interactions: pd.DataFrame,
        min_interactions: int = 5,
    ) -> "ALSRecommender":
        self.matrix, self.user_ids, self.track_ids = build_interaction_matrix(
            interactions, min_interactions
        )
        # ALS expects item×user for training
        confidence_matrix = (self.matrix * self.alpha).astype(np.float32)
        self.model.fit(confidence_matrix.T)
        logger.info(f"ALS trained: {len(self.user_ids):,} users, {len(self.track_ids):,} tracks")
        return self

    def recommend_for_user(
        self,
        user_id: str | int,
        n: int = 20,
        filter_already_liked: bool = True,
    ) -> list[tuple[str | int, float]]:
        """Returns [(track_id, score), ...]"""
        if user_id not in self.user_ids:
            raise KeyError(f"Unknown user: {user_id}")
        user_idx = self.user_ids.index(user_id)
        ids, scores = self.model.recommend(
            user_idx,
            self.matrix[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )
        return [(self.track_ids[i], float(s)) for i, s in zip(ids, scores)]

    def get_similar_tracks(
        self,
        track_id: str | int,
        n: int = 20,
    ) -> list[tuple[str | int, float]]:
        """Find similar tracks using item factor similarity."""
        if track_id not in self.track_ids:
            raise KeyError(f"Unknown track: {track_id}")
        track_idx = self.track_ids.index(track_id)
        ids, scores = self.model.similar_items(track_idx, N=n + 1)
        return [(self.track_ids[i], float(s)) for i, s in zip(ids, scores) if i != track_idx][:n]

    def get_item_factors(self) -> np.ndarray:
        return self.model.item_factors

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved ALS model → {path}")

    @staticmethod
    def load(path: Path) -> "ALSRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


def train(config_path: str = "config/config.yaml") -> ALSRecommender | None:
    if not HAS_IMPLICIT:
        logger.error("Cannot train ALS — `implicit` not installed")
        return None

    config    = load_config(config_path)
    proc_dir  = Path(config["paths"]["processed_data"])
    model_dir = Path(config["paths"]["models"])

    interactions_path = proc_dir / "interactions.parquet"
    if not interactions_path.exists():
        logger.warning(f"{interactions_path} not found — skipping collaborative filtering")
        return None

    interactions = pd.read_parquet(interactions_path)
    if interactions.empty:
        logger.warning("Interactions table is empty — skipping ALS training")
        return None

    cfg = config["models"]["als"]
    eval_cfg = config["evaluation"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="als_collaborative_filtering"):
        mlflow.log_params(cfg)

        recommender = ALSRecommender(
            factors=cfg["factors"],
            iterations=cfg["iterations"],
            regularization=cfg["regularization"],
            alpha=cfg["alpha"],
        )
        recommender.fit(interactions, min_interactions=eval_cfg["min_interactions"])

        out_path = model_dir / "als_model.pkl"
        recommender.save(out_path)
        mlflow.log_artifact(str(out_path))
        mlflow.log_metric("n_users",  len(recommender.user_ids))
        mlflow.log_metric("n_tracks", len(recommender.track_ids))

    return recommender


if __name__ == "__main__":
    train()
