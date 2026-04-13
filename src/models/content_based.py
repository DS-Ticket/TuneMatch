"""
Content-Based Recommender — K-Nearest Neighbors on audio (vibe) vectors.

This is Layer 1 in the TuneMatch stack and handles cold-start perfectly:
no user history needed, just a seed song.

Saves model to: models/content_based.pkl
"""

import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.neighbors import NearestNeighbors


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


VIBE_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "tempo",
    "speechiness", "liveness",
]


class ContentBasedRecommender:
    def __init__(self, n_neighbors: int = 20, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric      = metric
        self.model       = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because query track is its own neighbor
            metric=metric,
            algorithm="brute",
            n_jobs=-1,
        )
        self.tracks_df   = None
        self.feature_cols = None

    def fit(self, tracks_df: pd.DataFrame) -> "ContentBasedRecommender":
        self.tracks_df    = tracks_df.reset_index(drop=True)
        self.feature_cols = [c for c in VIBE_FEATURES if c in tracks_df.columns]

        if not self.feature_cols:
            raise ValueError("No vibe features found in tracks_df")

        X = tracks_df[self.feature_cols].values
        self.model.fit(X)
        logger.info(f"Fitted K-NN on {len(tracks_df):,} tracks with features: {self.feature_cols}")
        return self

    def recommend(
        self,
        seed_track_idx: int,
        n: int = 10,
        exclude_idxs: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Return top-n most similar tracks to seed_track_idx.
        exclude_idxs: track indices to exclude (e.g. already in playlist).
        """
        if self.tracks_df is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        query = self.tracks_df.iloc[seed_track_idx][self.feature_cols].values.reshape(1, -1)
        distances, indices = self.model.kneighbors(query)

        # Flatten and remove the seed track itself
        distances = distances.flatten()
        indices   = indices.flatten()
        mask = indices != seed_track_idx
        distances, indices = distances[mask], indices[mask]

        if exclude_idxs:
            exclude_set = set(exclude_idxs)
            keep = [i for i, idx in enumerate(indices) if idx not in exclude_set]
            distances = distances[keep]
            indices   = indices[keep]

        result = self.tracks_df.iloc[indices[:n]].copy()
        result["similarity"] = 1 - distances[:n]  # cosine: distance → similarity
        result = result.reset_index(drop=True)
        return result

    def batch_recommend(
        self,
        history_idxs: list[int],
        n: int = 20,
    ) -> pd.DataFrame:
        """
        Given a list of track indices (user history), recommend n songs
        by averaging vibe vectors and finding nearest neighbors.
        """
        history_vibe = self.tracks_df.iloc[history_idxs][self.feature_cols].values
        avg_vibe = history_vibe.mean(axis=0).reshape(1, -1)

        distances, indices = self.model.kneighbors(avg_vibe, n_neighbors=n + len(history_idxs))
        distances = distances.flatten()
        indices   = indices.flatten()

        exclude = set(history_idxs)
        keep    = [i for i, idx in enumerate(indices) if idx not in exclude]
        distances, indices = distances[keep], indices[keep]

        result = self.tracks_df.iloc[indices[:n]].copy()
        result["similarity"] = 1 - distances[:n]
        return result.reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved content-based model → {path}")

    @staticmethod
    def load(path: Path) -> "ContentBasedRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


def train(config_path: str = "config/config.yaml") -> ContentBasedRecommender:
    config   = load_config(config_path)
    feat_dir = Path(config["paths"]["features"])
    model_dir = Path(config["paths"]["models"])

    vibe_path = feat_dir / "vibe_matrix.parquet"
    if not vibe_path.exists():
        raise FileNotFoundError(f"{vibe_path} not found. Run audio_features.py first.")

    vibe_df = pd.read_parquet(vibe_path)

    # Merge normalized vibe features onto track metadata.
    # Drop raw audio columns from tracks_df first to avoid _x/_y suffix conflicts —
    # vibe_df has the properly normalized versions we want the KNN to use.
    proc_dir = Path(config["paths"]["processed_data"])
    tracks_df = pd.read_parquet(proc_dir / "tracks_train.parquet")
    meta_cols = [c for c in tracks_df.columns if c not in VIBE_FEATURES]
    merged = tracks_df[meta_cols].merge(
        vibe_df[["track_idx"] + [c for c in VIBE_FEATURES if c in vibe_df.columns]],
        on="track_idx",
        how="left",
    )

    cfg = config["models"]["content_based"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="content_based_knn"):
        mlflow.log_params(cfg)

        recommender = ContentBasedRecommender(
            n_neighbors=cfg["n_neighbors"],
            metric=cfg["metric"],
        )
        recommender.fit(merged)

        out_path = model_dir / "content_based.pkl"
        recommender.save(out_path)
        mlflow.log_artifact(str(out_path))

    return recommender


if __name__ == "__main__":
    train()
