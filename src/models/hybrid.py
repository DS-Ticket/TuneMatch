"""
Hybrid Re-Ranker — combines content-based and collaborative filtering
scores using LightGBM with LambdaRank objective (learns to rank, not
just classify).

Input features per candidate track:
  - content_score       (cosine similarity from K-NN)
  - cf_score            (ALS recommendation score)
  - audio features      (raw vibe features)
  - popularity          (Spotify popularity score)
  - genre_match         (binary: seed and candidate share genre)

Output: ranked list of tracks
"""

import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import GroupKFold


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


RANKER_FEATURES = [
    "content_score",
    "cf_score",
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo",
    "popularity",
    "genre_match",
    "energy_delta",
    "valence_delta",
    "tempo_delta",
]


class HybridRanker:
    def __init__(self, lgb_params: dict):
        self.lgb_params = lgb_params
        self.model: Optional[lgb.Booster] = None
        self.feature_cols: list[str] = []

    def _build_features(
        self,
        candidates_df: pd.DataFrame,
        seed_track: pd.Series,
    ) -> pd.DataFrame:
        """Build ranker feature matrix for a set of candidate tracks."""
        df = candidates_df.copy()

        # Content and CF scores should already be in candidates_df
        for col in ["content_score", "cf_score"]:
            if col not in df.columns:
                df[col] = 0.0

        # Delta features vs seed track
        for feat in ["energy", "valence", "tempo"]:
            if feat in df.columns and feat in seed_track.index:
                df[f"{feat}_delta"] = abs(df[feat] - seed_track[feat])

        # Genre match
        if "genre" in df.columns and "genre" in seed_track.index:
            df["genre_match"] = (df["genre"] == seed_track["genre"]).astype(int)
        else:
            df["genre_match"] = 0

        # Popularity (normalize 0-1)
        if "popularity" in df.columns:
            df["popularity"] = df["popularity"] / 100.0
        else:
            df["popularity"] = 0.5

        self.feature_cols = [c for c in RANKER_FEATURES if c in df.columns]
        return df

    def fit(
        self,
        train_df: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        val_df: Optional[pd.DataFrame] = None,
        val_labels: Optional[np.ndarray] = None,
        val_groups: Optional[np.ndarray] = None,
    ) -> "HybridRanker":
        """
        train_df:  rows are (query, candidate) pairs
        labels:    relevance labels (0=not relevant, 1=relevant, 2=highly relevant)
        groups:    query group sizes for LambdaRank
        """
        self.feature_cols = [c for c in RANKER_FEATURES if c in train_df.columns]
        X_train = train_df[self.feature_cols].values

        train_data = lgb.Dataset(X_train, label=labels, group=groups)

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(50)]

        if val_df is not None and val_labels is not None:
            X_val = val_df[self.feature_cols].values
            val_data = lgb.Dataset(X_val, label=val_labels, group=val_groups)
            valid_sets = [val_data]
        else:
            valid_sets = []
            callbacks = [lgb.log_evaluation(50)]

        params = {
            "objective":     "lambdarank",
            "metric":        "ndcg",
            "ndcg_eval_at":  [5, 10, 20],
            "learning_rate": self.lgb_params.get("learning_rate", 0.05),
            "num_leaves":    self.lgb_params.get("num_leaves", 63),
            "n_jobs":        -1,
            "verbose":       -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.lgb_params.get("n_estimators", 500),
            valid_sets=valid_sets if valid_sets else None,
            callbacks=callbacks,
        )

        logger.info("Hybrid ranker trained")
        return self

    def rank(
        self,
        candidates_df: pd.DataFrame,
        seed_track: pd.Series,
        n: int = 20,
    ) -> pd.DataFrame:
        """Re-rank candidate tracks for a given seed track."""
        if self.model is None:
            raise RuntimeError("Ranker not fitted")

        df = self._build_features(candidates_df, seed_track)
        X  = df[self.feature_cols].fillna(0).values
        scores = self.model.predict(X)
        df["hybrid_score"] = scores
        return df.sort_values("hybrid_score", ascending=False).head(n).reset_index(drop=True)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Ranker not fitted")
        return pd.Series(
            self.model.feature_importance(importance_type="gain"),
            index=self.feature_cols,
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved hybrid ranker → {path}")

    @staticmethod
    def load(path: Path) -> "HybridRanker":
        with open(path, "rb") as f:
            return pickle.load(f)


def train(config_path: str = "config/config.yaml") -> None:
    config    = load_config(config_path)
    model_dir = Path(config["paths"]["models"])
    cfg       = config["models"]["hybrid_ranker"]

    logger.info("Hybrid ranker training requires assembled (query, candidate, label) data.")
    logger.info("This is built in notebooks/05_hybrid_model.ipynb after content + CF models are ready.")

    ranker = HybridRanker(lgb_params=cfg)
    out = model_dir / "hybrid_ranker.pkl"
    ranker.save(out)
    logger.info(f"Saved untrained ranker scaffold → {out}")


if __name__ == "__main__":
    train()
