"""
Audio feature engineering.

Produces two artifacts:
  data/features/vibe_matrix.parquet   — normalized vibe vectors (used for similarity)
  data/features/full_feature_matrix.parquet — all audio features (used for ML models)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


VIBE_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "tempo",
]

ALL_CONTINUOUS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]

CATEGORICAL = ["key", "mode", "time_signature"]


def normalize_vibe_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale vibe features to [0, 1].
    Tempo is scaled relative to [40, 250] BPM range.
    """
    present = [c for c in VIBE_FEATURES if c in df.columns]
    vibe = df[present].copy()

    scaler = MinMaxScaler()
    vibe[present] = scaler.fit_transform(vibe[present])

    vibe["track_id"]  = df.get("track_id",  df.index).values
    vibe["track_idx"] = df.get("track_idx", df.index).values

    return vibe


def build_full_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines:
      - StandardScaler on continuous features
      - One-hot encoding on categorical features
    Returns a dense feature matrix ready for sklearn models.
    """
    continuous_cols = [c for c in ALL_CONTINUOUS if c in df.columns]
    cat_cols        = [c for c in CATEGORICAL if c in df.columns]

    # Scale continuous
    scaler = StandardScaler()
    cont_scaled = pd.DataFrame(
        scaler.fit_transform(df[continuous_cols]),
        columns=continuous_cols,
        index=df.index,
    )

    # One-hot categoricals
    present_cat = [c for c in cat_cols if c in df.columns]
    cat_encoded = pd.get_dummies(df[present_cat]) if present_cat else pd.DataFrame(index=df.index)

    # Combine
    feature_matrix = pd.concat([cont_scaled, cat_encoded], axis=1)
    feature_matrix["track_id"]  = df.get("track_id",  df.index).values
    feature_matrix["track_idx"] = df.get("track_idx", df.index).values

    return feature_matrix


def compute_song_similarity(
    vibe_matrix: pd.DataFrame,
    query_track_idx: int,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Cosine similarity between a query track and all others.
    Returns top_k most similar tracks.
    """
    feature_cols = [c for c in VIBE_FEATURES if c in vibe_matrix.columns]
    matrix = vibe_matrix[feature_cols].values
    query  = matrix[query_track_idx].reshape(1, -1)

    # Cosine similarity
    norms   = np.linalg.norm(matrix, axis=1, keepdims=True)
    q_norm  = np.linalg.norm(query)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (matrix @ query.T).flatten() / (norms.flatten() * q_norm + 1e-8)

    sims[query_track_idx] = -1  # exclude self

    top_indices = np.argsort(sims)[::-1][:top_k]
    result = vibe_matrix.iloc[top_indices][["track_id", "track_idx"]].copy()
    result["similarity"] = sims[top_indices]
    return result.reset_index(drop=True)


def run(config_path: str = "config/config.yaml") -> None:
    config   = load_config(config_path)
    proc_dir = Path(config["paths"]["processed_data"])
    feat_dir = Path(config["paths"]["features"])
    feat_dir.mkdir(parents=True, exist_ok=True)

    train_path = proc_dir / "tracks_train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run split.py first.")

    df = pd.read_parquet(train_path)
    logger.info(f"Building features for {len(df):,} training tracks")

    vibe   = normalize_vibe_vector(df)
    full   = build_full_feature_matrix(df)

    vibe.to_parquet(feat_dir / "vibe_matrix.parquet", index=False)
    full.to_parquet(feat_dir / "full_feature_matrix.parquet", index=False)

    logger.success(f"Wrote vibe matrix:  {feat_dir / 'vibe_matrix.parquet'}")
    logger.success(f"Wrote full matrix:  {feat_dir / 'full_feature_matrix.parquet'}")


if __name__ == "__main__":
    run()
