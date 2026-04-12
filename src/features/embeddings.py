"""
Song embeddings — learned low-dimensional representations used by
the sequence model and hybrid ranker.

Two approaches:
  1. Audio feature PCA embedding (fast, no training needed)
  2. Learned ALS item embeddings from collaborative filtering (richer)

Both are saved to data/features/embeddings.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pca_embeddings(
    feature_matrix: pd.DataFrame,
    n_components: int = 32,
) -> np.ndarray:
    """
    Reduce full audio feature matrix to n_components via PCA.
    Retains ~95% of variance with 32 components typically.
    """
    feature_cols = [
        c for c in feature_matrix.columns
        if c not in ("track_id", "track_idx")
    ]
    X = feature_matrix[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    embeddings = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA: {X_scaled.shape[1]} features → {n_components} dims "
                f"({explained:.1%} variance explained)")

    return embeddings


def als_embeddings(model_path: Path) -> np.ndarray | None:
    """Load item factor matrix from a trained ALS model (implicit library)."""
    try:
        import implicit
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model.item_factors
    except Exception as e:
        logger.warning(f"Could not load ALS embeddings: {e}")
        return None


def run(config_path: str = "config/config.yaml") -> None:
    config   = load_config(config_path)
    feat_dir = Path(config["paths"]["features"])

    full_matrix_path = feat_dir / "full_feature_matrix.parquet"
    if not full_matrix_path.exists():
        raise FileNotFoundError(
            f"{full_matrix_path} not found. Run audio_features.py first."
        )

    feature_matrix = pd.read_parquet(full_matrix_path)
    logger.info(f"Computing PCA embeddings for {len(feature_matrix):,} tracks")

    embeddings = pca_embeddings(feature_matrix, n_components=32)

    embed_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)
    embed_df["track_id"]  = feature_matrix["track_id"].values
    embed_df["track_idx"] = feature_matrix["track_idx"].values

    out = feat_dir / "embeddings.parquet"
    embed_df.to_parquet(out, index=False)
    logger.success(f"Wrote embeddings ({embeddings.shape}) → {out}")


if __name__ == "__main__":
    run()
