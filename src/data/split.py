"""
Temporal train/val/test split.

Train:  2014–2020  (model learns audio trends of this era)
Val:    2021       (hyperparameter tuning)
Test:   2022       (held-out evaluation — "how does it perform on current data?")

Writes three Parquet files to data/processed/:
  tracks_train.parquet
  tracks_val.parquet
  tracks_test.parquet
"""

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _infer_year(df: pd.DataFrame) -> pd.Series:
    """Try to extract release year from common column names."""
    for col in ["release_date", "year", "release_year", "album_release_date"]:
        if col in df.columns:
            try:
                return pd.to_datetime(df[col], errors="coerce").dt.year
            except Exception:
                try:
                    return df[col].astype(str).str[:4].astype(float)
                except Exception:
                    continue
    logger.warning("No year column found — assigning all tracks to train split")
    return pd.Series([2018] * len(df), index=df.index)


def split(config_path: str = "config/config.yaml") -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_data"])
    split_cfg = config["split"]

    tracks_path = processed_dir / "tracks.parquet"
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"{tracks_path} not found. Run `python -m src.data.preprocess` first."
        )

    df = pd.read_parquet(tracks_path)
    df["year"] = _infer_year(df)

    train_years = set(split_cfg["train_years"])
    val_years   = set(split_cfg["val_years"])
    test_years  = set(split_cfg["test_years"])

    train = df[df["year"].isin(train_years)].copy()
    val   = df[df["year"].isin(val_years)].copy()
    test  = df[df["year"].isin(test_years)].copy()
    # Tracks with no year → training set
    no_year = df[~df["year"].isin(train_years | val_years | test_years)].copy()
    train = pd.concat([train, no_year], ignore_index=True)

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = processed_dir / f"tracks_{name}.parquet"
        split_df.to_parquet(out, index=False)
        logger.info(f"{name:5s}: {len(split_df):>8,} tracks → {out}")

    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    split()
