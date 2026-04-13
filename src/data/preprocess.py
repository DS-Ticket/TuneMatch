"""
Preprocessing pipeline — cleans raw CSVs, merges datasets, and writes
Parquet files to data/processed/.

Steps:
  1. Load each raw dataset
  2. Standardize column names
  3. Drop duplicates (by track_id or track_name+artist)
  4. Handle nulls in audio features
  5. Merge into a unified tracks table
  6. Build user interactions table from Last.fm + MSD data
  7. Write Parquet to data/processed/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms", "key", "mode", "time_signature",
]

RENAME_MAPS = {
    # spotify_tracks dataset (maharshipandya)
    "track_id":        "track_id",
    "track_name":      "track_name",
    "artists":         "artist",
    "album_name":      "album",
    "track_genre":     "genre",
    "popularity":      "popularity",
    # spotify_12m_songs dataset (rodolfofigueroa) — different column names
    "id":              "track_id",
    "name":            "track_name",
    # 30000_spotify_songs dataset
    "track.name":      "track_name",
    "artist.name":     "artist",
    "playlist_genre":  "genre",
    "playlist_subgenre": "subgenre",
}


def _clean_artist_column(series: pd.Series) -> pd.Series:
    """Convert "['Artist A', 'Artist B']" → "Artist A, Artist B"."""
    import ast

    def _parse(val):
        if not isinstance(val, str):
            return val
        val = val.strip()
        if val.startswith("["):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return ", ".join(str(v) for v in parsed)
            except Exception:
                pass
            return val.strip("[]").replace("'", "").replace('"', "").strip()
        return val

    return series.apply(_parse)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower().strip() for c in df.columns]
    df.rename(columns=RENAME_MAPS, inplace=True)
    if "artist" in df.columns:
        df["artist"] = _clean_artist_column(df["artist"])
    return df


def load_spotify_tracks(raw_dir: Path) -> pd.DataFrame:
    """maharshipandya/-spotify-tracks-dataset"""
    candidates = list(raw_dir.rglob("dataset.csv"))
    if not candidates:
        candidates = list(raw_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under {raw_dir}")

    path = candidates[0]
    logger.info(f"Loading Spotify Tracks from {path}")
    df = pd.read_csv(path, low_memory=False)
    df = _standardize_columns(df)

    # Ensure required columns exist
    missing = [c for c in AUDIO_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"Missing audio features: {missing}")

    return df


def load_million_song_lastfm(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """undefinenull/million-song-dataset-spotify-lastfm
    Returns: (spotify_df, lastfm_interactions_df)
    """
    spotify_files = list(raw_dir.rglob("spotify_data.csv"))
    lastfm_files  = list(raw_dir.rglob("lastfm_data.csv"))

    spotify_df = pd.DataFrame()
    interactions_df = pd.DataFrame()

    if spotify_files:
        logger.info(f"Loading MSD-Spotify from {spotify_files[0]}")
        spotify_df = pd.read_csv(spotify_files[0], low_memory=False)
        spotify_df = _standardize_columns(spotify_df)

    if lastfm_files:
        logger.info(f"Loading Last.fm interactions from {lastfm_files[0]}")
        interactions_df = pd.read_csv(lastfm_files[0], low_memory=False)
        interactions_df = _standardize_columns(interactions_df)

    return spotify_df, interactions_df


def load_spotify_1m(raw_dir: Path) -> pd.DataFrame:
    """rodolfofigueroa/spotify-12m-songs"""
    candidates = list(raw_dir.rglob("tracks_features.csv"))
    if not candidates:
        candidates = list(raw_dir.rglob("*.csv"))
    if not candidates:
        logger.warning("Spotify 1M dataset not found — skipping")
        return pd.DataFrame()

    path = candidates[0]
    logger.info(f"Loading Spotify 1M from {path}")
    df = pd.read_csv(path, low_memory=False)
    return _standardize_columns(df)


def deduplicate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"])
    elif {"track_name", "artist"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["track_name", "artist"])
    logger.info(f"Deduplication: {before:,} → {len(df):,} tracks")
    return df


def clean_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in AUDIO_FEATURES if c in df.columns]

    # Drop rows where ALL audio features are null
    df = df.dropna(subset=present, how="all")

    # Fill remaining nulls with median per feature
    for col in present:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.debug(f"Filled {col} nulls with median={median:.4f}")

    # Clamp 0-1 features
    for col in ["danceability", "energy", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence"]:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    # Tempo sanity: remove tracks with tempo < 40 or > 250 BPM
    if "tempo" in df.columns:
        before = len(df)
        df = df[df["tempo"].between(40, 250) | df["tempo"].isna()]
        removed = before - len(df)
        if removed:
            logger.warning(f"Removed {removed} tracks with invalid tempo")

    return df


def build_tracks_table(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple track DataFrames, keeping richer audio feature rows."""
    combined = pd.concat(dfs, ignore_index=True)
    combined = deduplicate_tracks(combined)
    combined = clean_audio_features(combined)

    # Assign integer track index for matrix operations
    combined = combined.reset_index(drop=True)
    combined["track_idx"] = combined.index

    logger.info(f"Final tracks table: {len(combined):,} tracks")
    return combined


def build_interactions_table(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Last.fm / user interaction data."""
    if interactions_df.empty:
        logger.warning("No interaction data — will use implicit play-count simulation")
        return pd.DataFrame(columns=["user_id", "track_id", "play_count", "timestamp"])

    col_map = {}
    for col in interactions_df.columns:
        if "user" in col:
            col_map[col] = "user_id"
        elif "track" in col or "song" in col:
            col_map[col] = "track_id"
        elif "count" in col or "play" in col:
            col_map[col] = "play_count"
        elif "date" in col or "time" in col:
            col_map[col] = "timestamp"

    interactions_df = interactions_df.rename(columns=col_map)

    # Aggregate multiple listens into play counts
    group_cols = [c for c in ["user_id", "track_id"] if c in interactions_df.columns]
    if group_cols:
        if "play_count" not in interactions_df.columns:
            interactions_df["play_count"] = 1
        interactions_df = (
            interactions_df.groupby(group_cols)["play_count"]
            .sum()
            .reset_index()
        )

    logger.info(f"Interactions table: {len(interactions_df):,} records")
    return interactions_df


def run(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    raw_dir       = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    track_dfs = []

    # ── Load each dataset ──────────────────────────────────────────────────────
    try:
        df1 = load_spotify_tracks(raw_dir / "spotify_tracks_dataset")
        track_dfs.append(df1)
    except FileNotFoundError as e:
        logger.warning(str(e))

    try:
        df_msd, df_interactions = load_million_song_lastfm(
            raw_dir / "million_song_dataset_spotify_lastfm"
        )
        if not df_msd.empty:
            track_dfs.append(df_msd)
    except Exception as e:
        logger.warning(f"MSD/Last.fm load failed: {e}")
        df_interactions = pd.DataFrame()

    try:
        df_1m = load_spotify_1m(raw_dir / "spotify_12m_songs")
        if not df_1m.empty:
            track_dfs.append(df_1m)
    except Exception as e:
        logger.warning(f"Spotify 1M load failed: {e}")

    if not track_dfs:
        logger.error("No datasets loaded. Run `python -m src.data.ingest` first.")
        return

    # ── Build tables ───────────────────────────────────────────────────────────
    tracks = build_tracks_table(track_dfs)
    interactions = build_interactions_table(df_interactions)

    # ── Write Parquet ──────────────────────────────────────────────────────────
    tracks_path = processed_dir / "tracks.parquet"
    interactions_path = processed_dir / "interactions.parquet"

    tracks.to_parquet(tracks_path, index=False)
    interactions.to_parquet(interactions_path, index=False)

    logger.success(f"Wrote {len(tracks):,} tracks → {tracks_path}")
    logger.success(f"Wrote {len(interactions):,} interactions → {interactions_path}")


if __name__ == "__main__":
    run()
