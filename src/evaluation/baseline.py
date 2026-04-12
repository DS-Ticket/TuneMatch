"""
Genre-only baseline recommender — what Apple Music essentially does.

Recommend the most popular songs within the same genre as the seed track.
This is what we're beating. If our model doesn't outperform this, it's not useful.
"""

import numpy as np
import pandas as pd
from loguru import logger


class GenreBaseline:
    """
    Recommend top-N most popular songs in the same genre.
    Falls back to overall popularity if genre match is empty.
    """

    def __init__(self):
        self.tracks_df = None

    def fit(self, tracks_df: pd.DataFrame) -> "GenreBaseline":
        self.tracks_df = tracks_df.copy()
        if "popularity" not in self.tracks_df.columns:
            self.tracks_df["popularity"] = 50
        if "genre" not in self.tracks_df.columns:
            self.tracks_df["genre"] = "unknown"
        logger.info(f"Baseline fitted on {len(tracks_df):,} tracks")
        return self

    def recommend(self, seed_track_idx: int, n: int = 20) -> pd.DataFrame:
        seed = self.tracks_df.iloc[seed_track_idx]
        genre = seed.get("genre", "unknown")

        same_genre = self.tracks_df[
            (self.tracks_df["genre"] == genre) &
            (self.tracks_df.index != seed_track_idx)
        ]

        if len(same_genre) < n:
            # Pad with overall popularity
            other = self.tracks_df[
                (self.tracks_df["genre"] != genre) &
                (self.tracks_df.index != seed_track_idx)
            ]
            same_genre = pd.concat([same_genre, other])

        result = (
            same_genre
            .sort_values("popularity", ascending=False)
            .head(n)
            .copy()
            .reset_index(drop=True)
        )
        result["baseline_score"] = result["popularity"] / 100.0
        return result
