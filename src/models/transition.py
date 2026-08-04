"""
Transition Playlist Generator — orders a set of songs so each one
blends smoothly into the next using harmonic mixing + BPM matching.

Algorithm: Beam Search over the transition graph built in harmonic.py
  1. Start from a seed track
  2. At each step, expand the top-B partial paths
  3. Score each extension by transition_score
  4. Return the highest-scoring complete path

This is the "DJ mode" feature — every song ending blends into the next.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.features.harmonic import build_transition_graph, transition_score


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TransitionPlaylistBuilder:
    def __init__(
        self,
        bpm_tolerance: float = 5.0,
        energy_arc: str = "gradual",
        beam_width: int = 5,
    ):
        self.bpm_tolerance = bpm_tolerance
        self.energy_arc    = energy_arc
        self.beam_width    = beam_width

    def build(
        self,
        tracks: pd.DataFrame,
        seed_track_idx: int,
        n_songs: int = 15,
    ) -> pd.DataFrame:
        """
        Build a transition playlist starting from seed_track_idx.

        Args:
            tracks:          DataFrame with audio features for candidate pool
            seed_track_idx:  Index of the starting track in `tracks`
            n_songs:         Target playlist length (including seed)

        Returns:
            Ordered DataFrame of tracks with transition scores
        """
        if len(tracks) < n_songs:
            logger.warning(f"Pool has only {len(tracks)} tracks, requested {n_songs}")
            n_songs = len(tracks)

        logger.info(f"Building transition graph for {len(tracks)} tracks...")
        graph = build_transition_graph(tracks, self.bpm_tolerance)

        # Beam search
        # Each beam entry: (path: list[int], cumulative_score: float)
        beam = [([seed_track_idx], 0.0)]
        completed = []

        for step in range(n_songs - 1):
            candidates = []
            for path, score in beam:
                last_idx  = path[-1]
                visited   = set(path)
                neighbors = graph.get(last_idx, [])

                if not neighbors:
                    # No harmonically compatible next track — fall back to energy similarity
                    neighbors = self._fallback_neighbors(tracks, last_idx, visited)

                for neighbor_idx, edge_score in neighbors:
                    if neighbor_idx in visited:
                        continue
                    # Apply energy arc modifier
                    arc_bonus = self._energy_arc_bonus(
                        tracks, path, neighbor_idx
                    )
                    new_score = score + edge_score + arc_bonus
                    candidates.append((path + [neighbor_idx], new_score))

            if not candidates:
                logger.warning(f"No candidates at step {step+1}, stopping early")
                break

            # Keep top beam_width paths
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_width]

        # Take the best path
        best_path, best_score = max(beam, key=lambda x: x[1])
        logger.info(f"Best path score: {best_score:.3f} over {len(best_path)} songs")

        result = tracks.iloc[best_path].copy()
        result["order"] = range(len(best_path))

        # Compute transition score for adjacent pairs
        t_scores = [None]  # seed has no incoming transition
        for i in range(1, len(best_path)):
            a = tracks.iloc[best_path[i - 1]]
            b = tracks.iloc[best_path[i]]
            t_scores.append(transition_score(a, b, self.bpm_tolerance))
        result["transition_score"] = t_scores

        return result.reset_index(drop=True)

    def _fallback_neighbors(
        self,
        tracks: pd.DataFrame,
        current_idx: int,
        visited: set,
    ) -> list[tuple[int, float]]:
        """When no harmonically compatible track exists, use energy proximity."""
        current = tracks.iloc[current_idx]
        neighbors = []
        for i in range(len(tracks)):
            if i in visited:
                continue
            candidate = tracks.iloc[i]
            score = transition_score(current, candidate, self.bpm_tolerance * 2)
            neighbors.append((i, score * 0.5))  # penalize fallback
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:10]

    def _energy_arc_bonus(
        self,
        tracks: pd.DataFrame,
        path: list[int],
        next_idx: int,
    ) -> float:
        """
        Reward paths that follow the desired energy arc.
        gradual:  energy should increase slowly over the playlist
        dramatic: energy should peak in the middle
        flat:     energy should stay consistent
        """
        if "energy" not in tracks.columns or len(path) < 2:
            return 0.0

        energies = [tracks.iloc[i]["energy"] for i in path]
        next_energy = tracks.iloc[next_idx]["energy"]
        avg_energy  = np.mean(energies)

        if self.energy_arc == "gradual":
            # Reward upward energy trajectory
            return 0.05 if next_energy > avg_energy else -0.02

        elif self.energy_arc == "dramatic":
            # Peak in middle: reward increase in first half, decrease in second
            progress = len(path) / 15  # normalize to expected length
            if progress < 0.5:
                return 0.05 if next_energy > avg_energy else -0.02
            else:
                return 0.05 if next_energy < avg_energy else -0.02

        return 0.0  # flat arc


def build_transition_playlist(
    tracks_df: pd.DataFrame,
    seed_track_id: str,
    n_songs: int = 15,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Public API for building a transition playlist.

    Args:
        tracks_df:      DataFrame of candidate tracks with audio features
        seed_track_id:  The track_id to start from
        n_songs:        Playlist length

    Returns:
        Ordered playlist DataFrame
    """
    config = load_config(config_path)
    trans_cfg = config["transition"]

    builder = TransitionPlaylistBuilder(
        bpm_tolerance=trans_cfg["bpm_tolerance"],
        energy_arc=trans_cfg["energy_arc"],
        beam_width=trans_cfg["beam_width"],
    )

    # Find seed index
    if "track_id" in tracks_df.columns:
        seed_mask = tracks_df["track_id"] == seed_track_id
        if not seed_mask.any():
            raise ValueError(f"Seed track {seed_track_id} not in candidate pool")
        seed_idx = tracks_df.index[seed_mask][0]
    else:
        seed_idx = int(seed_track_id)

    return builder.build(tracks_df.reset_index(drop=True), seed_idx, n_songs)
