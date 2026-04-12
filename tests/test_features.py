"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.audio_features import normalize_vibe_vector, build_full_feature_matrix
from src.features.harmonic import (
    get_camelot, camelot_compatible, bpm_compatible, transition_score
)


# ── Audio features ────────────────────────────────────────────────────────────

def make_tracks(n=10):
    np.random.seed(42)
    return pd.DataFrame({
        "track_id":        [f"track_{i}" for i in range(n)],
        "track_idx":       list(range(n)),
        "danceability":    np.random.uniform(0, 1, n),
        "energy":          np.random.uniform(0, 1, n),
        "valence":         np.random.uniform(0, 1, n),
        "acousticness":    np.random.uniform(0, 1, n),
        "instrumentalness": np.random.uniform(0, 1, n),
        "tempo":           np.random.uniform(80, 180, n),
        "loudness":        np.random.uniform(-20, 0, n),
        "speechiness":     np.random.uniform(0, 0.3, n),
        "liveness":        np.random.uniform(0, 0.5, n),
        "duration_ms":     np.random.randint(150000, 300000, n),
        "key":             np.random.randint(0, 12, n),
        "mode":            np.random.randint(0, 2, n),
        "time_signature":  np.random.choice([3, 4], n),
    })


def test_vibe_vector_shape():
    df = make_tracks(20)
    vibe = normalize_vibe_vector(df)
    assert len(vibe) == 20
    assert "danceability" in vibe.columns
    assert "track_id" in vibe.columns


def test_vibe_vector_range():
    df = make_tracks(50)
    vibe = normalize_vibe_vector(df)
    for col in ["danceability", "energy", "valence", "acousticness", "instrumentalness"]:
        assert vibe[col].between(0, 1).all(), f"{col} out of [0,1] range"


def test_full_feature_matrix():
    df = make_tracks(10)
    feat = build_full_feature_matrix(df)
    assert len(feat) == 10
    assert "track_id" in feat.columns
    # One-hot encoded key should expand columns
    assert len(feat.columns) > 10


# ── Harmonic mixing ───────────────────────────────────────────────────────────

def test_camelot_known_keys():
    assert get_camelot(0, 1) == "8B"   # C major
    assert get_camelot(9, 0) == "8A"   # A minor (relative of C major)
    assert get_camelot(7, 1) == "9B"   # G major


def test_camelot_compatible_same():
    assert camelot_compatible("8B", "8B")


def test_camelot_compatible_adjacent():
    assert camelot_compatible("8B", "9B")
    assert camelot_compatible("8B", "7B")


def test_camelot_compatible_relative():
    # C major (8B) and A minor (8A) are relative — compatible
    assert camelot_compatible("8B", "8A")


def test_camelot_incompatible():
    assert not camelot_compatible("1A", "6B")


def test_bpm_compatible():
    assert bpm_compatible(120, 122)      # within tolerance
    assert bpm_compatible(120, 240)      # harmonic double
    assert not bpm_compatible(120, 145)  # too far


def test_transition_score_range():
    a = {"key": 0, "mode": 1, "tempo": 120, "energy": 0.7, "valence": 0.6}
    b = {"key": 7, "mode": 1, "tempo": 118, "energy": 0.65, "valence": 0.55}
    score = transition_score(a, b)
    assert 0.0 <= score <= 1.0


def test_transition_score_identical():
    a = {"key": 0, "mode": 1, "tempo": 120, "energy": 0.7, "valence": 0.6}
    score = transition_score(a, a)
    assert score >= 0.7  # should be high for identical tracks
