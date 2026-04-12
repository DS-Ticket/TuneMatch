"""Tests for recommendation models."""

import numpy as np
import pandas as pd
import pytest

from src.models.content_based import ContentBasedRecommender
from src.evaluation.baseline import GenreBaseline
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, diversity, novelty


def make_tracks(n=100):
    np.random.seed(42)
    genres = ["pop", "rock", "hip-hop", "electronic", "jazz"]
    return pd.DataFrame({
        "track_id":        [f"track_{i}" for i in range(n)],
        "track_idx":       list(range(n)),
        "track_name":      [f"Song {i}" for i in range(n)],
        "artist":          [f"Artist {i % 20}" for i in range(n)],
        "genre":           [genres[i % len(genres)] for i in range(n)],
        "popularity":      np.random.randint(0, 100, n),
        "danceability":    np.random.uniform(0, 1, n),
        "energy":          np.random.uniform(0, 1, n),
        "valence":         np.random.uniform(0, 1, n),
        "acousticness":    np.random.uniform(0, 1, n),
        "instrumentalness": np.random.uniform(0, 1, n),
        "tempo":           np.random.uniform(80, 180, n),
    })


# ── Content-Based ─────────────────────────────────────────────────────────────

class TestContentBasedRecommender:
    def test_fit_and_recommend(self):
        tracks = make_tracks(100)
        model = ContentBasedRecommender(n_neighbors=10)
        model.fit(tracks)
        recs = model.recommend(seed_track_idx=0, n=10)
        assert len(recs) == 10
        assert 0 not in recs["track_idx"].values  # seed not in recommendations

    def test_similarity_scores(self):
        tracks = make_tracks(100)
        model = ContentBasedRecommender(n_neighbors=10)
        model.fit(tracks)
        recs = model.recommend(0, n=5)
        assert "similarity" in recs.columns
        assert (recs["similarity"] >= 0).all()
        assert (recs["similarity"] <= 1).all()

    def test_batch_recommend(self):
        tracks = make_tracks(100)
        model = ContentBasedRecommender(n_neighbors=20)
        model.fit(tracks)
        recs = model.batch_recommend([0, 1, 2], n=10)
        assert len(recs) == 10
        # Seed tracks should not be in output
        assert not any(idx in recs["track_idx"].values for idx in [0, 1, 2])

    def test_exclude_already_played(self):
        tracks = make_tracks(100)
        model = ContentBasedRecommender(n_neighbors=30)
        model.fit(tracks)
        exclude = [5, 10, 15]
        recs = model.recommend(0, n=10, exclude_idxs=exclude)
        rec_idxs = recs["track_idx"].tolist() if "track_idx" in recs.columns else []
        for exc in exclude:
            assert exc not in rec_idxs


# ── Baseline ──────────────────────────────────────────────────────────────────

class TestGenreBaseline:
    def test_recommend_same_genre(self):
        tracks = make_tracks(100)
        baseline = GenreBaseline().fit(tracks)
        recs = baseline.recommend(seed_track_idx=0, n=10)
        assert len(recs) == 10

    def test_sorted_by_popularity(self):
        tracks = make_tracks(100)
        baseline = GenreBaseline().fit(tracks)
        recs = baseline.recommend(0, n=10)
        pops = recs["popularity"].tolist()
        assert pops == sorted(pops, reverse=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_precision_perfect(self):
        recs = ["a", "b", "c", "d", "e"]
        rel  = {"a", "b", "c", "d", "e"}
        assert precision_at_k(recs, rel, 5) == 1.0

    def test_precision_none(self):
        recs = ["a", "b", "c"]
        rel  = {"x", "y", "z"}
        assert precision_at_k(recs, rel, 3) == 0.0

    def test_recall_full(self):
        recs = ["a", "b", "c"]
        rel  = {"a", "b"}
        assert recall_at_k(recs, rel, 3) == 1.0

    def test_ndcg_perfect(self):
        recs = ["a", "b", "c"]
        rel  = {"a", "b", "c"}
        assert ndcg_at_k(recs, rel, 3) == pytest.approx(1.0)

    def test_ndcg_empty(self):
        assert ndcg_at_k([], set(), 5) == 0.0

    def test_diversity(self):
        tracks = make_tracks(10)
        d = diversity(tracks, ["danceability", "energy", "valence"])
        assert 0.0 <= d <= 1.0

    def test_novelty(self):
        tracks = make_tracks(10)
        n = novelty(tracks)
        assert n >= 0.0
