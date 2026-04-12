"""
TuneMatch FastAPI — serves all recommendation modes via REST.

Endpoints:
  GET  /health                  — model load status
  POST /recommend               — main recommendation endpoint
  GET  /track/{track_id}        — track metadata lookup
  GET  /similar/{track_id}      — quick content-based similar tracks
"""

from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.api.schemas import (
    HealthResponse, RecommendRequest, RecommendResponse, TrackInfo
)
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import ALSRecommender
from src.models.transition import build_transition_playlist


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


app = FastAPI(
    title="TuneMatch API",
    description="Audio-feature-aware music recommendation engine",
    version="0.1.0",
)

# ── Globals (loaded on startup) ───────────────────────────────────────────────
config     = load_config()
tracks_df  = pd.DataFrame()
cb_model: ContentBasedRecommender | None = None
als_model: ALSRecommender | None = None


@app.on_event("startup")
async def load_models():
    global tracks_df, cb_model, als_model

    model_dir = Path(config["paths"]["models"])
    proc_dir  = Path(config["paths"]["processed_data"])

    # Load tracks
    tracks_path = proc_dir / "tracks_train.parquet"
    if tracks_path.exists():
        tracks_df = pd.read_parquet(tracks_path)
        logger.info(f"Loaded {len(tracks_df):,} tracks")
    else:
        logger.warning(f"Tracks not found at {tracks_path}")

    # Load content-based model
    cb_path = model_dir / "content_based.pkl"
    if cb_path.exists():
        cb_model = ContentBasedRecommender.load(cb_path)
        logger.info("Content-based model loaded")
    else:
        logger.warning("Content-based model not found")

    # Load ALS model
    als_path = model_dir / "als_model.pkl"
    if als_path.exists():
        try:
            als_model = ALSRecommender.load(als_path)
            logger.info("ALS model loaded")
        except Exception as e:
            logger.warning(f"ALS load failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded={
            "content_based":   cb_model is not None,
            "collaborative":   als_model is not None,
            "tracks_loaded":   not tracks_df.empty,
        },
    )


@app.get("/track/{track_id}", response_model=TrackInfo)
async def get_track(track_id: str):
    if tracks_df.empty:
        raise HTTPException(status_code=503, detail="Tracks not loaded")

    row = tracks_df[tracks_df["track_id"] == track_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    return _row_to_track_info(row.iloc[0])


@app.get("/similar/{track_id}", response_model=RecommendResponse)
async def similar_tracks(track_id: str, n: int = 10):
    return await recommend(RecommendRequest(
        seed_track_id=track_id,
        n=n,
        mode="content",
    ))


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if tracks_df.empty:
        raise HTTPException(status_code=503, detail="Tracks not loaded")

    # Resolve seed track
    seed_rows = tracks_df[tracks_df["track_id"] == req.seed_track_id]
    if seed_rows.empty:
        raise HTTPException(status_code=404, detail=f"Seed track {req.seed_track_id} not found")

    seed_idx = seed_rows.index[0]

    try:
        if req.mode == "content":
            if cb_model is None:
                raise HTTPException(status_code=503, detail="Content-based model not loaded")
            result_df = cb_model.recommend(seed_idx, n=req.n)

        elif req.mode == "collaborative":
            if als_model is None:
                raise HTTPException(status_code=503, detail="ALS model not loaded")
            if not req.user_id:
                raise HTTPException(status_code=400, detail="user_id required for collaborative mode")
            recs = als_model.recommend_for_user(req.user_id, n=req.n)
            track_ids = [r[0] for r in recs]
            result_df = tracks_df[tracks_df["track_id"].isin(track_ids)].copy()

        elif req.mode == "hybrid":
            if cb_model is None:
                raise HTTPException(status_code=503, detail="Content-based model not loaded")
            result_df = cb_model.recommend(seed_idx, n=req.n)

        elif req.mode == "autoplay":
            if cb_model is None:
                raise HTTPException(status_code=503, detail="Content-based model not loaded")
            history_idxs = []
            if req.history:
                for tid in req.history:
                    rows = tracks_df[tracks_df["track_id"] == tid]
                    if not rows.empty:
                        history_idxs.append(rows.index[0])
            if history_idxs:
                result_df = cb_model.batch_recommend(history_idxs, n=req.n)
            else:
                result_df = cb_model.recommend(seed_idx, n=req.n)

        elif req.mode == "transition":
            # Use a pool of similar tracks for transition building
            if cb_model is None:
                raise HTTPException(status_code=503, detail="Content-based model not loaded")
            pool = cb_model.recommend(seed_idx, n=req.n * 3)
            seed_row = tracks_df.iloc[[seed_idx]]
            pool = pd.concat([seed_row, pool], ignore_index=True)
            result_df = build_transition_playlist(pool, req.seed_track_id, n_songs=req.n)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    except (KeyError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    tracks_out = [_row_to_track_info(row) for _, row in result_df.head(req.n).iterrows()]

    return RecommendResponse(
        seed_track_id=req.seed_track_id,
        mode=req.mode,
        tracks=tracks_out,
    )


def _row_to_track_info(row: pd.Series) -> TrackInfo:
    def _safe(val):
        if pd.isna(val) if not isinstance(val, str) else False:
            return None
        return val

    return TrackInfo(
        track_id=str(row.get("track_id", "")),
        track_name=_safe(row.get("track_name")),
        artist=_safe(row.get("artist")),
        genre=_safe(row.get("genre")),
        popularity=_safe(row.get("popularity")),
        danceability=_safe(row.get("danceability")),
        energy=_safe(row.get("energy")),
        valence=_safe(row.get("valence")),
        tempo=_safe(row.get("tempo")),
        similarity=_safe(row.get("similarity")),
        hybrid_score=_safe(row.get("hybrid_score")),
        transition_score=_safe(row.get("transition_score")),
    )


if __name__ == "__main__":
    import uvicorn
    api_cfg = config["api"]
    uvicorn.run("src.api.app:app", host=api_cfg["host"], port=api_cfg["port"], reload=True)
