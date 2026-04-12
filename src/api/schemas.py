"""Pydantic schemas for the TuneMatch FastAPI."""

from typing import Optional
from pydantic import BaseModel, Field


class TrackInfo(BaseModel):
    track_id:         str
    track_name:       Optional[str] = None
    artist:           Optional[str] = None
    genre:            Optional[str] = None
    popularity:       Optional[float] = None
    danceability:     Optional[float] = None
    energy:           Optional[float] = None
    valence:          Optional[float] = None
    tempo:            Optional[float] = None
    similarity:       Optional[float] = None
    hybrid_score:     Optional[float] = None
    transition_score: Optional[float] = None


class RecommendRequest(BaseModel):
    seed_track_id: str = Field(..., description="Track ID of the seed song")
    n: int = Field(20, ge=1, le=50, description="Number of recommendations")
    mode: str = Field(
        "hybrid",
        description="Recommendation mode: 'content', 'collaborative', 'hybrid', 'autoplay', 'transition'"
    )
    user_id: Optional[str] = Field(None, description="User ID for collaborative/autoplay modes")
    history: Optional[list[str]] = Field(None, description="Recent track IDs for autoplay mode")


class RecommendResponse(BaseModel):
    seed_track_id: str
    mode:          str
    tracks:        list[TrackInfo]
    model_version: str = "0.1.0"


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
