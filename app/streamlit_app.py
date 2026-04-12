"""
TuneMatch — Streamlit Demo

Run: streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TuneMatch",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 TuneMatch")
st.caption("Audio-feature-aware music recommendation — built to beat Apple Music's autoplay")


# ── Load config + data ────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_tracks():
    config = load_config()
    proc_dir = Path(config["paths"]["processed_data"])
    path = proc_dir / "tracks_train.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_resource
def load_cb_model():
    config = load_config()
    model_path = Path(config["paths"]["models"]) / "content_based.pkl"
    if model_path.exists():
        from src.models.content_based import ContentBasedRecommender
        return ContentBasedRecommender.load(model_path)
    return None


config   = load_config()
tracks   = load_tracks()
cb_model = load_cb_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
mode = st.sidebar.radio(
    "Recommendation Mode",
    options=["content", "autoplay", "transition"],
    format_func=lambda x: {
        "content":    "🎯 Similar Vibe",
        "autoplay":   "▶️  Autoplay",
        "transition": "🎛️  Transition Playlist (DJ Mode)",
    }[x],
)
n_songs = st.sidebar.slider("Playlist Length", min_value=5, max_value=30, value=15)

# ── Main content ──────────────────────────────────────────────────────────────
if tracks.empty:
    st.warning("No track data found. Run the data pipeline first:")
    st.code("python -m src.data.ingest\npython -m src.data.preprocess\npython -m src.data.split")
    st.stop()

# Seed track selector
col1, col2 = st.columns([2, 1])

with col1:
    search_query = st.text_input("Search for a song", placeholder="Type track name or artist...")

    if search_query and len(search_query) >= 2:
        name_col   = "track_name" if "track_name" in tracks.columns else None
        artist_col = "artist"     if "artist"     in tracks.columns else None

        mask = pd.Series(False, index=tracks.index)
        if name_col:
            mask |= tracks[name_col].str.contains(search_query, case=False, na=False)
        if artist_col:
            mask |= tracks[artist_col].str.contains(search_query, case=False, na=False)

        matches = tracks[mask].head(20)

        if matches.empty:
            st.info("No tracks found. Try a different search.")
            seed_idx = None
        else:
            display_col = []
            if name_col:
                display_col.append(name_col)
            if artist_col:
                display_col.append(artist_col)
            if "genre" in tracks.columns:
                display_col.append("genre")

            selected_row = st.selectbox(
                "Select seed track",
                options=matches.index.tolist(),
                format_func=lambda i: " — ".join(
                    str(tracks.loc[i, c]) for c in display_col if c in tracks.columns
                ),
            )
            seed_idx = selected_row
    else:
        seed_idx = None

with col2:
    if seed_idx is not None and "energy" in tracks.columns:
        seed = tracks.loc[seed_idx]
        st.metric("Energy",      f"{seed.get('energy', 0):.2f}")
        st.metric("Danceability", f"{seed.get('danceability', 0):.2f}")
        st.metric("Valence",     f"{seed.get('valence', 0):.2f}")
        if "tempo" in tracks.columns:
            st.metric("Tempo", f"{seed.get('tempo', 0):.0f} BPM")


# ── Generate Playlist ─────────────────────────────────────────────────────────
if seed_idx is not None:
    if st.button("🎵 Generate Playlist", type="primary"):
        if cb_model is None:
            st.warning("Models not trained yet. Run the training pipeline first.")
            st.code("python -m src.models.content_based")
        else:
            with st.spinner("Building your playlist..."):
                if mode == "content":
                    result = cb_model.recommend(seed_idx, n=n_songs)

                elif mode == "autoplay":
                    result = cb_model.recommend(seed_idx, n=n_songs)

                elif mode == "transition":
                    from src.models.transition import TransitionPlaylistBuilder
                    pool = cb_model.recommend(seed_idx, n=n_songs * 3)
                    seed_row_df = tracks.iloc[[seed_idx]].copy()
                    pool = pd.concat([seed_row_df, pool], ignore_index=True)

                    trans_cfg = config["transition"]
                    builder = TransitionPlaylistBuilder(
                        bpm_tolerance=trans_cfg["bpm_tolerance"],
                        energy_arc=trans_cfg["energy_arc"],
                        beam_width=trans_cfg["beam_width"],
                    )
                    result = builder.build(pool, 0, n_songs=n_songs)

            # ── Display playlist ───────────────────────────────────────────────
            st.subheader(f"Your {mode.title()} Playlist")

            display_cols = ["track_name", "artist", "genre", "tempo", "energy",
                           "valence", "danceability"]
            if mode == "transition":
                display_cols += ["transition_score"]
            if "similarity" in result.columns:
                display_cols += ["similarity"]

            show_cols = [c for c in display_cols if c in result.columns]
            st.dataframe(result[show_cols], use_container_width=True)

            # ── Vibe radar chart ───────────────────────────────────────────────
            vibe_features = ["danceability", "energy", "valence",
                            "acousticness", "instrumentalness"]
            vibe_present  = [c for c in vibe_features if c in result.columns]

            if vibe_present:
                st.subheader("Playlist Vibe Profile")
                avg_vibe = result[vibe_present].mean()

                fig = go.Figure(go.Scatterpolar(
                    r=avg_vibe.values,
                    theta=avg_vibe.index.tolist(),
                    fill="toself",
                    name="Playlist Avg",
                ))
                seed_vibe = tracks.loc[seed_idx, vibe_present]
                fig.add_trace(go.Scatterpolar(
                    r=seed_vibe.values,
                    theta=seed_vibe.index.tolist(),
                    fill="toself",
                    name="Seed Track",
                    opacity=0.5,
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Energy arc ────────────────────────────────────────────────────
            if "energy" in result.columns:
                st.subheader("Energy Arc")
                fig2 = px.line(
                    result.reset_index(),
                    x="index",
                    y="energy",
                    markers=True,
                    labels={"index": "Track #", "energy": "Energy"},
                )
                fig2.update_layout(height=250)
                st.plotly_chart(fig2, use_container_width=True)

            # ── Transition scores ─────────────────────────────────────────────
            if mode == "transition" and "transition_score" in result.columns:
                avg_trans = result["transition_score"].dropna().mean()
                st.metric("Avg Transition Score", f"{avg_trans:.3f}",
                          help="1.0 = perfectly smooth transition, 0.0 = jarring")


# ── Dataset stats ─────────────────────────────────────────────────────────────
with st.expander("Dataset Overview"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tracks", f"{len(tracks):,}")
    if "genre" in tracks.columns:
        col2.metric("Genres", f"{tracks['genre'].nunique():,}")
    if "artist" in tracks.columns:
        col3.metric("Artists", f"{tracks['artist'].nunique():,}")

    if "genre" in tracks.columns:
        genre_counts = tracks["genre"].value_counts().head(20)
        fig = px.bar(genre_counts, title="Top 20 Genres", labels={"value": "Count", "index": "Genre"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
