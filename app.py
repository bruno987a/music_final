import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval
from random import choice
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================================================
# Paths / Config
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def get_conn():
    return sqlite3.connect(DATA_DIR / "app.db")


DB = get_conn()

st.set_page_config(
    page_title="Smart Playlist Generator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Global Styles (Cookable-ish, with per-step cards only)
# =========================================================
st.markdown(
    """
    <style>
    /* Overall background – white */
    .stApp {
        background: #ffffff;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    /* Only INNER vertical blocks that contain .step-card get card styling
       (so NOT the whole main page) */
    .block-container div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]:has(.step-card) {
        background: radial-gradient(circle at top left, #fdfbfb 0, #ebedee 40%, #f7f7f7 100%);
        border-radius: 1.1rem;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin-bottom: 1rem;
    }

    /* The marker itself is invisible */
    .step-card {
        height: 0;
        margin: 0;
        padding: 0;
    }

    /* Sidebar steps */
    .step-label {
        padding: 0.35rem 0.5rem;
        border-radius: 0.8rem;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .step-done {
        background: rgba(34, 197, 94, 0.12);
        color: #166534;
    }
    .step-current {
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
    }
    .step-todo {
        background: rgba(148, 163, 184, 0.12);
        color: #475569;
    }

    /* Song list rows */
    .song-row {
        background: #fafafa;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .song-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #111827;
    }
    .song-artist {
        font-size: 0.82rem;
        color: #6b7280;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 20px 0 5px 0;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Session State Initialization
# =========================================================
defaults = {
    "step": 1,
    "num_raters": 1,
    "rater_names": ["User 1"],
    "active_rater_idx": 0,
    "ratings": {},
    "criteria_confirmed": False,
    "evaluation_done": False,
    "final_success_message": False,
    "chosen_genre": None,
    "n_desired_songs": 15,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# Data Loading (cached)
# =========================================================
@st.cache_data(show_spinner=False)
def load_genre_map():
    gmi = pd.read_sql_query("SELECT * FROM genre_with_main_identity", DB)
    return gmi[["genre_id", "main_category_id"]]


@st.cache_data(show_spinner=False)
def load_tracks():
    t = pd.read_sql_query("SELECT * FROM tracks_small", DB)
    df = pd.DataFrame(
        {
            "track_id": t["track_id"],
            "genres_all": t["genres_all"].fillna("[]").apply(literal_eval),
            "title": t["title"],
            "artist": t["artist"],
        }
    )
    return df


@st.cache_data(show_spinner=False)
def load_features():
    features = pd.read_csv(DATA_DIR / "reduced_features.csv", index_col=0)
    feature_cols = [
        "mfcc_01_mean",
        "mfcc_02_mean",
        "mfcc_03_mean",
        "mfcc_04_mean",
        "mfcc_05_mean",
        "mfcc_06_mean",
        "mfcc_07_mean",
        "mfcc_08_mean",
        "mfcc_09_mean",
        "mfcc_10_mean",
        "rmse_01_mean",
        "spectral_centroid_01_mean",
        "spectral_bandwidth_01_mean",
        "chroma_var",
    ]
    features_14 = features[feature_cols].copy()
    scaler = StandardScaler()
    X_14 = scaler.fit_transform(features_14)
    features_14_scaled = pd.DataFrame(X_14, index=features.index, columns=feature_cols)
    return features_14_scaled


# =========================================================
# Utility Functions
# =========================================================
def rand_track_genre(main_cat_id: int, n: int) -> pd.DataFrame:
    """Pick n random tracks within a chosen main genre."""
    s_genres = load_genre_map()
    s_t = load_tracks()

    genre_ids = list(
        set(s_genres.loc[s_genres["main_category_id"] == main_cat_id, "genre_id"])
    )
    if not genre_ids:
        return s_t.sample(n=min(n, len(s_t)))

    rand_gen_l = [choice(genre_ids) for _ in range(n)]

    p_to_rate = []
    for g_id in rand_gen_l:
        poss_songs = s_t[s_t["genres_all"].apply(lambda ids: g_id in ids)]
        if len(poss_songs) == 0:
            continue
        p_to_rate.append(poss_songs.sample(1))
    if not p_to_rate:
        return s_t.sample(n=min(n, len(s_t)))
    return pd.concat(p_to_rate, ignore_index=True)


def weight_adjustment(points: int) -> float:
    """Transform ratings: dislikes pull away, likes push stronger."""
    return (points / 3.0) ** 2


def build_user_profile(ratings_list, rated_ids, features_df):
    ratings = np.asarray(ratings_list, dtype=float)
    vecs = features_df.loc[rated_ids].values
    return np.average(vecs, axis=0, weights=ratings)


def recommend_group_playlist(ratings_dict, n_songs: int):
    features_14_scaled = load_features()
    user_profiles = []

    for _, rating_dict in ratings_dict.items():
        if not rating_dict:
            continue

        rated_ids = [tid for tid in rating_dict.keys() if tid in features_14_scaled.index]
        if not rated_ids:
            continue

        ratings_list = [weight_adjustment(rating_dict[tid]) for tid in rated_ids]
        user_profiles.append(
            build_user_profile(ratings_list, rated_ids, features_14_scaled)
        )

    if len(user_profiles) == 0:
        return []

    group_vector = np.mean(user_profiles, axis=0)
    X = features_14_scaled.values
    track_ids = features_14_scaled.index.to_numpy()

    knn_model = NearestNeighbors(metric="cosine", n_neighbors=min(200, len(track_ids)))
    knn_model.fit(X)

    _, nn_idx = knn_model.kneighbors(group_vector.reshape(1, -1), n_neighbors=n_songs)
    return track_ids[nn_idx[0]].tolist()


# =========================================================
# Sidebar / Progress
# =========================================================
def render_sidebar():
    st.sidebar.title("Progress")

    step0_done = st.session_state.step > 1
    step1_done = st.session_state.criteria_confirmed
    step2_done = st.session_state.evaluation_done
    step3_done = st.session_state.step >= 4

    steps = [
        ("Step 0 – Group setup", step0_done),
        ("Step 1 – Criteria", step1_done),
        ("Step 2 – Quick evaluation", step2_done),
        ("Step 3 – Final playlist", step3_done),
    ]

    current_step = st.session_state.step

    for idx, (label, done) in enumerate(steps, start=1):
        if done:
            css = "step-label step-done"
            icon = "✅"
        elif idx == current_step:
            css = "step-label step-current"
            icon = "▶️"
        else:
            css = "step-label step-todo"
            icon = "▫️"

        st.sidebar.markdown(
            f"<div class='{css}'>{icon} {label}</div>", unsafe_allow_html=True
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tip: You can always come back and start over once you’ve seen your playlist."
    )


# =========================================================
# Step 0 – Group Setup
# =========================================================
def step_group_setup():
    with st.container():
        # Marker so the whole block becomes a card
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Step 0 – Group setup")
        st.caption(
            "Add everyone who will rate songs. We’ll combine all tastes into one smart playlist."
        )

        if st.session_state.step == 1:
            col1, col2 = st.columns([1, 2])

            with col1:
                num = st.number_input(
                    "Number of raters",
                    min_value=1,
                    max_value=10,
                    value=int(st.session_state.num_raters),
                    step=1,
                    key="num_raters_input",
                )

            names = []
            with col2:
                for i in range(int(num)):
                    default_name = (
                        st.session_state.rater_names[i]
                        if i < len(st.session_state.rater_names)
                        else f"User {i+1}"
                    )
                    names.append(
                        st.text_input(
                            f"Rater {i+1} name",
                            value=default_name,
                            key=f"rater_name_{i}",
                        )
                    )

            if st.button("✅ Confirm group & continue", use_container_width=True):
                clean_names = [(n.strip() or f"User {i+1}") for i, n in enumerate(names)]
                st.session_state.num_raters = int(num)
                st.session_state.rater_names = clean_names
                st.session_state.ratings = {name: {} for name in clean_names}
                st.session_state.active_rater_idx = 0
                st.session_state.step = 2
                st.rerun()
        else:
            # New summary format
            total = st.session_state.num_raters
            names_display = ", ".join(st.session_state.rater_names)
            st.info(f"**Total raters:** {total} – {names_display}")


# =========================================================
# Step 1 – Criteria
# =========================================================
def step_criteria():
    if st.session_state.step < 2:
        return

    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Step 1 – Playlist generation criteria")
        st.caption(
            "Choose how focused the playlist should be and the kind of vibe you want."
        )

        genre_map = {
            "Rock/Metal/Punk": 1,
            "Pop/Synth": 2,
            "Electronic/IDM": 3,
            "Hip-Hop/RnB": 4,
            "Jazz/Blues": 5,
            "Classical": 6,
            "Folk/Country/Americana": 7,
            "World/Reggae/Latin": 8,
            "Experimental/Sound Art": 9,
            "Spoken/Soundtrack/Misc": 10,
            "Funk": 11,
        }

        reverse_genre_map = {v: k for k, v in genre_map.items()}

        if st.session_state.step == 2:
            col1, col2 = st.columns(2)

            with col1:
                similarity = st.selectbox(
                    "Similarity level",
                    ["None", "Genre", "Artist", "Mixed"],
                    index=0,
                    format_func=lambda x: "✨ Let the algorithm choose"
                    if x == "None"
                    else x,
                    key="similarity",
                )

            with col2:
                key_genre = st.selectbox("Preferred genre", list(genre_map.keys()))
                st.session_state.chosen_genre = genre_map[key_genre]

            st.session_state.n_desired_songs = st.slider(
                "Playlist length (number of songs)",
                5,
                30,
                15,
            )

            if st.button(
                "✅ Confirm criteria & start rating", use_container_width=True
            ):
                st.session_state.criteria_confirmed = True
                st.session_state.step = 3
                st.session_state.evaluation_done = False
                st.session_state.active_rater_idx = 0
                if "candidate_songs" in st.session_state:
                    del st.session_state["candidate_songs"]
                st.rerun()
        else:
            similarity_value = st.session_state.get("similarity", "None")
            chosen_genre_id = st.session_state.get("chosen_genre")
            chosen_genre_name = reverse_genre_map.get(chosen_genre_id, "Unknown")

            st.info(
                f"""
🎛️ **Criteria selected**  
• Similarity level: **{similarity_value}**  
• Genre: **{chosen_genre_name}**  
• Desired playlist length: **{st.session_state.n_desired_songs} songs**
"""
            )


# =========================================================
# Step 2 – Quick Evaluation
# =========================================================
def step_quick_evaluation():
    if st.session_state.step < 3 or not st.session_state.criteria_confirmed:
        return

    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Step 2 – Quick song evaluation")

        # Dynamic description depending on 1 vs many raters
        if st.session_state.num_raters > 1:
            st.caption(
                "Everyone rates a handful of songs. We’ll learn what the whole group likes and dislikes."
            )
        else:
            st.caption(
                "Please rate a handful of songs. We'll learn what you like and dislike."
            )

        rater_names = st.session_state.rater_names
        idx_rater = st.session_state.active_rater_idx
        current_user = rater_names[idx_rater]

        st.write(f"**Rater {idx_rater + 1} / {len(rater_names)}:** {current_user}")

        st.session_state.ratings.setdefault(current_user, {})
        user_ratings = st.session_state.ratings[current_user]

        # Generate candidate songs once
        if "candidate_songs" not in st.session_state:
            st.session_state.candidate_songs = rand_track_genre(
                st.session_state.chosen_genre, 5
            )

        songs_df = st.session_state.candidate_songs

        # Header row
        header_song_col, header_rating_col = st.columns([3, 2])
        with header_song_col:
            st.markdown(
                "<div style='background-color:#e5e7eb; padding:0.5rem; "
                "border-radius:0.75rem 0 0 0.75rem; font-weight:600;'>Songs</div>",
                unsafe_allow_html=True,
            )
        with header_rating_col:
            st.markdown(
                "<div style='background-color:#e5e7eb; padding:0.5rem; "
                "border-radius:0 0.75rem 0.75rem 0; font-weight:600; "
                "display:flex; justify-content:space-between; align-items:center;'>"
                "<span>Rating</span>"
                "<span style='font-weight:400; font-size:0.75rem;'>1 = dislike · 5 = love</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Data rows
        for i, row in songs_df.iterrows():
            song_col, rating_col = st.columns([3, 2])

            with song_col:
                st.markdown(
                    f"""
                    <div class="song-row">
                        <div class="song-title">{row['title']}</div>
                        <div class="song-artist">{row['artist']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rating_col:
                rating = st.slider(
                    label="",
                    min_value=1,
                    max_value=5,
                    value=int(user_ratings.get(row["track_id"], 3)),
                    key=f"rating_{current_user}_{i}",
                    step=1,
                )
                user_ratings[row["track_id"]] = rating

        # Navigation / action buttons – centered
        if idx_rater < len(rater_names) - 1:
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                if st.button(
                    "➡️ Save ratings & next person", use_container_width=True
                ):
                    st.session_state.active_rater_idx += 1
                    st.rerun()
        else:
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                if st.button(
                    "🎉 Generate final playlist",
                    type="primary",
                    use_container_width=True,
                ):
                    recommended_ids = recommend_group_playlist(
                        st.session_state.ratings,
                        st.session_state.n_desired_songs,
                    )

                    if not recommended_ids:
                        st.error(
                            "We didn’t get enough usable ratings to generate recommendations."
                        )
                        st.stop()

                    st.session_state.recommended_ids = recommended_ids
                    st.session_state.evaluation_done = True
                    st.session_state.step = 4
                    st.session_state.final_success_message = True
                    st.rerun()


# =========================================================
# Step 3 – Final Playlist
# =========================================================
def step_final_playlist():
    if st.session_state.step < 4 or not st.session_state.evaluation_done:
        return

    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Step 3 – Your final recommended playlist")

        if st.session_state.final_success_message:
            # Dynamic success message depending on number of raters
            if st.session_state.num_raters > 1:
                msg = "✅ Playlist generated based on the whole group’s preferences!"
            else:
                msg = "✅ Playlist generated based on your preferences!"
            st.success(msg)
            st.session_state.final_success_message = False

        s_t = load_tracks()
        df_final = s_t[s_t["track_id"].isin(st.session_state.recommended_ids)][
            ["title", "artist"]
        ]

        st.dataframe(
            df_final.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Summary**")
        st.write(f"- Total songs: **{len(df_final)}**")

        st.markdown("---")
        if st.button("🔁 Start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# =========================================================
# Main Layout
# =========================================================
render_sidebar()

st.markdown('<div class="main-title">Smart Playlist Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    "Create group playlists that balance everyone’s taste, using a little data magic."
    "</div>",
    unsafe_allow_html=True,
)

step_group_setup()
step_criteria()
step_quick_evaluation()
step_final_playlist()

st.markdown('<div class="footer">© 2025 Smart Playlist</div>', unsafe_allow_html=True)
