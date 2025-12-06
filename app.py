import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.neighbors import NearestNeighbors #Machine Learning algorithm @Lorenz
from sklearn.preprocessing import StandardScaler

candidate_songs = []

# Set up pathways to data folder
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# Set up SQLite connection to database as DB
def get_conn():
    return sqlite3.connect(DATA_DIR / "app.db")

DB = get_conn()



# -------------------------
# Streamlit app setup
# -------------------------
st.set_page_config(page_title="Smart Playlist Generator", page_icon="🎧", layout="wide")

st.title("Smart Playlist Generator")
st.markdown("Create personalized playlists based on your musical preferences and feedback.")

# Initialize session state for progress tracking
# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1   # start at group setup

if "num_raters" not in st.session_state:
    st.session_state.num_raters = 1

if "rater_names" not in st.session_state:
    st.session_state.rater_names = ["User 1"]

if "active_rater_idx" not in st.session_state:
    st.session_state.active_rater_idx = 0

if "ratings" not in st.session_state:
    st.session_state.ratings = {}

if "criteria_confirmed" not in st.session_state:
    st.session_state.criteria_confirmed = False

if "evaluation_done" not in st.session_state:
    st.session_state.evaluation_done = False

if "final_success_message" not in st.session_state:
    st.session_state.final_success_message = False

# store preferences
if "chosen_genre" not in st.session_state:
    st.session_state.chosen_genre = None

if "n_desired_songs" not in st.session_state:
    st.session_state.n_desired_songs = 15

# -------------------------
# Sidebar: progress indicator
# -------------------------
def render_sidebar():
    st.sidebar.title("🎧 Smart Playlist")

    # Define when each step is considered "done"
    step0_done = st.session_state.step > 1                      # Confirmed group
    step1_done = st.session_state.criteria_confirmed            # Confirmed criteria
    step2_done = st.session_state.evaluation_done               # Pressed "Generate Final Playlist"
    step3_done = st.session_state.step >= 4                     # At final playlist

    steps = [
        ("Step 0 – Group setup", step0_done),
        ("Step 1 – Criteria", step1_done),
        ("Step 2 – Quick evaluation", step2_done),
        ("Step 3 – Final playlist", step3_done),
    ]

    current_step = st.session_state.step

    for idx, (label, done) in enumerate(steps, start=1):
        if done:
            st.sidebar.markdown(f"✅ {label}")
        elif idx == current_step:
            st.sidebar.markdown(f"▶️ {label}")
        else:
            st.sidebar.markdown(f"▫️ {label}")

render_sidebar()


# -------------------------
# ---------Group Setup
# -------------------------
if st.session_state.step >= 1:
    st.header("Step 0 – Group setup")

    # BEFORE "Confirm group" is clicked → show editable inputs
    if st.session_state.step == 1:
        num = st.number_input(
            "How many people are going to rate?",
            min_value=1,
            max_value=10,
            value=int(st.session_state.num_raters),
            step=1,
            key="num_raters_input"
        )

        # name inputs
        names = []
        for i in range(int(num)):
            default_name = (
                st.session_state.rater_names[i]
                if i < len(st.session_state.rater_names)
                else f"User {i+1}"
            )
            names.append(
                st.text_input(f"Rater {i+1} name", value=default_name, key=f"rater_name_{i}")
            )

        if st.button("Confirm group"):
            clean_names = [(n.strip() or f"User {i+1}") for i, n in enumerate(names)]
            st.session_state.num_raters = int(num)
            st.session_state.rater_names = clean_names

            # initialize ratings dict per person
            st.session_state.ratings = {name: {} for name in clean_names}

            st.session_state.active_rater_idx = 0
            st.session_state.step = 2  # GO TO STEP 1

            st.rerun()

    # AFTER confirm group → show summary permanently
    else:
        st.info(
            " **Group:** "
            + ", ".join(st.session_state.rater_names)
            + f"  —  Total raters: {st.session_state.num_raters}"
        )
    



# -------------------------
# STEP 1 — Generation Criteria
# -------------------------
if st.session_state.step >= 2:
    st.header("Step 1 – Playlist generation criteria")

    # BEFORE confirming → show full criteria form
    if st.session_state.step == 2:
        similarity = st.selectbox(
            "Select similarity level:",
            ["None", "Genre", "Artist", "Mixed"],
            index=0,  # default selection is "None"
            format_func=lambda x: f"*{x}*" if x == "None" else x,
            key="similarity"
        )

        # Song selection for rating 
        genre_map = {
            "Rock/Metal/Punk": 1, "Pop/Synth": 2, "Electronic/IDM": 3,
            "Hip-Hop/RnB": 4, "Jazz/Blues": 5, "Classical": 6,
            "Folk/Country/Americana": 7, "World/Reggae/Latin": 8,
            "Experimental/Sound Art": 9, "Spoken/Soundtrack/Misc": 10,
            "Funk": 11
        }

        key_genre = st.selectbox("Select Genre:", list(genre_map.keys()))
        st.session_state.chosen_genre = genre_map[key_genre]
        st.session_state.n_desired_songs = st.slider(
            "Select desired playlist length (songs):",
            5, 30, 15
        )

        # button for continuing the workflow and start the rating process    
        if st.button("Confirm and Continue"):
            st.session_state.criteria_confirmed = True
            st.session_state.step = 3
            st.session_state.evaluation_done = False
            st.session_state.active_rater_idx = 0
            if "candidate_songs" in st.session_state:
                del st.session_state.candidate_songs

            st.rerun()

    # AFTER confirming → show criteria summary
    else:
        reverse_genre_map = {
            1: "Rock/Metal/Punk", 2: "Pop/Synth", 3: "Electronic/IDM",
            4: "Hip-Hop/RnB", 5: "Jazz/Blues", 6: "Classical",
            7: "Folk/Country/Americana", 8: "World/Reggae/Latin",
            9: "Experimental/Sound Art", 10: "Spoken/Soundtrack/Misc",
            11: "Funk"
        }

        # SAFE access with defaults
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


# -------------------------
# STEP 2 — Quick Evaluation
# -------------------------
if st.session_state.step >= 3 and st.session_state.criteria_confirmed:
    st.header("Step 2 – Quick song evaluation")

    rater_names = st.session_state.rater_names
    idx_rater = st.session_state.active_rater_idx
    current_user = rater_names[idx_rater]

    st.write(f"Rater **{idx_rater + 1} / {len(rater_names)}**: **{current_user}**")
    st.write("### Please rate the following songs:")

    # make sure this user's dict exists
    st.session_state.ratings.setdefault(current_user, {})
    user_ratings = st.session_state.ratings[current_user]

    from ast import literal_eval
    from random import choice

    gmi = pd.read_sql_query("SELECT * FROM genre_with_main_identity", DB)                                     #reading in the list with all subgenres linked with the main genres
    s_genres = gmi[["genre_id", "main_category_id"]]                                                          #filtering out the needed genre column

    t = pd.read_sql_query("SELECT * FROM tracks_small", DB)                                                   #importing the table with the tracks
    s_t = pd.DataFrame({                                                                                      #clean out the table whilst implementing it as a dataframe 
        "track_id": t["track_id"],
        "genres_all": t["genres_all"].fillna("[]").apply(literal_eval),                                       #we're changing the Genre numbers from type string to actual python format, empty ones would be transfered to []
        "title": t["title"],
        "artist": t["artist"]
    })

    def rand_track_genre(main_cat_id, n):                                                                     #implementing the function giving out random songs, with input of number of songs to rate (n) and the chosen main genre (main_cat_id) 
        genre_ids = list(set(s_genres.loc[s_genres["main_category_id"] == main_cat_id, "genre_id"]))          #constructing a list with all sub genres matching the chosen genre
        rand_gen_l = [choice(genre_ids) for _ in range(n)]                                                    #creating a list with n randomly chosen sub genres out the just created list

        p_to_rate = []
        for g_id in rand_gen_l:                                                                               #for every randomly chosen sub genre we choose one song that has this sub genre in the following lines
            poss_songs = s_t[s_t["genres_all"].apply(lambda ids: g_id in ids)]                                #we create a list of songs with the current sub genre g_id
            p_to_rate.append(poss_songs.sample(1))                                                            #one of the songs gets randomly chosen from this list and appended to the list of songs that will be displayed for rating
        return pd.concat(p_to_rate, ignore_index=True)

    # Generate candidate songs ONCE for the whole group
    if "candidate_songs" not in st.session_state:
        st.session_state.candidate_songs = rand_track_genre(st.session_state.chosen_genre, 5)

    songs_df = st.session_state.candidate_songs


# Header row (Songs / Rating)
    header_song_col, header_rating_col = st.columns([3, 2])
    with header_song_col:
        st.markdown(
            "<div style='background-color:#f0f2f6; padding:0.5rem; "
            "border-radius:0.5rem 0 0 0.5rem; font-weight:600;'>Songs</div>",
            unsafe_allow_html=True,
        )

    with header_rating_col:
        st.markdown(
            "<div style='background-color:#f0f2f6; padding:0.5rem; "
            "border-radius:0 0.5rem 0.5rem 0; font-weight:600; display:flex; "
            "justify-content:space-between; align-items:center;'>"
            "<span>Rating</span>"
            "<span style='font-weight:400; font-size:0.75rem;'>1 = dislike · 5 = love</span>"
            "</div>",
            unsafe_allow_html=True,
        )

# Data rows
    for i, row in songs_df.iterrows():
        song_col, rating_col = st.columns([3, 2])

        # SONG CELL
        with song_col:
         st.markdown(
                f"""
                <div style='background-color:#fafafa; padding:0.5rem; 
                            border-bottom:1px solid #e0e0e0;'>
                    <strong>{row['title']}</strong><br>
                    <span style='font-size:0.85rem; color:#555;'>{row['artist']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # RATING CELL (NO LABEL)
        with rating_col:
            rating = st.slider(
                label="",                # <-- removes the label
                min_value=1,
                max_value=5,
                value=int(user_ratings.get(row["track_id"], 3)),
                key=f"rating_{current_user}_{i}",
                step=1,
            )

    # Save rating
        user_ratings[row["track_id"]] = rating



    # Buttons to go to next person
    if idx_rater < len(rater_names) - 1:
        if st.button("Save ratings & Next person"):
            st.session_state.active_rater_idx += 1
            st.rerun()
    else:
        # allow generation for last rater
        if st.button("Generate Final Playlist"):

            

            # ------------------------------
            # START MACHINE LEARNING PART
            # ------------------------------
            features = pd.read_csv("data/reduced_features.csv", index_col=0)

            feature_cols = [
                "mfcc_01_mean", "mfcc_02_mean", "mfcc_03_mean", "mfcc_04_mean", "mfcc_05_mean",
                "mfcc_06_mean", "mfcc_07_mean", "mfcc_08_mean", "mfcc_09_mean", "mfcc_10_mean",
                "rmse_01_mean",
                "spectral_centroid_01_mean",
                "spectral_bandwidth_01_mean",
                "chroma_var"
            ]
            features_14 = features[feature_cols].copy()

            scaler = StandardScaler()
            X_14 = scaler.fit_transform(features_14)
            features_14_scaled = pd.DataFrame(X_14, index=features.index, columns=feature_cols)

            def build_user_profile(ratings_list, rated_ids, features_df):
                ratings = np.asarray(ratings_list, dtype=float)
                vecs = features_df.loc[rated_ids].values
                return np.average(vecs, axis=0, weights=ratings)
            
            def weight_adjustment(points: int) -> float:                                                        # Transform ratings of 1 & 2 into negative weights for the vector, and ratings of 3 to neutral
                return (points / 3.0) ** 2

            user_profiles = []
            for username, rating_dict in st.session_state.ratings.items():
                if not rating_dict:
                    continue

                rated_ids = [tid for tid in rating_dict.keys() if tid in features_14_scaled.index]
                if not rated_ids:
                    continue

                ratings_list = [weight_adjustment(rating_dict[tid]) for tid in rated_ids]
                user_profiles.append(build_user_profile(ratings_list, rated_ids, features_14_scaled))

            if len(user_profiles) == 0:
                st.error("There are no usable ratings - no recommendation possible.")
                st.stop()

            group_vector = np.mean(user_profiles, axis=0)

            X = features_14_scaled.values
            track_ids = features_14_scaled.index.to_numpy()

            knn_model = NearestNeighbors(metric="cosine", n_neighbors=200)
            knn_model.fit(X)

            def recommend(group_vec, n_songs):
                _, nn_idx = knn_model.kneighbors(group_vec.reshape(1, -1), n_neighbors=n_songs)
                return track_ids[nn_idx[0]]

            recommended_ids = recommend(group_vector, st.session_state.n_desired_songs).tolist()

            # store for step 4
            st.session_state.recommended_ids = recommended_ids
            st.session_state.evaluation_done = True
            st.session_state.step = 4

            # 🔹 tell next run to show success message on the final page
            st.session_state.final_success_message = True

            # 🔹 force rerun so sidebar + final playlist update immediately
            st.rerun() 

          # -------------------------
            # END MACHINE LEARNING
            # -------------------------
            # STEP 3 — Final Playlist
            # -------------------------
if st.session_state.step >= 4 and st.session_state.evaluation_done:
    st.header("Step 3 – Your final recommended playlist")
    st.write("Generated based on your preferences and evaluations:")

    df_final = s_t[s_t["track_id"].isin(st.session_state.recommended_ids)][["title", "artist"]]
    st.dataframe(df_final, use_container_width=True)

    st.markdown("**Summary:**")
    st.write(f"- Total songs: {len(df_final)}")

    if st.button(" Start Over"):
        # Completely clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Rerun to reinitialize everything
        st.rerun()  # use st.experimental_rerun() if your Streamlit version is older

st.markdown(
    "<div style='text-align: center; color: #888; padding: 20px;'>"
    "© 2025 Cookable"
    "</div>",
    unsafe_allow_html=True
)


