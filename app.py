import streamlit as st
import pandas as pd
import pickle
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_movies():
    with open("movie.pkl", "rb") as file:
        return pickle.load(file)

movies = load_movies()

movies = load_movies()

if isinstance(movies, dict):
    movies = movies.get("movie") or movies.get("movies")

# ----------------------------
# Build cosine similarity
# ----------------------------
@st.cache_data
def build_similarity(df):
    df = df.copy()
    df["tags"] = df["tags"].fillna("").astype(str)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()

    return cosine_similarity(vectors)

cosine_sim = build_similarity(movies)

# ----------------------------
# Recommendation logic
# ----------------------------
def get_recommendations(title):
    idx = movies[movies["title"] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "movie_id"]]

# ----------------------------
# Poster fetch (TMDB)
# ----------------------------
def fetch_poster(movie_id):
    api_key = "12f9594299752132f38d70494e4c9992"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    response = requests.get(url)
    data = response.json()

    if data.get("poster_path"):
        return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    return None

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Movie Recommendation System")
st.write("Content-based recommender using **cosine similarity on movie tags**")

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)

    st.subheader("Top 10 Similar Movies")

    cols = st.columns(5)

    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster = fetch_poster(row["movie_id"])

        with cols[i % 5]:
            if poster:
                st.image(poster, use_column_width=True)
            st.caption(row["title"])
