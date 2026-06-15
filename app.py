"""Streamlit GUI for a content-based movie recommendation system."""

from __future__ import annotations

import streamlit as st
from recommender import MovieRecommender


st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource
def load_recommender() -> MovieRecommender:
    return MovieRecommender("movies.csv")


recommender = load_recommender()

st.title("🎬 AI Movie Recommendation System")
st.write(
    "This app recommends movies using a content-based AI technique. "
    "It compares movie genres and descriptions with TF-IDF and cosine similarity."
)

with st.sidebar:
    st.header("Recommendation Controls")
    selected_movie = st.selectbox("Choose a movie you like", recommender.get_titles())
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    genre_filter = st.selectbox("Optional genre filter", recommender.available_genres())
    min_rating = st.slider("Minimum rating", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

st.subheader(f"Recommendations based on: {selected_movie}")

recommendations = recommender.recommend(
    title=selected_movie,
    top_n=top_n,
    genre_filter=genre_filter,
    min_rating=min_rating,
)

if not recommendations:
    st.warning("No recommendations matched the selected filters. Try lowering the rating or changing the genre filter.")
else:
    for rank, movie in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"### {rank}. {movie.title} ({movie.year})")
            st.write(f"**Genres:** {movie.genres}")
            st.write(f"**Rating:** {movie.rating}")
            st.write(f"**Similarity Score:** {movie.similarity_score}")
            st.caption(movie.explanation)

st.divider()
st.subheader("HCI and Ethics Notes")
st.write(
    "The interface is designed to be transparent by showing why each recommendation was produced. "
    "Users can adjust filters, which supports control and reduces the feeling that the algorithm is a black box."
)
