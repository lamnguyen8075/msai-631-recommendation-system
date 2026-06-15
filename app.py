"""Streamlit GUI for a content-based hobby recommendation system."""

from __future__ import annotations

import streamlit as st
from recommender import HobbyRecommender


st.set_page_config(
    page_title="AI Hobby Recommendation System",
    page_icon="🎯",
    layout="centered",
)


@st.cache_resource
def load_recommender() -> HobbyRecommender:
    return HobbyRecommender("hobbies.csv")


recommender = load_recommender()

st.title("🎯 AI Hobby Recommendation System")
st.write(
    "This app recommends hobbies using a content-based AI technique. "
    "It compares hobby categories and descriptions with TF-IDF and cosine similarity."
)

with st.sidebar:
    st.header("Recommendation Controls")
    selected_hobby = st.selectbox("Choose a hobby you like", recommender.get_titles())
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    category_filter = st.selectbox("Optional category filter", recommender.available_categories())
    min_popularity = st.slider("Minimum popularity", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

st.subheader(f"Recommendations based on: {selected_hobby}")

recommendations = recommender.recommend(
    title=selected_hobby,
    top_n=top_n,
    category_filter=category_filter,
    min_popularity=min_popularity,
)

if not recommendations:
    st.warning("No recommendations matched the selected filters. Try lowering the popularity threshold or changing the category filter.")
else:
    for rank, hobby in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"### {rank}. {hobby.title}")
            st.write(f"**Categories:** {hobby.categories}")
            st.write(f"**Difficulty:** {hobby.difficulty}/10")
            st.write(f"**Popularity:** {hobby.popularity}")
            st.write(f"**Similarity Score:** {hobby.similarity_score}")
            st.caption(hobby.explanation)

st.divider()
st.subheader("HCI and Ethics Notes")
st.write(
    "The interface is designed to be transparent by showing why each recommendation was produced. "
    "Users can adjust filters, which supports control and reduces the feeling that the algorithm is a black box."
)
