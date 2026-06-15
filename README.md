# AI Movie Recommendation System

A content-based movie recommender built with Python and Streamlit. It uses TF-IDF and cosine similarity to suggest movies similar to one you already enjoy, based on genres and descriptions.

## Features

- Movie picker with top-N recommendations
- Genre and rating filters
- Similarity scores and short explanations for each result

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit UI
- `recommender.py` — recommendation engine
- `movies.csv` — sample dataset
