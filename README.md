# AI Hobby Recommendation System

A content-based hobby recommender built with Python and Streamlit. It uses TF-IDF and cosine similarity to suggest hobbies similar to one you already enjoy, based on categories and descriptions.

## Features

- Hobby picker with top-N recommendations
- Category and popularity filters
- Similarity scores and short explanations for each result

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit UI
- `recommender.py` — recommendation engine
- `hobbies.csv` — sample dataset
