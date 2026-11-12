# CineMatch — Personalised Movie Discovery

CineMatch is a beginner-friendly movie recommendation prototype built with Python, NumPy, Pandas, Scikit-learn and Streamlit.

## Features
- Recommend movies similar to a selected title using genres and user-generated tags.
- Filter recommendations by genre and minimum rating count.
- Explore highly rated less-mainstream titles in the Hidden Gems tab.
- Inspect basic dataset statistics.

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

The dataset is MovieLens Latest Small from GroupLens. Run `download_data.py` to download it into `data/`.

## How it works
Pandas loads and cleans the CSV files, then groups tags by movie. Scikit-learn's `TfidfVectorizer` converts genre/tag text into numerical vectors. Cosine similarity compares those vectors to find movies with similar descriptions. NumPy is used to work with the similarity scores. The hidden-gem ranking uses a weighted rating score so a movie with very few ratings does not automatically dominate.

## Limitations
This is a content-based prototype, not a production recommendation system. It relies on available genres and tags, may recommend similar rather than surprising movies, and does not yet learn an individual user's taste from their personal rating history.
