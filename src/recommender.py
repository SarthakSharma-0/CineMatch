"""CineMatch recommendation engine.

A content-based movie recommender built with:
  - Pandas       : load, clean and group the MovieLens data
  - NumPy        : average and sort similarity scores
  - Scikit-learn : TF-IDF (text -> numbers) and cosine similarity
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------- 1. Load and clean the data ----------
def load_data():
    """Read the three MovieLens CSV files and drop duplicate / empty rows."""
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")

    movies = (
        movies.drop_duplicates(subset="movieId")
        .dropna(subset=["title", "genres"])
        .reset_index(drop=True)
    )
    ratings = ratings.dropna(subset=["movieId", "rating"])
    tags = tags.dropna(subset=["movieId", "tag"])
    return movies, ratings, tags


# ---------- 2. Build the recommender (TF-IDF + cosine similarity) ----------
def build_recommender(movies, ratings, tags):
    """Return (data, similarity, vectorizer)."""
    # Join all user tags of a movie into one string
    tag_text = tags.groupby("movieId")["tag"].apply(
        lambda values: " ".join(values.astype(str))
    )

    data = movies.copy()
    data["tags"] = data["movieId"].map(tag_text).fillna("")

    # One text string per movie = genres + tags
    data["features"] = (
        data["genres"].str.replace("|", " ", regex=False) + " " + data["tags"]
    ).str.lower()

    # TF-IDF: distinctive words get a high weight, common words a low one
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["features"])

    # Cosine similarity: angle between two movie vectors (1 = very similar)
    similarity = cosine_similarity(tfidf_matrix, dense_output=False)

    # Average rating and number of ratings per movie
    rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    rating_stats.columns = ["movieId", "avg_rating", "rating_count"]
    data = data.merge(rating_stats, on="movieId", how="left")
    data["avg_rating"] = data["avg_rating"].fillna(0)
    data["rating_count"] = data["rating_count"].fillna(0).astype(int)

    return data, similarity, vectorizer


# ---------- 3. Small helpers ----------
def get_titles(data):
    """Sorted list of all movie titles."""
    return sorted(data["title"].unique())


def get_genres(data):
    """Sorted list of all individual genres."""
    genres = data["genres"].str.split("|").explode().unique()
    return sorted(g for g in genres if g and g != "(no genres listed)")


def _find_index(title, data):
    """Row number of a movie: exact title first, then partial match."""
    title = (title or "").strip()
    if not title:
        return None
    exact = data[data["title"].str.lower() == title.lower()]
    if not exact.empty:
        return exact.index[0]
    partial = data[data["title"].str.contains(title, case=False, regex=False, na=False)]
    return None if partial.empty else partial.index[0]


def _score_row(index, similarity):
    """Similarity of one movie to every movie, as a flat NumPy array."""
    row = similarity[index]
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row).ravel()


def _rank(scores, data, exclude, genre, min_ratings, limit):
    """Attach scores, apply filters, return the top `limit` movies."""
    results = data.copy()
    results["similarity"] = scores
    results["match_pct"] = (scores * 100).round().astype(int)

    results = results[~results.index.isin(exclude)]
    results = results[results["rating_count"] >= min_ratings]
    if genre and genre != "All":
        results = results[results["genres"].str.contains(genre, case=False, regex=False)]

    return results.sort_values(["similarity", "avg_rating"], ascending=False).head(limit)


# ---------- 4. Recommendations ----------
def recommend(title, data, similarity, genre=None, min_ratings=0, limit=10):
    """Movies most similar to ONE selected movie."""
    index = _find_index(title, data)
    if index is None:
        return pd.DataFrame()

    scores = _score_row(index, similarity)
    return _rank(scores, data, [index], genre, min_ratings, limit)


def recommend_multi(titles, data, similarity, genre=None, min_ratings=0, limit=10):
    """Movies similar to SEVERAL favourites, using the average of their scores."""
    indices = [i for i in (_find_index(t, data) for t in titles) if i is not None]
    if not indices:
        return pd.DataFrame()

    all_scores = np.vstack([_score_row(i, similarity) for i in indices])
    scores = all_scores.mean(axis=0)  # NumPy average across the chosen movies
    return _rank(scores, data, indices, genre, min_ratings, limit)


def explain_match(source_titles, rec_title, data):
    """Say WHY a movie was recommended (shared genres and shared tag words)."""
    if isinstance(source_titles, str):
        source_titles = [source_titles]

    rec_rows = data[data["title"] == rec_title]
    if rec_rows.empty:
        return ""
    rec = rec_rows.iloc[0]
    rec_genres = set(rec["genres"].split("|"))
    rec_tags = set(rec["tags"].lower().split())

    shared_genres, shared_tags = set(), set()
    for title in source_titles:
        index = _find_index(title, data)
        if index is None:
            continue
        src = data.loc[index]
        shared_genres |= set(src["genres"].split("|")) & rec_genres
        shared_tags |= {
            t for t in set(src["tags"].lower().split()) & rec_tags if len(t) > 3
        }

    parts = []
    if shared_genres:
        parts.append("shared genres: " + ", ".join(sorted(shared_genres)))
    if shared_tags:
        parts.append("shared tags: " + ", ".join(sorted(shared_tags)[:5]))
    return "; ".join(parts) if parts else "similar overall wording in genres and tags"


# ---------- 5. Hidden gems and insights ----------
def hidden_gems(data, min_ratings=10, limit=10, genre=None):
    """Well-rated movies ranked with a weighted rating.

    weighted = (v / (v + m)) * R + (m / (v + m)) * C
      R = movie's average rating, v = its number of ratings,
      C = average rating of all movies, m = minimum-votes weight.
    Movies with few ratings are pulled toward C, so one 5-star rating
    cannot beat a movie with hundreds of ratings averaging 4.5.
    """
    results = data[data["rating_count"] >= min_ratings].copy()
    if genre and genre != "All":
        results = results[results["genres"].str.contains(genre, case=False, regex=False)]
    if results.empty:
        return results

    global_mean = data.loc[data["rating_count"] > 0, "avg_rating"].mean()
    m = max(min_ratings, 20)
    v = results["rating_count"]
    r = results["avg_rating"]
    results["gem_score"] = (v / (v + m)) * r + (m / (v + m)) * global_mean
    return results.sort_values("gem_score", ascending=False).head(limit)


def genre_stats(data):
    """Number of movies and average rating per genre (for the insights chart)."""
    exploded = data.assign(genre=data["genres"].str.split("|")).explode("genre")
    exploded = exploded[
        (exploded["genre"] != "(no genres listed)") & (exploded["rating_count"] > 0)
    ]
    return (
        exploded.groupby("genre")
        .agg(movies=("movieId", "count"), avg_rating=("avg_rating", "mean"))
        .round(2)
        .sort_values("movies", ascending=False)
    )