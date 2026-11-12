# 🎬 CineMatch

A content-based movie recommendation web app built with **Python, Pandas, NumPy and Scikit-learn**, with a **Streamlit** interface.

Live demo: _add your .streamlit.app link here, or delete this line_

## What it does

- Pick one or more favourite movies and get similar recommendations
- Shows a **match percentage** for every recommendation
- Explains **why** a movie was recommended (shared genres and tags)
- Filters by genre and by minimum number of ratings
- **Hidden gems** tab: well-rated movies ranked fairly, even if few people rated them
- **Insights** tab: movies per genre, average rating per genre, most-rated movies

## How it works

1. **Load and clean (Pandas):** read movies, ratings and tags from the MovieLens CSV files and drop duplicate or empty rows.
2. **Build features:** for each movie, join its genres and all user tags into one text string.
3. **TF-IDF (Scikit-learn):** convert that text into numbers. Words that are distinctive to a movie get a high weight, and words found in almost every movie get a low weight.
4. **Cosine similarity (Scikit-learn):** compare every pair of movies by the angle between their vectors. A score close to 1 means very similar content.
5. **Recommend (NumPy + Pandas):** take the similarity scores of the chosen movie(s), average them if there are several (NumPy), apply the filters, and return the top results.
6. **Hidden gems:** a weighted rating that pulls movies with few ratings toward the overall average, so one 5-star rating cannot beat a movie that hundreds of people rated highly.

## Tech used

| Library | Used for |
| --- | --- |
| Pandas | Loading CSVs, cleaning, grouping, filtering |
| NumPy | Averaging similarity scores across chosen movies |
| Scikit-learn | `TfidfVectorizer` and `cosine_similarity` |
| Streamlit | The interactive web interface |

## Project structure

```text
CineMatch/
├── app.py              # Streamlit interface
├── download_data.py    # Downloads the MovieLens dataset
├── requirements.txt
├── src/
│   └── recommender.py  # Data loading, TF-IDF, similarity, recommendations
└── data/               # movies.csv, ratings.csv, tags.csv
```

## Run it locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The `data` folder already contains the CSV files. If they are missing, run `python download_data.py` first, or download the small MovieLens dataset from GroupLens and copy `movies.csv`, `ratings.csv` and `tags.csv` into `data/`.

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) (small version) by GroupLens Research.

## Limitations

- Recommendations use only genres and user tags. Many movies have no tags, so some matches rely on genre alone.
- It is content-based, so it does not learn from individual users' tastes.
- Adding plot summaries, cast or director information would improve the results.

## Built with AI assistance

I used an AI assistant (Claude) to help build and learn this project, including the starter structure and parts of the code. I went through the code to understand each function, and I can explain the algorithms used.

## Author

Sarthak Sharma, first-year IT, GTBIT