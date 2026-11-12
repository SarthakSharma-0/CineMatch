# CineMatch — Interview Notes

## 30-second explanation
CineMatch is a content-based movie recommendation prototype. A user selects a movie, and the system finds other movies with similar genres and tags. Pandas handles the data, Scikit-learn converts text into TF-IDF vectors and calculates cosine similarity, and NumPy helps work with the resulting numerical scores. Streamlit provides the interface.

## Key concepts
- **NumPy:** A Python library for efficient numerical arrays and calculations.
- **Pandas:** A library for tabular data. A DataFrame is like a programmable spreadsheet.
- **TF-IDF:** Gives words higher importance when they are informative for a document and less importance when they appear everywhere.
- **Cosine similarity:** Measures how close two vectors point in the same direction. A higher score means more similar text features.
- **Content-based filtering:** Recommends items with features similar to an item the user already likes.

## Important functions
- `load_data()`: reads the CSV files and removes incomplete/duplicate records.
- `build_recommender()`: combines genres and tags, creates TF-IDF vectors and computes similarities.
- `recommend()`: finds the selected title and ranks other movies by similarity, with optional filters.
- `hidden_gems()`: uses a weighted rating formula to balance rating average and number of ratings.

## Limitations / future improvements
- It does not yet use collaborative filtering or a user's personal history.
- Some movies have few or no tags, so recommendations may rely mostly on genres.
- Could add user profiles, collaborative filtering, poster images and evaluation metrics such as precision@k.
