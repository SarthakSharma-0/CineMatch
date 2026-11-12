import streamlit as st

from src.recommender import (
    load_data,
    build_recommender,
    recommend_multi,
    explain_match,
    hidden_gems,
    genre_stats,
    get_titles,
    get_genres,
)

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# Small bit of styling for the genre "chips"
st.markdown(
    """
    <style>
    .chip {
        display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
        border-radius: 12px; font-size: 0.78rem;
        background: rgba(255, 75, 75, 0.15); border: 1px solid rgba(255, 75, 75, 0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 CineMatch")
st.caption("Discover your next favourite movie through content-based recommendations.")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(movies, ratings, tags):
    return build_recommender(movies, ratings, tags)


try:
    movies, ratings, tags = get_data()
    data, similarity, vectorizer = get_model(movies, ratings, tags)
except Exception as error:
    st.error("Couldn't load the dataset. Run `python download_data.py` in the project terminal first.")
    st.exception(error)
    st.stop()

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("🎛️ Filters")
    genre_filter = st.selectbox("Genre", ["All"] + get_genres(data))
    min_votes = st.slider("Minimum number of ratings", 0, 100, 10, 5)
    limit = st.slider("How many recommendations", 5, 20, 10)
    st.divider()
    st.metric("Movies in catalogue", f"{len(data):,}")

tab1, tab2, tab3 = st.tabs(["🎯 Find similar movies", "💎 Hidden gems", "📊 Insights"])

# ---------- Tab 1: recommendations ----------
with tab1:
    st.subheader("Which movies do you like?")
    st.write("Pick one or more favourites. With several, CineMatch averages their similarity scores.")
    selected = st.multiselect(
        "Choose up to 5 movies",
        get_titles(data),
        max_selections=5,
        placeholder="Type a movie name...",
    )

    if st.button("Recommend movies", type="primary", disabled=not selected):
        results = recommend_multi(selected, data, similarity, genre_filter, min_votes, limit)
        if results.empty:
            st.info("No matches found. Try another movie, a different genre, or lower the minimum ratings.")
        else:
            st.subheader("Because you liked: " + ", ".join(selected))
            for _, movie in results.iterrows():
                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    left.markdown(f"**{movie['title']}**")
                    chips = "".join(f"<span class='chip'>{g}</span>" for g in movie["genres"].split("|"))
                    left.markdown(chips, unsafe_allow_html=True)
                    left.caption("Why: " + explain_match(selected, movie["title"], data))
                    right.metric("Match", f"{movie['match_pct']}%")
                    if movie["rating_count"] > 0:
                        right.caption(f"⭐ {movie['avg_rating']:.1f}/5 · {movie['rating_count']} ratings")
                    else:
                        right.caption("No ratings yet")

# ---------- Tab 2: hidden gems ----------
with tab2:
    st.subheader("Hidden gems")
    st.write(
        "Well-rated movies ranked with a weighted score, so a movie with only a few "
        "ratings can't beat one that many people rated highly."
    )
    gem_votes = st.slider("Minimum ratings", 5, 100, 10, key="gem_votes")
    gems = hidden_gems(data, gem_votes, 15, genre_filter)
    if gems.empty:
        st.info("No movies match these filters. Try lowering the minimum ratings.")
    else:
        table = gems[["title", "genres", "avg_rating", "rating_count", "gem_score"]].copy()
        table["avg_rating"] = table["avg_rating"].round(2)
        table["gem_score"] = table["gem_score"].round(2)
        st.dataframe(table, hide_index=True)

# ---------- Tab 3: insights ----------
with tab3:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", f"{len(movies):,}")
    c2.metric("Ratings", f"{len(ratings):,}")
    c3.metric("Tags", f"{len(tags):,}")

    stats = genre_stats(data)
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Movies per genre")
        st.bar_chart(stats["movies"])
    with col_b:
        st.write("Average rating per genre")
        st.bar_chart(stats["avg_rating"])

    st.write("Most-rated movies")
    popular = data.sort_values("rating_count", ascending=False).head(10)
    st.dataframe(popular[["title", "genres", "avg_rating", "rating_count"]], hide_index=True)

    st.write("Ratings distribution")
    st.bar_chart(ratings["rating"].value_counts().sort_index())