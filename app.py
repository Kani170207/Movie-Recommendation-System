import streamlit as st
import pandas as pd
import ast
import requests
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Session state
# -------------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

OMDB_API_KEY = "90e088c4"

# -------------------------------
# Fetch poster from OMDb
# -------------------------------
def fetch_poster(title):
    try:
        safe_title = urllib.parse.quote(title)
        url = f"http://www.omdbapi.com/?t={safe_title}&apikey={OMDB_API_KEY}"
        res = requests.get(url, timeout=5).json()
        if res.get("Poster") and res["Poster"] != "N/A":
            return res["Poster"]
    except:
        pass
    return None

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
}
h1, h2, h3 { color: #facc15; }
.stButton > button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 30px;
    font-size: 16px;
    padding: 8px 25px;
}
.stButton > button:hover {
    transform: scale(1.05);
}
img { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")
st.write("Content-based movie recommender using NLP (TF-IDF + Cosine Similarity)")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview', 'genres', 'vote_average', 'release_date']]
    df.dropna(inplace=True)

    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year
    df['combined_features'] = df['overview'] + " " + df['genres'].apply(lambda x: " ".join(x))

    return df.reset_index(drop=True)

df = load_data()

all_genres = sorted({g for sub in df['genres'] for g in sub})

# -------------------------------
# Build similarity
# -------------------------------
@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data['combined_features'])
    return cosine_similarity(matrix)

cosine_sim = build_model(df)

# -------------------------------
# Genre-only recommendation
# -------------------------------
def recommend_by_genre_only(genre, n):
    movies = df[df['genres'].apply(lambda g: genre in g)]
    return movies.sort_values(by="vote_average", ascending=False).head(n)

# -------------------------------
# UI controls
# -------------------------------
selected_genre = st.selectbox("🎭 Choose a genre", all_genres)
num_recs = st.slider("🔢 Number of recommendations", 5, 15, 5)

# -------------------------------
# Recommend Movies
# -------------------------------
if st.button("🚀 Recommend Movies", key="recommend_btn"):

    results = recommend_by_genre_only(selected_genre, num_recs)

    st.subheader("🎯 Recommended Movies")

    for _, row in results.iterrows():
        col1, col2, col3 = st.columns([1, 3, 1])

        # Poster
        with col1:
            poster = fetch_poster(row['title'])
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.text("No poster")

        # Details
        with col2:
            st.markdown(f"""
            ### 🎬 {row['title']}
            ⭐ **Rating:** {row['vote_average']}
            📅 **Year:** {row['release_year']}
            🎭 **Genres:** {', '.join(row['genres'])}
            """)

        # Watchlist
        with col3:
            add_key = f"add_{row['title']}"
            if st.button("➕ Add", key=add_key):
                if row['title'] not in [m['title'] for m in st.session_state.watchlist]:
                    st.session_state.watchlist.append(row.to_dict())
                    st.success("Added 🎉")
                else:
                    st.warning("Already added")

        st.divider()

# -------------------------------
# Watchlist section
# -------------------------------
st.subheader("📌 My Watchlist")

if not st.session_state.watchlist:
    st.info("Your watchlist is empty 🎬")
else:
    for i, movie in enumerate(st.session_state.watchlist):
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            poster = fetch_poster(movie['title'])
            if poster:
                st.image(poster, use_container_width=True)

        with col2:
            st.markdown(f"""
            ### 🎬 {movie['title']}
            ⭐ **Rating:** {movie['vote_average']}
            📅 **Year:** {movie['release_year']}
            🎭 **Genres:** {', '.join(movie['genres'])}
            """)

        with col3:
            if st.button("❌ Remove", key=f"remove_{i}"):
                st.session_state.watchlist.pop(i)
                st.experimental_rerun()


