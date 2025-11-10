"""
recommender.py
---------------
Content-based recommender system for the Netflix dataset
using TF-IDF vectorization and cosine similarity.

Usage Example:
    from recommender import build_recommender, recommend

    df_clean, cosine_sim, indices = build_recommender(df)
    print(recommend("Money Heist", df_clean, cosine_sim, indices))
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

try:
    import streamlit as st
except ImportError:
    # fallback if Streamlit isn't installed (for notebook or script use)
    class st:
        @staticmethod
        def info(msg): print(msg)
        @staticmethod
        def error(msg): print(msg)


def build_recommender(df):
    """
    Build a content-based recommender using TF-IDF on combined text features.

    Args:
        df (pd.DataFrame): Netflix dataset with columns:
            ['title', 'listed_in', 'description', 'country', 'director']

    Returns:
        tuple: (cleaned dataframe, cosine similarity matrix, title indices)
    """
    df = df.copy()

    # Clean and normalize
    df['title'] = df['title'].astype(str).str.strip().str.lower()
    df['listed_in'] = df['listed_in'].fillna('')
    df['description'] = df['description'].fillna('')
    df['country'] = df.get('country', '').fillna('')
    df['director'] = df.get('director', '').fillna('')

    # Combine key text features into a single string
    df['combined_features'] = (
        df['listed_in'] + " " +
        df['country'] + " " +
        df['director'] + " " +
        df['description']
    )

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Create title → index mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return df, cosine_sim, indices


def recommend(title, df, cosine_sim, indices, n=3):
    """
    Recommend N similar titles based on content similarity.

    Args:
        title (str): Title to base recommendations on.
        df (pd.DataFrame): Cleaned dataframe.
        cosine_sim (ndarray): Cosine similarity matrix.
        indices (pd.Series): Mapping from title → index.
        n (int): Number of recommendations to return (default=5).

    Returns:
        list: List of recommended titles (capitalized).
    """
    title = title.strip().lower()

    # Handle fuzzy title matching
    if title not in indices:
        close_matches = get_close_matches(title, indices.index, n=1, cutoff=0.6)
        if close_matches:
            suggestion = close_matches[0]
            st.info(f"🔍 Did you mean **{suggestion.title()}**?")
            title = suggestion
        else:
            st.error("❌ No such title found. Try another one.")
            return []

    # Get index of the given title
    idx = indices[title]

    # Compute similarity scores for the title
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n + 1]

    # Get top N recommendations
    recommendations = df['title'].iloc[[i[0] for i in sim_scores]]

    # Capitalize for cleaner display
    return [t.title() for t in recommendations.tolist()]
