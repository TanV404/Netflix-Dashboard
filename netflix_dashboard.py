import streamlit as st
import pandas as pd
from textblob import TextBlob
from src.sentiment_analysis import add_sentiment
from src.recommender import build_recommender, recommend

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Netflix Dashboard",
    layout="wide",
)

st.title("🎬 Netflix Dashboard")
st.caption("Explore, analyze, and get recommendations using data-driven insights.")

# --- Load and Process Data ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/TanV404/Netflix-Dashboard/main/data/netflix_titles.csv"
    df = pd.read_csv(url, quotechar='"', on_bad_lines='skip', encoding='utf-8')
    df = add_sentiment(df)
    df_clean, cosine_sim, indices = build_recommender(df)
    return df, df_clean, cosine_sim, indices

df, df_clean, cosine_sim, indices = load_data()

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["📈 Overview", "💬 Sentiment Analysis", "🎯 Recommendations"])

# ======================
# Tab 1: Overview
# ======================
with tab1:

    st.subheader("📊 Dataset Overview")

    # Sidebar Filters (moved inside tab 1)
    st.write("🎛️ Filter Options")

    def split_unique_values(series):
        """Split comma-separated values and return sorted unique cleaned list."""
        all_items = []
        for val in series.dropna():
            all_items.extend([
                v.strip() for v in val.split(",")
                if v.strip() and v.strip().lower() != "unknown"
            ])
        return sorted(set(all_items))

    # Create clean unique lists
    type_options = sorted(df["type"].dropna().unique())
    country_options = split_unique_values(df["country"])
    genre_options = split_unique_values(df["listed_in"])

    # Streamlit filter widgets
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_type = st.multiselect("Select Type", options=type_options)
    with col2:
        selected_country = st.multiselect("Select Country", options=country_options)
    with col3:
        selected_genre = st.multiselect("Select Genre", options=genre_options)

    # Apply filters
    filtered_df = df.copy()
    if selected_type:
        filtered_df = filtered_df[filtered_df["type"].isin(selected_type)]
    if selected_country:
        filtered_df = filtered_df[
            filtered_df["country"].apply(
                lambda x: any(c in str(x) for c in selected_country)
            )
        ]
    if selected_genre:
        filtered_df = filtered_df[
            filtered_df["listed_in"].apply(
                lambda x: any(g in str(x) for g in selected_genre)
            )
        ]

    # --- Charts ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎞️ Titles by Type")
        st.bar_chart(filtered_df["type"].value_counts())

    with col2:
        st.write("🌍 Titles by Country")
        country_counts = filtered_df["country"].value_counts().head(10)
        st.bar_chart(country_counts)

# ======================
# Tab 2: Sentiment Analysis
# ======================
with tab2:
    st.subheader("💬 Sentiment Analysis on Descriptions")

    # Use filtered data from Tab 1
    filtered_df["sentiment"] = filtered_df["sentiment"].astype(str).str.lower()

    sentiment_counts = filtered_df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("😀 Positive Samples")
        positive_samples = filtered_df[filtered_df["sentiment"] == "positive"][["title", "description"]].head(5)
        if len(positive_samples) == 0:
            st.info("No positive samples found.")
        else:
            st.dataframe(positive_samples, hide_index=True)

    with col2:
        st.subheader("😔 Negative Samples")
        negative_samples = filtered_df[filtered_df["sentiment"] == "negative"][["title", "description"]].head(5)
        if len(negative_samples) == 0:
            st.info("No negative samples found.")
        else:
            st.dataframe(negative_samples, hide_index=True)

    # Sentiment Prediction Box
    st.subheader("🧠 Try Your Own Description")
    user_input = st.text_input("Enter a movie/show description to analyze sentiment:")
    if user_input.strip():
        polarity = TextBlob(user_input).sentiment.polarity
        sentiment = 'Positive' if polarity > 0.1 else ('Negative' if polarity < -0.1 else 'Neutral')
        st.success(f"**Predicted Sentiment:** {sentiment}")

# ======================
# Tab 3: Recommendation
# ======================
with tab3:
    st.subheader("🎯 Content Recommendation")

    user_input = st.text_input("Enter a title to get similar recommendations:")
    if user_input:
        recs = recommend(user_input, df_clean, cosine_sim, indices)
        if recs:
            st.success(f"Top {len(recs)} recommendations for **{user_input.title()}**:")
            for r in recs:
                show = df_clean[df_clean["title"] == r.lower()].iloc[0]
                with st.expander(f"🎥 {r}"):
                    st.write(f"**Genre:** {show['listed_in']}")
                    st.write(f"**Description:** {show['description']}")
        else:
            st.warning("❌ No similar titles found. Try another one.")
    else:
        st.info("💡 Type a show or movie title above to get recommendations!")

# --- Footer ---
st.markdown("---")
st.caption("Netflix Dashboard • Built with ❤️ using Streamlit")
