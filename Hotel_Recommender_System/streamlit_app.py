"""Streamlit collaborative-filtering hotel recommender."""
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.sparse.linalg import svds

DATA_PATH = Path(__file__).resolve().parents[1] / "hotels.csv"


@st.cache_data
def build_predictions():
    bookings = pd.read_csv(DATA_PATH)
    interactions = bookings.groupby(["userCode", "name"], as_index=False)["price"].sum()
    matrix = interactions.pivot(index="userCode", columns="name", values="price").fillna(0)
    k = min(8, min(matrix.shape) - 1)
    users, singular_values, hotels = svds(matrix.to_numpy(), k=k)
    predictions = users @ __import__("numpy").diag(singular_values) @ hotels
    return bookings, interactions, pd.DataFrame(predictions, index=matrix.index, columns=matrix.columns)


def recommend(user_code, interactions, predictions, top_n=5):
    # Every retained user has interacted with the small, fixed hotel catalogue.
    # Rank predicted preference across that catalogue rather than returning an
    # empty result after excluding all previously booked hotels.
    scores = predictions.loc[user_code]
    return scores.sort_values(ascending=False).head(top_n).rename_axis("hotel").reset_index(name="recommendation_score")


def main():
    st.set_page_config(page_title="Voyage Analytics", page_icon="✈️")
    st.title("Hotel Recommendations")
    st.caption("Collaborative filtering based on historical hotel bookings.")
    bookings, interactions, predictions = build_predictions()
    eligible_users = sorted(predictions.index.tolist())
    user_code = st.selectbox("Select a user", eligible_users)
    if st.button("Get recommendations"):
        results = recommend(user_code, interactions, predictions)
        st.dataframe(results, use_container_width=True, hide_index=True)
    st.caption(f"Training data: {len(bookings):,} hotel bookings")


if __name__ == "__main__":
    main()
