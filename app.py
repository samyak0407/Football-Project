import streamlit as st
import pandas as pd

# Set title
st.title("⚽ Premier League Player Performance Predictor")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Processed_Premier_League_Dataset.csv")

df = load_data()

# Display dataset
st.subheader("📊 Player Performance Data")
st.write(df)

# Allow users to filter by squad
squads = df["Squad"].unique()
selected_squad = st.selectbox("Select a Squad", ["All"] + list(squads))

# Filter by squad
if selected_squad != "All":
    filtered_df = df[df["Squad"] == selected_squad]
    st.write(filtered_df)
else:
    st.write(df)

st.write("🔍 Use filters to explore player stats!")


