import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration with a background image
st.set_page_config(
    page_title="Premier League Player Performance Predictor",
    layout="wide"
)

# Custom CSS for background image
page_bg_img = '''
<style>
.stApp {
    background: url("file:///Users/sam/Downloads/GettyImages-2184014913.webp");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Set title
st.title("Premier League Player Performance Predictor")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Processed_Premier_League_Dataset.csv")
    df.index += 1  # Adjust index so it starts from 1 instead of 0
    
    # Rename columns for better readability
    df.rename(columns=lambda x: x.replace("_", " ").title(), inplace=True)
    
    # Round all numeric values to 2 decimal places
    df = df.round(2)
    return df

df = load_data()

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Player Analysis", "Data Visualizations"])

if menu == "Player Analysis":
    st.subheader("Player Performance Data")
    squads = df["Squad"].unique()
    selected_squad = st.selectbox("Select a Squad", ["All"] + list(squads))

    if selected_squad != "All":
        df_filtered = df[df["Squad"] == selected_squad]
    else:
        df_filtered = df

    # Position-based Filtering Using "Pos" Column
    positions = ["All"] + df["Pos"].unique().tolist()
    selected_pos = st.sidebar.selectbox("Select Position", positions)

    if selected_pos != "All":
        df_filtered = df_filtered[df_filtered["Pos"] == selected_pos]
    
    st.write(df_filtered)

elif menu == "Data Visualizations":
    available_columns = df.columns.tolist()
    chart_selection = st.sidebar.radio("Select a Chart", ["None"] + available_columns)

    if chart_selection != "None":
        st.title(f"Top Players Based on {chart_selection}")
        top_players = df.nlargest(10, chart_selection)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_players, x=chart_selection, y="Player", ax=ax, palette="coolwarm")
        ax.set_title(f"Top 10 Players by {chart_selection}")
        st.pyplot(fig)

# Adjust Player Contribution Based on Playtime Using Log Scaling
if "Minutes" in df.columns and "Goal Contribution" in df.columns:
    df["Fair Contribution"] = df["Goal Contribution"] / np.log1p(df["Minutes"])  # Log-based normalization
    df = df.round(2)  # Ensure rounding is consistent
