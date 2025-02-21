import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure dependencies are installed
os.system("pip install matplotlib seaborn numpy pandas")

# Set page configuration with a background image
st.set_page_config(
    page_title="Premier League Player Performance Predictor",
    layout="wide"
)

# Custom CSS for background image
page_bg_img = '''
<style>
.stApp {
    background: url("https://raw.githubusercontent.com/samyak0407/Football-Project/main/GettyImages-2184014913.webp");
    background-size: cover;
    background-position: center;
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
menu = st.sidebar.radio("Navigation", ["Player Analysis", "Data Visualizations", "Project Overview", "About Me"])

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
    # Only allow numeric columns for visualization
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = ["Player", "Nation", "Pos", "Squad"]
    numeric_columns = [col for col in numeric_columns if col not in excluded_columns]
    chart_selection = st.sidebar.radio("Select a Chart", ["None"] + numeric_columns)

    if chart_selection != "None":
        st.title(f"Top Players Based on {chart_selection}")
        top_players = df.nlargest(10, chart_selection)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_players, x=chart_selection, y="Player", ax=ax, palette="coolwarm")
        ax.set_title(f"Top 10 Players by {chart_selection}")
        st.pyplot(fig)

elif menu == "Project Overview":
    st.title("Project Overview")
    st.write("This project analyzes Premier League player performances using advanced data metrics.")
    st.write("### Key Features:")
    st.write("- Data filtering based on teams and positions")
    st.write("- Advanced visualizations of player statistics")
    st.write("- Fair Contribution metric that balances playtime and performance")
    st.write("- Custom-built dashboard for intuitive data exploration")

elif menu == "About Me":
    st.title("About Me")
    st.write("Hello! I'm Samyak Pokharna, a passionate data enthusiast with a background in Mechanical Engineering and Data Science.")
    st.write("I enjoy working with data analytics, predictive modeling, and football-related analytics projects.")
    st.write("This project was developed as part of my deep interest in sports analytics and machine learning.")

# Adjust Player Contribution Based on Playtime Using Log Scaling
if "Minutes" in df.columns and "Goal Contribution" in df.columns:
    df["Fair Contribution"] = df["Goal Contribution"] / (np.log1p(df["Minutes"]) + 1)  # Improved Log-based normalization
    df["Fair Contribution"] = df["Fair Contribution"].round(2)  # Ensure rounding

