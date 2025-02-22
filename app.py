import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px

# Ensure dependencies are installed
os.system("pip install matplotlib seaborn numpy pandas plotly")

# Set page configuration with a blurred background image for better readability
st.set_page_config(
    page_title="Premier League Player Performance Predictor",
    layout="wide"
)

# Custom CSS for background image with blur effect on background only
page_bg_img = '''
<style>
.stApp {
    background: url("https://raw.githubusercontent.com/samyak0407/Football-Project/main/NINTCHDBPICT000733331015.webp");
    background-size: cover;
    background-position: center;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: inherit;
    filter: blur(5px);
    z-index: -1;
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
    
    # Clean DOB column (remove commas)
    df["Born"] = df["Born"].astype(str).str.replace(",", "")
    
    # Standardize nation names
    country_mapping = {
        "FRA": "France", "FR": "France", "ENG": "England", "ESP": "Spain", "GER": "Germany", "ITA": "Italy",
        "BRA": "Brazil", "ARG": "Argentina", "POR": "Portugal", "BEL": "Belgium", "NED": "Netherlands",
        "USA": "United States", "MEX": "Mexico", "COL": "Colombia", "URU": "Uruguay", "SEN": "Senegal",
        "CIV": "Ivory Coast", "NGA": "Nigeria", "MAR": "Morocco", "GHA": "Ghana", "CHL": "Chile",
        "POL": "Poland", "DEN": "Denmark", "SWE": "Sweden", "NOR": "Norway", "CRO": "Croatia",
        "SRB": "Serbia", "SUI": "Switzerland", "AUT": "Austria", "JPN": "Japan", "KOR": "South Korea"
    }
    df["Nation"] = df["Nation"].map(lambda x: country_mapping.get(x, x))
    
    # Extract only the primary position
    df["Pos"] = df["Pos"].apply(lambda x: x.split(",")[0])
    
    # Rename columns for better readability
    column_name_mapping = {
        "MP": "Matches Played",
        "Min": "Minutes Played",
        "Gls": "Goals",
        "Ast": "Assists",
        "G+A": "Goals + Assists",
        "G-PK": "Goals (Non-Penalty)",
        "PK": "Penalties Scored",
        "PKatt": "Penalties Attempted",
        "CrdY": "Yellow Cards",
        "CrdR": "Red Cards",
        "xG": "Expected Goals",
        "npxG": "Non-Penalty Expected Goals",
        "xAG": "Expected Assists",
        "npxG+xAG": "Non-Penalty xG + xA",
        "PrgC": "Progressive Carries",
        "PrgP": "Progressive Passes",
        "PrgR": "Progressive Runs",
        "G+A-PK": "Goals + Assists (Non-Penalty)",
        "xG+xAG": "Expected Goals + Expected Assists",
        "G_per_xG": "Goals per Expected Goals",
        "Ast_per_xAG": "Assists per Expected Assists",
        "Progressive_Actions_90": "Progressive Actions per 90 Minutes"
    }
    df.rename(columns=column_name_mapping, inplace=True)
    
    # Round all numeric values to 2 decimal places
    df = df.round(2)
    return df

df = load_data()

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Player Analysis", "Compare Players", "Data Visualizations", "Project Overview", "About Me", "Abbreviations"])

if menu == "Player Analysis":
    st.subheader("Player Performance Analysis")
    st.dataframe(df)  # Ensure the table displays correctly

if menu == "Data Visualizations":
    st.subheader("Top Performers by Category")
    min_minutes = st.slider("Minimum Minutes Played", min_value=0, max_value=int(df["Minutes Played"].max()), value=1000)
    position_filter = st.selectbox("Select Position", ["All"] + df["Pos"].unique().tolist())
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = ["Player", "Nation", "Pos", "Squad"]
    numeric_columns = [col for col in numeric_columns if col not in excluded_columns]
    
    selected_metric = st.selectbox("Select Metric to View Top Players", numeric_columns)
    
    if selected_metric:
        filtered_df = df[df["Minutes Played"] >= min_minutes]
        if position_filter != "All":
            filtered_df = filtered_df[filtered_df["Pos"] == position_filter]
        top_players = filtered_df.nlargest(10, selected_metric)
        fig = px.bar(top_players, x=selected_metric, y="Player", orientation='h',
                     title=f"Top 10 {position_filter} Players by {selected_metric} (Min. {min_minutes} Minutes)", color="Player")
        st.plotly_chart(fig)

if menu == "Compare Players":
    st.subheader("Compare Player Performance")
    player_options = df["Player"].unique()
    selected_players = st.multiselect("Select Players to Compare", player_options)
    
    if selected_players:
        comparison_df = df[df["Player"].isin(selected_players)]
        st.dataframe(comparison_df)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_stat = st.selectbox("Select Statistic to Compare", numeric_columns)
        fig = px.bar(comparison_df, x="Player", y=selected_stat, title=f"Comparison of {selected_stat}", color="Player")
        st.plotly_chart(fig)


# Function to display content inside a styled box
def display_text_box(title, content):
    st.markdown(f'<div class="text-box"><h1>{title}</h1>{content}</div>', unsafe_allow_html=True)

# Project Overview Section
if menu == "Project Overview":
    display_text_box(
        "PROJECT OVERVIEW",
        """
        <p><b>Premier League Player Performance Predictor</b></p>
        <ul>
        <li><b>Data Cleaning:</b> Handling missing values, normalizing nation names, and refining player positions.</li>
        <li><b>Feature Engineering:</b> Constructing performance metrics such as 'Fair Contribution Score' to balance evaluations.</li>
        <li><b>Visualizations & Insights:</b> Interactive dashboards to explore player statistics and analyze trends.</li>
        <li><b>Predictive Analytics:</b> Implementing machine learning models to forecast player performance.</li>
        <li><b>Web Scraping & Automation:</b> Extracting live player data for up-to-date analysis.</li>
        </ul>
        """
    )

# About Me Section
if menu == "About Me":
    display_text_box(
        "ABOUT ME - SAMYAK POKHARNA",
        """
        I am Samyak Pokharna, a data scientist passionate about football analytics. With an engineering background, I transitioned into AI-driven analytics, focusing on predictive modeling, machine learning, and visualization techniques.

        Technical Skills:
        Data Analytics & Visualization (Python, SQL, Tableau, Power BI), 
        Machine Learning & Predictive Modeling (Regression, Classification, Time-Series Forecasting), 
        Data Engineering (Web Scraping, Data Cleaning, Feature Engineering), 
        Football Analytics (Player Performance Prediction, Tactical Data Analysis).

        Hobbies & Interests:
        Playing Football, Cricket, and Badminton, 
        Reading Autobiographies and Analytical Books, 
        Exploring New Technologies in Data Science & AI.

        Let's Connect:
        Email: samyakp3@illinois.edu, 
        LinkedIn: [linkedin.com/in/samyakpokharna](https://www.linkedin.com/in/samyakpokharna), 
        GitHub: [github.com/samyak0407](https://github.com/samyak0407).
        """
    )


# Abbreviations Section
if menu == "Abbreviations":
    display_text_box(
        "FOOTBALL STATISTICAL ABBREVIATIONS",
        """
        <ul>
        <li><b>xG:</b> Expected Goals</li>
        <li><b>xAG:</b> Expected Assists</li>
        <li><b>npxG:</b> Non-Penalty Expected Goals</li>
        <li><b>PrgP:</b> Progressive Passes</li>
        <li><b>PrgC:</b> Progressive Carries</li>
        <li><b>PrgR:</b> Progressive Runs</li>
        </ul>
        """
    )
