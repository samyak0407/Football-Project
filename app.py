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
    background: url("https://github.com/samyak0407/Football-Project/blob/main/NINTCHDBPICT000733331015.webp");
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

if menu == "Abbreviations":
    st.subheader("Football Statistical Abbreviations")
    abbreviations = {
        "xG": "Expected Goals",
        "xAG": "Expected Assists",
        "npxG": "Non-Penalty Expected Goals",
        "PrgP": "Progressive Passes",
        "PrgC": "Progressive Carries",
        "PrgR": "Progressive Runs"
    }
    for abbr, meaning in abbreviations.items():
        st.write(f"- **{abbr}**: {meaning}")

if menu == "Project Overview":
    st.title("Project Overview")
    st.write("## Premier League Player Performance Predictor")
    st.write(
        "This project aims to provide a deep, unbiased analysis of player performance using advanced statistical models. "
        "By leveraging data from the Premier League, we process raw statistics, clean and transform the data, apply feature engineering, "
        "and build predictive models to assess player impact fairly."
    )
    
    st.write("### Key Aspects of the Project")
    st.write("- **Data Cleaning:** Handling missing values, normalizing nation names, and refining player positions.")
    st.write("- **Feature Engineering:** Constructing new performance metrics such as 'Fair Contribution Score' to balance impact evaluation for players with different playing times.")
    st.write("- **Visualizations & Insights:** Interactive dashboards to explore player statistics, compare performances, and analyze trends across different playing positions.")
    st.write("- **Predictive Analytics:** Implementing machine learning techniques to forecast player contributions and identify undervalued talent.")
    st.write("- **Web Scraping & Automation:** Extracting live player data to keep the analysis up-to-date and relevant.")

if menu == "About Me":
    st.title("About Me - Samyak Pokharna")
    st.write("## Data Scientist | Football Analytics Enthusiast | Engineer")
    st.write(
        "I am Samyak Pokharna, a passionate data scientist with a deep love for football and analytics. "
        "My journey into data science began with my engineering background, where I realized the power of data-driven decision-making. "
        "Over time, I transitioned into analytics, focusing on predictive modeling, machine learning, and visualization techniques."
    )
    
    st.write("### Technical Skills")
    st.write("- **Data Analytics & Visualization:** Python, SQL, Tableau, Power BI")
    st.write("- **Machine Learning & Predictive Modeling:** Regression, Classification, Time-Series Forecasting")
    st.write("- **Data Engineering:** Web Scraping, Data Cleaning, Feature Engineering")
    st.write("- **Football Analytics:** Player Performance Prediction, Tactical Data Analysis")

    st.write("### Hobbies & Interests")
    st.write("- Playing **Football, Cricket, and Badminton**")
    st.write("- Reading **Autobiographies and Analytical Books**")
    st.write("- Exploring **New Technologies in Data Science & AI**")

    st.write("### Let's Connect!")
    st.write("📧 Email: samyakp3@illinois.edu")
    st.write("📱 LinkedIn: [linkedin.com/in/samyakpokharna](https://www.linkedin.com/in/samyakpokharna)")
    st.write("📂 GitHub: [github.com/samyak0407](https://github.com/samyak0407)")
    # Custom CSS for text box to improve readability
text_box_style = """
    <style>
    .text-box {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 18px;
    }
    </style>
"""
st.markdown(text_box_style, unsafe_allow_html=True)

# Ensure that text content is in a readable box
if menu == "Project Overview":
    st.subheader("Project Overview")
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    st.write("""
    This project is a **comprehensive analysis of Premier League player performance** using real-world data. 
    We have implemented advanced **data processing, filtering, visualization, and modeling** techniques.
    
    - **Data Cleaning:** Standardized country names, removed duplicates, and ensured correct data formatting.
    - **Feature Engineering:** Created new metrics to better represent **player contributions, efficiency, and consistency**.
    - **Data Visualization:** Built interactive charts to compare performance metrics, allowing **position-based filtering**.
    - **Machine Learning:** Developed a predictive model to estimate player market value based on performance statistics.
    - **Web Deployment:** Hosted an interactive dashboard on **Streamlit**, allowing fans, analysts, and scouts to explore data dynamically.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if menu == "About Me":
    st.subheader("About Me")
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    st.write("""
    **Hey, I'm Samyak Pokharna!**  
    I have a background in **Mechanical Engineering** and an intense passion for **Data Science, Analytics, and Football**.
    
    - **Education:** MS in Analytics at **University of Illinois Urbana-Champaign**.
    - **Technical Skills:** Expertise in **data visualization, statistical modeling, SQL, Python**, and more.
    - **Football Enthusiast:** Passionate about analyzing player performances, tactics, and data-driven football insights.
    - **Other Interests:** I enjoy **reading autobiographies, playing football, cricket, badminton**, and watching tactical football breakdowns.
    
    Want to collaborate or discuss football analytics?  
    📧 **Email:** samyakp3@illinois.edu  
    🔗 **LinkedIn:** [linkedin.com/in/samyakpokharna](#)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if menu == "Abbreviations":
    st.subheader("Football Statistical Abbreviations")
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    abbreviations = {
        "xG": "Expected Goals - A measure of the quality of chances a player gets.",
        "xAG": "Expected Assists - The likelihood of a pass resulting in a goal.",
        "npxG": "Non-Penalty Expected Goals - Expected goals excluding penalties.",
        "PrgP": "Progressive Passes - Passes that move the ball significantly forward.",
        "PrgC": "Progressive Carries - Carries that move the ball forward into attacking areas.",
        "PrgR": "Progressive Runs - Runs with the ball that advance play into dangerous zones.",
        "G_per_xG": "Goals per Expected Goals - Measures finishing efficiency.",
        "Ast_per_xAG": "Assists per Expected Assists - Evaluates the quality of key passes.",
    }
    for abbr, meaning in abbreviations.items():
        st.write(f"- **{abbr}**: {meaning}")
    st.markdown('</div>', unsafe_allow_html=True)

