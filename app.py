import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px

# Ensure dependencies are installed
os.system("pip install matplotlib seaborn numpy pandas plotly")

# Set page configuration with a background image and improved readability
st.set_page_config(
    page_title="Premier League Player Performance Predictor",
    layout="wide"
)

# Custom CSS for background image with improved text readability
page_bg_img = '''
<style>
.stApp {
    background: url("https://raw.githubusercontent.com/samyak0407/Football-Project/main/GettyImages-2184014913.webp");
    background-size: cover;
    background-position: center;
    color: white;
    text-shadow: 1px 1px 2px black;
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

# Identify correct column name for Minutes
minutes_col = "Min"  # Directly assigning since we confirmed the column name

# Adjust Player Contribution Based on Playtime Using Weighted Normalization
if minutes_col in df.columns and "Goal Contribution" in df.columns:
    df["Fair Contribution"] = (df["Goal Contribution"] * (df[minutes_col] / df[minutes_col].max())) * (1 - np.exp(-df[minutes_col] / 1500))
    df["Fair Contribution"] = df["Fair Contribution"].round(2)

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Player Analysis", "Compare Players", "Data Visualizations", "Project Overview", "About Me"])

if menu == "Player Analysis":
    st.subheader("Player Performance Data")
    squads = df["Squad"].unique()
    selected_squad = st.selectbox("Select a Squad", ["All"] + list(squads))
    
    if selected_squad != "All":
        df_filtered = df[df["Squad"] == selected_squad]
    else:
        df_filtered = df
    
    st.write(df_filtered)

elif menu == "Compare Players":
    st.subheader("Compare Player Performance")
    player_options = df["Player"].unique()
    selected_players = st.multiselect("Select Players to Compare", player_options)
    
    if selected_players:
        comparison_df = df[df["Player"].isin(selected_players)]
        st.write(comparison_df)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_stat = st.selectbox("Select Statistic to Compare", numeric_columns)
        
        fig = px.bar(comparison_df, x="Player", y=selected_stat, title=f"Comparison of {selected_stat}", color="Player")
        st.plotly_chart(fig)

elif menu == "Data Visualizations":
    st.subheader("Top Performers by Category (Min. 1000 Minutes)")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = ["Player", "Nation", "Pos", "Squad"]
    numeric_columns = [col for col in numeric_columns if col not in excluded_columns]

    selected_metric = st.selectbox("Select Metric to View Top Players", numeric_columns)
    selected_position = st.selectbox("Filter by Position", ["All", "FW", "MF", "DF"])

    if selected_metric:
        filtered_df = df[df[minutes_col] >= 1000]
        if selected_position != "All":
            filtered_df = filtered_df[filtered_df["Pos"] == selected_position]
        
        top_players = filtered_df.nlargest(10, selected_metric)
        fig = px.bar(top_players, x=selected_metric, y="Player", orientation='h',
                     title=f"Top 10 {selected_position if selected_position != 'All' else ''} Players by {selected_metric} (Min. 1000 Minutes)", color="Player")
        st.plotly_chart(fig)

elif menu == "Project Overview":
    st.title("Project Overview")
    st.write("## Premier League Player Performance Predictor")
    st.write("This project is a comprehensive analytical tool designed to evaluate player performances in the Premier League fairly and precisely. We leverage advanced statistics, feature engineering, and machine learning to provide deep insights into player contributions beyond traditional metrics.")
    
    st.write("### Key Stages of the Project:")
    st.write("#### 1. Data Collection & Cleaning:")
    st.write("- Aggregated player statistics from trusted sources.")
    st.write("- Removed inconsistencies, standardized naming conventions, and handled missing values.")
    
    st.write("#### 2. Feature Engineering:")
    st.write("- Developed key metrics like Expected Goal Contribution, Progressive Actions, and Fair Contribution Index.")
    st.write("- Introduced positional impact factors for better player evaluation.")
    
    st.write("#### 3. Machine Learning & Predictive Modeling:")
    st.write("- Built predictive models to analyze future player performances and goal involvement probabilities.")
    st.write("- Used regression and clustering techniques to forecast player trends.")
    
    st.write("#### 4. Dynamic Dashboard:")
    st.write("- Created an interactive interface that allows users to analyze, compare, and explore player statistics.")
    st.write("- Integrated squad and positional filtering for targeted insights.")

elif menu == "About Me":
    st.title("About Me - Samyak Pokharna")
    st.write("## Data Scientist | Football Analytics Enthusiast | Engineer")
    st.write("I am Samyak Pokharna, a passionate data scientist specializing in sports analytics, predictive modeling, and machine learning. My love for football statistics and data-driven decision-making inspired me to develop this project.")
    
    st.write("### Background:")
    st.write("- **Education:** B.Tech in Mechanical Engineering, currently pursuing an MS in Analytics Statistics at UIUC.")
    st.write("- **Skills:** Predictive Analytics, Machine Learning, Data Visualization, Python, SQL, Tableau.")
    st.write("- **Experience:** Worked on projects in supply chain optimization, product analytics, and advanced sports data modeling.")
    
    st.write("### Interests:")
    st.write("- **Football Enthusiast:** Loves analyzing player performances and tactical strategies.")
    st.write("- **Cricket & Badminton:** Enjoys playing and studying the nuances of different sports.")
    st.write("- **Reading Autobiographies:** Passionate about learning from influential figures in sports and business.")
    
    st.write("### Let's Connect!")
    st.write("Email: samyakp3@illinois.edu")
    st.write("LinkedIn: [linkedin.com/in/samyakpokharna](https://www.linkedin.com/in/samyakpokharna)")
