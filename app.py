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

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Player Analysis", "Compare Players", "Data Visualizations", "Project Overview", "About Me"])

if menu == "Project Overview":
    st.title("Project Overview")
    st.write("## Premier League Player Performance Predictor")
    st.write("This project is a groundbreaking analysis tool designed to evaluate player performances in the Premier League with precision and fairness. By leveraging cutting-edge statistical methods, feature engineering, and machine learning, we provide a deep dive into player contributions beyond traditional metrics.")
    
    st.write("### Key Stages of the Project:")
    st.write("#### 1. Data Collection & Cleaning:")
    st.write("- Aggregated player statistics from reliable databases.")
    st.write("- Removed inconsistencies, standardized naming conventions, and handled missing values.")
    
    st.write("#### 2. Feature Engineering:")
    st.write("- Created unique metrics such as Expected Goal Contribution, Progressive Actions, and Fair Contribution Index.")
    st.write("- Developed position-specific impact factors for more accurate player evaluation.")
    
    st.write("#### 3. Adjusting Bias in Player Rankings:")
    st.write("- Traditional per 90-minute stats often overrate players with limited playtime.")
    st.write("- Implemented logarithmic and square-root transformation methods to balance player rankings and ensure fairness.")
    
    st.write("#### 4. Engaging Data Visualization:")
    st.write("- Used Plotly and Matplotlib for dynamic, interactive visualizations.")
    st.write("- Integrated heatmaps, sunburst charts, radar graphs, and trend analytics.")
    
    st.write("#### 5. Machine Learning & Predictive Modeling:")
    st.write("- Trained models to analyze future player performances and goal involvement probabilities.")
    st.write("- Used regression and clustering to forecast statistical trends across seasons.")
    
    st.write("#### 6. Interactive Streamlit Dashboard:")
    st.write("- Designed a fully functional, user-friendly interface to explore, compare, and analyze player statistics.")
    st.write("- Added squad, positional, and statistical filtering to allow personalized analysis.")

elif menu == "About Me":
    st.title("About Me - Samyak Pokharna")
    st.write("## Data Scientist | Football Analytics Enthusiast | Engineer")
    st.write("I am Samyak Pokharna, a passionate data scientist with expertise in sports analytics, predictive modeling, and machine learning. My fascination with football statistics and data-driven decision-making led to the development of this analytical project.")
    
    st.write("### Professional Background:")
    st.write("- **Education:** B.Tech in Mechanical Engineering, currently pursuing an MS in Analytics at UIUC.")
    st.write("- **Skills:** Predictive Analytics, Machine Learning, Data Visualization, Python, SQL, Tableau.")
    st.write("- **Industry Experience:** Worked on real-world projects in supply chain optimization, product analytics, and advanced sports data modeling.")
    
    st.write("### Personal Interests:")
    st.write("- **Football:** A dedicated follower of the game, with a keen interest in tactical breakdowns and player evaluation.")
    st.write("- **Cricket & Badminton:** Enjoys playing and analyzing strategies across multiple sports.")
    st.write("- **Reading Autobiographies:** Passionate about learning from influential figures in sports, business, and leadership.")
    
    st.write("### Let's Connect!")
    st.write("📧 Email: samyak.pokharna@example.com")
    st.write("📱 LinkedIn: [linkedin.com/in/samyakpokharna](https://www.linkedin.com/in/samyakpokharna)")

elif menu == "Player Analysis":
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

elif menu == "Compare Players":
    st.subheader("Compare Player Performance")
    player_options = df["Player"].unique()
    selected_players = st.multiselect("Select Players to Compare", player_options)
    
    if selected_players:
        comparison_df = df[df["Player"].isin(selected_players)]
        st.write(comparison_df)
        
        # Dynamic visualization with different chart types
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_stat = st.selectbox("Select Statistic to Compare", numeric_columns)
        chart_type = st.radio("Choose a Chart Type", ["Bar Chart", "Line Chart", "Radar Chart"])
        
        if chart_type == "Bar Chart":
            fig = px.bar(comparison_df, x="Player", y=selected_stat, title=f"Comparison of {selected_stat}", color="Player")
        elif chart_type == "Line Chart":
            fig = px.line(comparison_df, x="Player", y=selected_stat, title=f"Trend Analysis of {selected_stat}", color="Player", markers=True)
        else:
            fig = px.scatter_polar(comparison_df, r=selected_stat, theta="Player", title=f"Radar Comparison of {selected_stat}", color="Player")
        
        st.plotly_chart(fig)
