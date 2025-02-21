import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px

# Ensure dependencies are installed
os.system("pip install matplotlib seaborn numpy pandas plotly")

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
menu = st.sidebar.radio("Navigation", ["Player Analysis", "Compare Players", "Data Visualizations", "Project Overview", "About Me"])

if menu == "Project Overview":
    st.title("Project Overview")
    st.write("## Premier League Player Performance Predictor")
    st.write("This project is a cutting-edge analytical tool designed to revolutionize player evaluation in the Premier League. By leveraging advanced statistics, feature engineering, and interactive visualization, we provide a fair and insightful analysis of player performances across different positions.")
    
    st.write("### Key Stages of the Project:")
    st.write("#### 1. Data Collection & Cleaning:")
    st.write("- Gathered player statistics from trusted sources.")
    st.write("- Removed inconsistencies, standardized names, and handled missing values.")
    
    st.write("#### 2. Feature Engineering:")
    st.write("- Developed metrics like Expected Goal Contribution, Progressive Actions, and Fair Contribution Index.")
    st.write("- Introduced new performance indicators that give deeper insights into player impact.")
    
    st.write("#### 3. Addressing Bias in Player Contribution Metrics:")
    st.write("- Ensured fairness by implementing a logarithmic and square-root adjustment for minutes played.")
    st.write("- Prevented inflation of rankings for players with limited playtime while recognizing consistent top performers.")
    
    st.write("#### 4. Data Visualization & Comparative Analysis:")
    st.write("- Designed engaging and interactive visualizations using Matplotlib and Plotly.")
    st.write("- Developed charts, heatmaps, and comparative analytics to assess team and player strengths.")
    
    st.write("#### 5. Machine Learning Model Implementation:")
    st.write("- Built predictive models to analyze future player performances based on historical data.")
    st.write("- Leveraged regression and classification techniques for goal involvement and assist probabilities.")
    
    st.write("#### 6. Interactive Streamlit Dashboard:")
    st.write("- Created a dynamic interface that allows users to analyze, compare, and explore player data in an engaging manner.")
    st.write("- Included squad and positional filtering to enable targeted analysis.")

elif menu == "About Me":
    st.title("About Me - Samyak Pokharna")
    st.write("## Data Science Enthusiast | Football Analytics Expert | Engineer")
    st.write("I am Samyak Pokharna, a passionate data scientist with a Mechanical Engineering background and a strong interest in sports analytics. My love for football and statistical analysis has driven me to work on this project, combining my expertise in data modeling with my deep appreciation for the game.")
    
    st.write("### Professional Background:")
    st.write("- **Education:** B.Tech in Mechanical Engineering, currently pursuing an MS in Analytics at UIUC.")
    st.write("- **Expertise:** Predictive Analytics, Machine Learning, Data Visualization, Python, SQL, Tableau.")
    st.write("- **Experience:** Worked on real-world projects in supply chain optimization, product analytics, and football data modeling.")
    
    st.write("### Hobbies & Interests:")
    st.write("- **Football Enthusiast:** Dedicated to analyzing player performances and tactical strategies.")
    st.write("- **Cricket & Badminton:** Loves playing and studying the nuances of different sports.")
    st.write("- **Reading Autobiographies:** Passionate about learning from influential leaders and athletes.")
    
    st.write("### Connect with Me:")
    st.write("📧 Email: samyak.pokharna@example.com")
    st.write("📱 LinkedIn: [linkedin.com/in/samyakpokharna](https://www.linkedin.com/in/samyakpokharna)")

elif menu == "Compare Players":
    st.subheader("Compare Player Performance")
    player_options = df["Player"].unique()
    selected_players = st.multiselect("Select Players to Compare", player_options)
    
    if selected_players:
        comparison_df = df[df["Player"].isin(selected_players)]
        st.write(comparison_df)
        
        # Visual comparison
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_stat = st.selectbox("Select Statistic to Compare", numeric_columns)
        
        fig = px.line(comparison_df, x="Player", y=selected_stat, title=f"Comparison of {selected_stat}", color="Player", markers=True)
        st.plotly_chart(fig)

elif menu == "Data Visualizations":
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = ["Player", "Nation", "Pos", "Squad"]
    numeric_columns = [col for col in numeric_columns if col not in excluded_columns]
    chart_selection = st.sidebar.radio("Select a Chart", ["None"] + numeric_columns)

    if chart_selection != "None":
        st.title(f"Top Players Based on {chart_selection}")
        top_players = df.nlargest(10, chart_selection)

        fig = px.sunburst(top_players, path=["Squad", "Player"], values=chart_selection, title=f"Top 10 Players by {chart_selection}")
        st.plotly_chart(fig)

# Adjust Player Contribution Based on Playtime Using Log Scaling
if "Minutes" in df.columns and "Goal Contribution" in df.columns:
    df["Fair Contribution"] = (df["Goal Contribution"] * np.sqrt(df["Minutes"])) / (np.log1p(df["Minutes"]) + 1)  # Improved fairness in ranking
    df["Fair Contribution"] = df["Fair Contribution"].round(2)  # Ensure rounding
