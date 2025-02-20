import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set title
st.title("⚽ Premier League Player Performance Predictor")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Processed_Premier_League_Dataset.csv")

df = load_data()

# Feature Engineering: Add New Metrics for Defenders & Midfielders
df["Midfield_Control"] = df["Touches"] + df["Cmp"] + df["PrgP"]
df["Pressing_Effectiveness"] = df["Press"] / (df["Press"] + df["Tkl"] + df["Int"] + 1)

df["Defensive_Impact"] = df["Tkl"] + df["Int"] + df["Blocks"] + df["Clr"]
df["Aerial_Dominance"] = df["AerWon"] / (df["AerWon"] + df["AerLost"] + 1)

# Sidebar Navigation
menu = st.sidebar.radio("📌 Navigation", ["🏆 Player Analysis", "📊 Data Visualizations"])

if menu == "🏆 Player Analysis":
    st.subheader("📊 Player Performance Data")
    squads = df["Squad"].unique()
    selected_squad = st.selectbox("Select a Squad", ["All"] + list(squads))

    if selected_squad != "All":
        df_filtered = df[df["Squad"] == selected_squad]
    else:
        df_filtered = df

    # Position-based Filtering
    positions = ["All", "Defender", "Midfielder", "Forward"]
    selected_pos = st.sidebar.selectbox("Select Position", positions)

    if selected_pos == "Defender":
        df_filtered = df_filtered.nlargest(10, "Defensive_Impact")[["Player", "Squad", "Tkl", "Int", "Blocks", "Clr", "Defensive_Impact"]]
        st.subheader("🛡️ Top Defenders Based on Defensive Impact")
    elif selected_pos == "Midfielder":
        df_filtered = df_filtered.nlargest(10, "Midfield_Control")[["Player", "Squad", "Touches", "Cmp", "PrgP", "Press", "Midfield_Control"]]
        st.subheader("🎯 Top Midfielders Based on Ball Control")
    
    st.write(df_filtered)

elif menu == "📊 Data Visualizations":
    chart_selection = st.sidebar.radio("📊 Select a Chart", ["None", "Defensive Impact", "Midfield Control"])

    if chart_selection == "Defensive Impact":
        st.title("📊 Top Defenders – Defensive Impact Score")
        top_defenders = df.nlargest(10, "Defensive_Impact")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_defenders, x="Defensive_Impact", y="Player", ax=ax, palette="Blues")
        ax.set_title("Top 10 Defenders (Defensive Impact)")
        st.pyplot(fig)

    elif chart_selection == "Midfield Control":
        st.title("🎯 Top Midfielders – Midfield Control Score")
        top_midfielders = df.nlargest(10, "Midfield_Control")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_midfielders, x="Midfield_Control", y="Player", ax=ax, palette="Greens")
        ax.set_title("Top 10 Midfielders (Midfield Control)")
        st.pyplot(fig)
