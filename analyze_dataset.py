import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
import os

@st.cache_data
def load_data():
    if os.path.exists("UpdatedResumeDataSet_Encoded.csv"):
        return pd.read_csv("UpdatedResumeDataSet_Encoded.csv")
    return None

def show_visualizations():
    df = load_data()
    if df is None:
        st.error("Dataset not found.")
        return

    st.subheader("📊 Category Distribution")
    counts = df["Category"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    st.pyplot(fig)

    st.subheader("☁️ Resume Word Cloud")
    text = " ".join(df["Resume"].astype(str))
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)
