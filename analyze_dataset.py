import pandas as pd
import matplotlib.pyplot as plt
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

    counts = df["Category"].value_counts()

    fig, ax = plt.subplots()
    ax.barh(counts.index, counts.values)
    st.pyplot(fig)

    text = " ".join(df["Resume"].astype(str))
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)
