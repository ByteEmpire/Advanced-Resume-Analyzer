import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
import os


def show_visualizations():
    file_path = "UpdatedResumeDataSet_Encoded.csv"

    if not os.path.exists(file_path):
        st.error("Dataset file not found.")
        return

    df = pd.read_csv(file_path)

    st.subheader("📊 Resume Category Distribution")
    counts = df['Category'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("Category")
    st.pyplot(fig)

    st.subheader("☁️ Resume Word Cloud")
    text = " ".join(df['Resume'].astype(str))
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
