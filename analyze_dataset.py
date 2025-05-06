import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

def show_visualizations():
    df = pd.read_csv("UpdatedResumeDataSet_Encoded.csv")

    st.header("📊 Resume Category Distribution")
    category_counts = df['Category'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=category_counts.values, y=category_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Resumes")
    ax.set_ylabel("Category")
    st.pyplot(fig)

    st.header("☁️ Word Cloud of All Resume Text")
    text = " ".join(df['Resume'].tolist())
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
