import streamlit as st
import joblib
import fitz
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter

try:
    from analyze_dataset import show_visualizations
except Exception:
    show_visualizations = None

# Load ML assets
model = joblib.load('resume_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_mapping = joblib.load('label_mapping.pkl')


def predict_category(resume_text):
    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)[0]
    return list(label_mapping.keys())[list(label_mapping.values()).index(prediction)]


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)


def extract_skills(resume_text):
    skills_list = [
        'Python', 'Java', 'SQL', 'Machine Learning', 'Deep Learning',
        'Data Science', 'C++', 'JavaScript', 'Excel', 'Project Management'
    ]
    return Counter(skill for skill in skills_list if skill.lower() in resume_text.lower())


def extract_experience(resume_text):
    years = re.findall(r'(\d+)\s?year', resume_text.lower())
    return sum(map(int, years)) if years else 0


def extract_education(resume_text):
    levels = ['bachelor', 'master', 'phd', 'diploma']
    return [lvl for lvl in levels if lvl in resume_text.lower()]


# ================= STREAMLIT UI =================

st.set_page_config(page_title="Advanced Resume Analyzer", layout="wide")
st.title("📄 Advanced Resume Analyzer")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Resume", "📊 Resume Insights", "📁 Dataset Analysis"])

resume_text = ""

# -------- TAB 1 --------
with tab1:
    option = st.radio("Choose Input Method:", ["Paste Text", "Upload PDF"])

    if option == "Paste Text":
        resume_text = st.text_area("Paste your resume text here:")
    else:
        pdf_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])
        if pdf_file:
            resume_text = extract_text_from_pdf(pdf_file)
            st.text_area("Extracted Resume Text:", resume_text, height=200)

    if st.button("Analyze Resume"):
        if resume_text.strip():
            category = predict_category(resume_text)
            st.success(f"✅ Predicted Category: **{category}**")

            wc = WordCloud(width=800, height=300, background_color='white').generate(resume_text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Please provide resume content.")


# -------- TAB 2 --------
with tab2:
    if resume_text.strip():
        skills = extract_skills(resume_text)

        if skills:
            fig, ax = plt.subplots()
            ax.barh(*zip(*skills.items()))
            ax.set_title("Skills Found")
            st.pyplot(fig)

        st.write(f"**Experience:** {extract_experience(resume_text)} years")
        edu = extract_education(resume_text)
        st.write(f"**Education:** {', '.join(edu) if edu else 'Not Found'}")


# -------- TAB 3 --------
with tab3:
    if show_visualizations:
        show_visualizations()
    else:
        st.warning("Dataset visualizations unavailable.")
