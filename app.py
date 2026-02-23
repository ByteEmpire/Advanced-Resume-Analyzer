import streamlit as st
import joblib
import fitz
import re
import os
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from pdf_report import generate_resume_pdf
from predict_resume import predict_resume
from preprocess import clean_resume

st.set_page_config(page_title="Advanced Resume Analyzer", layout="wide")
st.title("📄 Advanced Resume Analyzer")

# ---------- Cache ----------
@st.cache_resource
def load_assets():
    model = joblib.load("resume_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    label_map = joblib.load("label_mapping.pkl")
    return model, tfidf, label_map

model, tfidf, label_map = load_assets()

# ---------- Helpers ----------
SKILLS = {
    "python","java","sql","c++","javascript",
    "machine learning","deep learning","data science",
    "pandas","numpy","excel"
}

ROLE_SKILLS = {
    "Data Scientist": {"python","pandas","numpy","machine learning"},
    "Software Engineer": {"java","python","c++","sql","javascript"},
    "ML Engineer": {"python","machine learning","deep learning"},
    "Data Analyst": {"sql","excel","python","pandas"}
}

def extract_text_from_pdf(pdf):
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    return " ".join(p.get_text() for p in doc)

def extract_skills(text):
    text = text.lower()
    return Counter(skill for skill in SKILLS if skill in text)

def extract_experience(text):
    matches = re.findall(r'(\d+)\s*(year|years|yr|yrs)', text.lower())
    return sum(int(m[0]) for m in matches) if matches else 0

def resume_score(text):
    score = 0
    score += min(len(extract_skills(text)) * 5, 30)
    score += min(extract_experience(text) * 5, 30)
    score += 20 if len(text) > 600 else 10
    return min(score, 100)

# ---------- UI ----------
tabs = st.tabs(["🔍 Analyze Resume", "📊 Insights"])

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# ---------- TAB 1 ----------
with tabs[0]:
    option = st.radio("Input Method", ["Paste Text", "Upload PDF"])

    if option == "Paste Text":
        st.session_state.resume_text = st.text_area(
            "Paste resume text",
            st.session_state.resume_text,
            height=220
        )
    else:
        pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])
        if pdf:
            st.session_state.resume_text = extract_text_from_pdf(pdf)
            st.text_area("Extracted Text", st.session_state.resume_text, height=220)

    if st.button("Analyze"):
        raw_text = st.session_state.resume_text.strip()

        if len(raw_text) < 150:
            st.warning("Resume text too short for reliable analysis.")
            st.stop()

        cleaned = clean_resume(raw_text)
        predictions = predict_resume(cleaned, top_k=3)

        role, conf = predictions[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Role", role)
        col2.metric("Confidence", f"{conf*100:.1f}%")
        col3.metric("Resume Score", f"{resume_score(raw_text)}/100")

        st.progress(conf)

        wc = WordCloud(width=900, height=300, background_color="white").generate(raw_text)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

# ---------- TAB 2 ----------
with tabs[1]:
    text = st.session_state.resume_text.strip()
    if text:
        skills = extract_skills(text)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛠 Skills Found")
            if skills:
                fig, ax = plt.subplots()
                ax.barh(list(skills.keys()), list(skills.values()))
                st.pyplot(fig)
            else:
                st.info("No core skills detected.")

        with col2:
            st.subheader("🎯 Top Role Suggestions")
            for r, p in predict_resume(clean_resume(text), top_k=3):
                st.write(f"• **{r}** — {p*100:.1f}%")

        predicted_role = predict_resume(clean_resume(text))[0][0]
        missing = ROLE_SKILLS.get(predicted_role, set()) - set(skills.keys())

        st.subheader("🚧 Skill Gap")
        if missing:
            st.warning(", ".join(missing))
        else:
            st.success("You meet core requirements for this role.")
