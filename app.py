import streamlit as st
import joblib
import fitz
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="Advanced Resume Analyzer",
    layout="wide"
)

# ================= CACHING =================
@st.cache_resource
def load_models():
    model = joblib.load("resume_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    label_mapping = joblib.load("label_mapping.pkl")
    return model, tfidf, label_mapping

@st.cache_data
def load_dataset():
    if os.path.exists("UpdatedResumeDataSet_Encoded.csv"):
        import pandas as pd
        return pd.read_csv("UpdatedResumeDataSet_Encoded.csv")
    return None

model, tfidf, label_mapping = load_models()
df_dataset = load_dataset()

# ================= HELPERS =================
SKILLS = {
    "python", "java", "sql", "c++", "javascript",
    "machine learning", "deep learning", "data science",
    "excel", "project management", "pandas", "numpy"
}

ROLE_SKILLS = {
    "Data Scientist": {"python", "pandas", "numpy", "machine learning", "statistics"},
    "Software Engineer": {"java", "python", "c++", "sql", "javascript"},
    "ML Engineer": {"python", "machine learning", "deep learning", "numpy"},
    "Data Analyst": {"sql", "excel", "python", "pandas"}
}

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def extract_skills(text):
    text = text.lower()
    return Counter(skill for skill in SKILLS if skill in text)

def extract_experience(text):
    matches = re.findall(r'(\d+)\s*\+?\s*(year|years|yr|yrs)', text.lower())
    return sum(int(m[0]) for m in matches) if matches else 0

def extract_education(text):
    levels = ["bachelor", "master", "phd", "diploma"]
    return [lvl for lvl in levels if lvl in text.lower()]

def predict_top_roles(text):
    vector = tfidf.transform([text])
    probs = model.predict_proba(vector)[0]
    labels = list(label_mapping.keys())
    indices = probs.argsort()[::-1][:3]
    return [(labels[i], probs[i]) for i in indices]

def resume_strength_score(text):
    score = 0
    score += min(len(extract_skills(text)) * 5, 30)
    score += min(extract_experience(text) * 5, 30)
    score += 20 if extract_education(text) else 0
    score += 20 if len(text) > 600 else 10
    return min(score, 100)

# ================= UI =================
st.title("📄 Advanced Resume Analyzer")

tabs = st.tabs([
    "🔍 Analyze Resume",
    "📊 Resume Insights",
    "📁 Dataset Insights"
])

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
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf:
            st.session_state.resume_text = extract_text_from_pdf(pdf)
            st.text_area("Extracted Text", st.session_state.resume_text, height=220)

    if st.button("Analyze Resume"):
        text = st.session_state.resume_text.strip()

        if len(text) < 150:
            st.warning("Resume text is too short for reliable analysis.")
            st.stop()

        top_roles = predict_top_roles(text)
        main_role, confidence = top_roles[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Role", main_role)
        col2.metric("Confidence", f"{confidence*100:.1f}%")
        col3.metric("Resume Score", f"{resume_strength_score(text)}/100")

        st.progress(confidence)

        wc = WordCloud(width=900, height=300, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

# ---------- TAB 2 ----------
with tabs[1]:
    text = st.session_state.resume_text.strip()
    if text:
        skills = extract_skills(text)
        experience = extract_experience(text)
        education = extract_education(text)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛠 Skills Found")
            if skills:
                fig, ax = plt.subplots()
                ax.barh(list(skills.keys()), list(skills.values()))
                st.pyplot(fig)
            else:
                st.info("No recognized skills found.")

        with col2:
            st.subheader("📌 Profile Summary")
            st.write(f"**Experience:** {experience} years")
            st.write(f"**Education:** {', '.join(education) if education else 'Not detected'}")

        st.subheader("🎯 Top Role Suggestions")
        for role, prob in predict_top_roles(text):
            st.write(f"- **{role}** → {prob*100:.1f}%")

        predicted_role = predict_top_roles(text)[0][0]
        required = ROLE_SKILLS.get(predicted_role, set())
        missing = required - set(skills.keys())

        st.subheader("🚧 Skill Gap Analysis")
        if missing:
            st.warning(f"Missing skills for **{predicted_role}**:")
            st.write(", ".join(missing))
        else:
            st.success("You meet the core skill requirements for this role.")

# ---------- TAB 3 ----------
with tabs[2]:
    if df_dataset is None:
        st.warning("Dataset not available.")
    else:
        st.subheader("📊 Resume Dataset Insights")
        role_counts = df_dataset["Category"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(role_counts.index[:10], role_counts.values[:10])
        ax.set_title("Top Resume Categories")
        st.pyplot(fig)
