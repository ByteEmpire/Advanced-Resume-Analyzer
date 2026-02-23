import streamlit as st
import joblib
import fitz
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from predict_resume import predict_resume
from preprocess import clean_resume
from pdf_report import generate_resume_pdf

# ================== CONFIG ==================
st.set_page_config(
    page_title="Advanced Resume Analyzer",
    layout="wide"
)

st.title("📄 Advanced Resume Analyzer")

# ================== CACHE ==================
@st.cache_resource
def load_assets():
    model = joblib.load("resume_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    label_map = joblib.load("label_mapping.pkl")
    return model, tfidf, label_map

model, tfidf, label_map = load_assets()

# ================== CONSTANTS ==================
SKILLS = {
    "python", "java", "sql", "c++", "javascript",
    "machine learning", "deep learning", "data science",
    "pandas", "numpy", "excel"
}

ROLE_SKILLS = {
    "Data Scientist": {"python", "pandas", "numpy", "machine learning"},
    "Software Engineer": {"java", "python", "c++", "sql", "javascript"},
    "ML Engineer": {"python", "machine learning", "deep learning"},
    "Data Analyst": {"sql", "excel", "python", "pandas"}
}

# ================== HELPERS ==================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

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

# ================== UI ==================
tabs = st.tabs(["🔍 Analyze Resume", "📊 Insights"])

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# ================== TAB 1 ==================
with tabs[0]:
    option = st.radio("Input Method", ["Paste Text", "Upload PDF"])

    if option == "Paste Text":
        st.session_state.resume_text = st.text_area(
            "Paste resume text",
            st.session_state.resume_text,
            height=220
        )
    else:
        pdf_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
        if pdf_file:
            st.session_state.resume_text = extract_text_from_pdf(pdf_file)
            st.text_area("Extracted Text", st.session_state.resume_text, height=220)

    if st.button("Analyze Resume"):
        raw_text = st.session_state.resume_text.strip()

        if len(raw_text) < 150:
            st.warning("Resume text is too short for reliable analysis.")
            st.stop()

        cleaned_text = clean_resume(raw_text)
        predictions = predict_resume(cleaned_text, top_k=3)

        predicted_role, confidence = predictions[0]
        score = resume_score(raw_text)
        skills_found = extract_skills(raw_text)
        missing_skills = ROLE_SKILLS.get(predicted_role, set()) - set(skills_found.keys())

        # ---- METRICS ----
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Role", predicted_role)
        col2.metric("Confidence", f"{confidence*100:.1f}%")
        col3.metric("Resume Score", f"{score}/100")

        st.progress(confidence)

        # ---- WORD CLOUD ----
        wc = WordCloud(width=900, height=300, background_color="white").generate(raw_text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # ---- PDF EXPORT ----
        pdf_bytes = generate_resume_pdf(
            predicted_role=predicted_role,
            confidence=confidence,
            resume_score=score,
            skills_found=list(skills_found.keys()),
            missing_skills=list(missing_skills),
            top_roles=predictions
        )

        st.download_button(
            label="📄 Download Resume Analysis PDF",
            data=pdf_bytes,
            file_name="resume_analysis_report.pdf",
            mime="application/pdf"
        )

# ================== TAB 2 ==================
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
            for role, prob in predict_resume(clean_resume(text), top_k=3):
                st.write(f"• **{role}** — {prob*100:.1f}%")

        predicted_role = predict_resume(clean_resume(text))[0][0]
        missing = ROLE_SKILLS.get(predicted_role, set()) - set(skills.keys())

        st.subheader("🚧 Skill Gap Analysis")
        if missing:
            st.warning(", ".join(missing))
        else:
            st.success("You meet core requirements for this role.")
