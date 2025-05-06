import streamlit as st
from analyze_dataset import show_visualizations
import joblib
import fitz
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter

# Load model/vectorizer/labels
model = joblib.load('resume_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_mapping = joblib.load('label_mapping.pkl')

def predict_category(resume_text):
    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)[0]
    category = list(label_mapping.keys())[list(label_mapping.values()).index(prediction)]
    return category

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(resume_text):
    skills_list = ['Python', 'Java', 'SQL', 'Machine Learning', 'Deep Learning', 'Data Science', 'C++', 'JavaScript', 'Excel', 'Project Management']
    skills_found = [skill for skill in skills_list if skill.lower() in resume_text.lower()]
    return Counter(skills_found)

def extract_experience(resume_text):
    experience = re.findall(r'(\d+)\s?year', resume_text.lower())
    return sum([int(exp) for exp in experience])

def extract_education(resume_text):
    education_levels = ['bachelor', 'master', 'phd', 'diploma']
    found_degrees = [degree for degree in education_levels if degree in resume_text.lower()]
    return found_degrees

# Streamlit layout
st.title("📄 Advanced Resume Analyzer")

tab1, tab2 = st.tabs(["🔍 Analyze Resume", "📊 Visual Insights"])

with tab1:
    option = st.radio("Choose Input Method:", ["Paste Text", "Upload PDF"])
    resume_text = ""

    if option == "Paste Text":
        resume_text = st.text_area("Paste your resume text here:")
    elif option == "Upload PDF":
        pdf_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])
        if pdf_file:
            resume_text = extract_text_from_pdf(pdf_file)
            st.text_area("Extracted Resume Text:", resume_text, height=200)

    if st.button("Analyze Resume"):
        if resume_text.strip():
            category = predict_category(resume_text)
            st.success(f"✅ Predicted Category: **{category}**")

            # ----- Word Cloud for Resume -----
            st.subheader("☁️ Word Cloud for This Resume")
            wc = WordCloud(width=800, height=300, background_color='white').generate(resume_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.warning("Please provide some resume text.")

with tab2:
    if resume_text.strip():
        st.header("📊 Visual Insights")

        # ----- Skills Match Visualization -----
        skills_counter = extract_skills(resume_text)
        if skills_counter:
            st.subheader("💼 Skills Present in the Resume")
            skills, counts = zip(*skills_counter.items())

            # Plot the bar chart
            fig, ax = plt.subplots()
            ax.barh(skills, counts, color='skyblue')
            ax.set_xlabel('Skill Frequency')
            ax.set_title('Top Skills in Resume')
            st.pyplot(fig)
        else:
            st.warning("No matching skills found in the resume.")

        # ----- Experience and Education Insights -----
        experience = extract_experience(resume_text)
        education = extract_education(resume_text)

        # Experience
        st.subheader(f"🔹 Years of Experience: {experience} years")

        # Education
        st.subheader(f"🔹 Education Level: {', '.join(education) if education else 'Not Found'}")

        # ----- Role Fit Visualization -----
        predicted_category = predict_category(resume_text)

        st.subheader(f"🔹 Predicted Role: {predicted_category}")
        role_distribution = {"Predicted Role": 1, "Other": 1}  # Example distribution

        fig, ax = plt.subplots()
        ax.pie(role_distribution.values(), labels=role_distribution.keys(), autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        st.pyplot(fig)

        # ----- Resume Structure Insights -----
        sections = ['Experience', 'Education', 'Skills', 'Certifications', 'Projects']
        section_counts = {section: resume_text.lower().count(section.lower()) for section in sections}

        fig, ax = plt.subplots()
        ax.bar(section_counts.keys(), section_counts.values(), color='salmon')
        ax.set_ylabel('Count')
        ax.set_title('Resume Structure Breakdown')
        st.pyplot(fig)
