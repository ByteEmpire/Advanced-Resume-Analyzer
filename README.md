# 📄 Advanced Resume Analyzer

An **AI-based Resume Analysis Web Application** that predicts suitable job roles, extracts skills, identifies skill gaps, and generates a **downloadable PDF report** using Machine Learning.

This project focuses on **practical ML application development**, **clean architecture**, and **production-ready deployment using Streamlit**.

---

## 🚀 Project Overview

The Advanced Resume Analyzer helps users:
- Understand which job role best matches their resume
- Identify strengths and missing skills
- Get structured insights instead of raw predictions
- Download a professional **Resume Analysis PDF**

The project is built as a **complete ML pipeline**:
preprocessing → vectorization → model training → inference → visualization → PDF export.

---

## ✨ Features

### 🔍 Resume Analysis
- Upload resume as **PDF** or paste resume text
- Automatic text extraction and preprocessing
- Predicts the **most relevant job role**
- Displays **prediction confidence**
- Shows **Top-3 role recommendations**

### 📊 Resume Insights
- Resume **strength score (0–100)**
- Skill extraction from resume text
- **Skill gap analysis** based on predicted role
- Resume word cloud visualization

### 📄 PDF Report Export
- One-click **Resume Analysis PDF download**
- PDF includes:
  - Predicted role and confidence
  - Resume score
  - Skills found
  - Missing skills
  - Top role suggestions
  - Timestamp

---

## 🧠 Machine Learning Approach

- Text Vectorization: **TF-IDF**
- Classification Model: **Logistic Regression**
- Label Encoding: Deterministic category mapping
- Training & inference use the **same preprocessing pipeline**
- Cached inference for faster performance

---

## 🏗️ Tech Stack

- **Frontend / App**: Streamlit  
- **Machine Learning**: Scikit-learn  
- **Text Processing**: TF-IDF  
- **PDF Handling**: PyMuPDF, ReportLab  
- **Visualization**: Matplotlib, WordCloud  
- **Language**: Python  

---

## 📂 Project Structure

```text
advanced-resume-analyzer/
├── app.py                     # Streamlit application
├── predict_resume.py          # Inference logic
├── preprocess.py              # Text preprocessing
├── pdf_report.py              # PDF report generator
├── analyze_dataset.py         # Dataset analytics
│
├── train_model.py             # Model training (offline)
├── encode_labels.py           # Label encoding (offline)
├── generate_predictions.py    # Batch prediction script
│
├── requirements.txt           # Dependencies
│
├── resume_classifier_model.pkl
├── tfidf_vectorizer.pkl
├── label_mapping.pkl
│
├── UpdatedResumeDataSet.csv
└── UpdatedResumeDataSet_Encoded.csv
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/advanced-resume-analyzer.git  
cd advanced-resume-analyzer  
```

### 2️⃣ (Optional) Create Virtual Environment
```bash
python -m venv venv  
source venv/bin/activate  
# Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt  
```

### 4️⃣ Run the Application
```bash
streamlit run app.py  
```

---

## ☁️ Streamlit Cloud Deployment

1. Push the repository to GitHub  
2. Open **Streamlit Cloud**
3. Select the repository
4. Set **Main file** to `app.py`
5. Deploy

Ensure `requirements.txt` is present in the root directory.

---

## 🧪 Model Training (Optional)

To retrain the model:

```bash
python encode_labels.py  
python train_model.py  
```

This regenerates:
- resume_classifier_model.pkl
- tfidf_vectorizer.pkl
- label_mapping.pkl

---

## 📈 Example Use Cases

- Students exploring suitable job roles
- Resume evaluation and improvement
- Skill gap identification
- Career guidance demonstrations
- Machine Learning portfolio project

---

## 🔐 Engineering Highlights

- Consistent preprocessing for training & inference
- Cached model loading for performance
- Backward-compatible label decoding
- Safe PDF generation
- No runtime NLP downloads

---

## 👨‍💻 Shubhranshu Kumar
B.Tech – Computer Science & Artificial Intelligence  
IIIT Lucknow

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub.
