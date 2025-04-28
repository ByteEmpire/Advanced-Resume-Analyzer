from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize Flask app
app = Flask(__name__)

# Load the resume dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Preprocess function (same as before)
def preprocess_text(text):
    # Tokenize, remove stop words, and lemmatize
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Preprocess the resumes and transform them into TF-IDF vectors
df['Cleaned_Resume'] = df['Resume'].apply(preprocess_text)
X = tfidf.fit_transform(df['Cleaned_Resume'])

# Function to get cosine similarity
def get_similarity(job_desc, resume_texts):
    job_desc_cleaned = preprocess_text(job_desc)
    job_desc_tfidf = tfidf.transform([job_desc_cleaned])
    similarities = cosine_similarity(job_desc_tfidf, X)
    return similarities

# Route to display the homepage and get job description input
@app.route('/')
def home():
    return render_template('index.html')

# Route to process job description and return results
@app.route('/analyze', methods=['POST'])
def analyze():
    job_desc = request.form['job_desc']
    similarity_scores = get_similarity(job_desc, df['Resume'])
    df['Similarity_Score'] = similarity_scores.flatten()
    df_sorted = df.sort_values(by='Similarity_Score', ascending=False)
    
    top_5_resumes = df_sorted[['Category', 'Similarity_Score']].head()
    return jsonify(top_5_resumes.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
