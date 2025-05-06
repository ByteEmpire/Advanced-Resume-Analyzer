import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# If you have already trained the model and vectorizer, load them here
# Load the trained model and vectorizer
model = joblib.load('resume_classifier_model.pkl')  # if saved previously
tfidf = joblib.load('tfidf_vectorizer.pkl')  # if saved previously

# Save the model and vectorizer again
joblib.dump(model, 'resume_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and Vectorizer saved successfully!")
