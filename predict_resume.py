import joblib
import pandas as pd

# Load the saved model and vectorizer
model = joblib.load('resume_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Function to predict the category of a new resume
def predict_resume(resume_text):
    # Transform the resume text using the saved vectorizer
    resume_vector = tfidf.transform([resume_text])
    
    # Make prediction using the loaded model
    predicted_category = model.predict(resume_vector)
    
    # Map the predicted category number back to the category name
    label_mapping = joblib.load('label_mapping.pkl')  # Assuming you saved this mapping earlier
    category_name = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_category[0])]
    
    return category_name

# Example usage
resume = "Experienced Data Scientist with a background in machine learning and AI."
predicted_category = predict_resume(resume)
print(f"The resume belongs to the '{predicted_category}' category.")
