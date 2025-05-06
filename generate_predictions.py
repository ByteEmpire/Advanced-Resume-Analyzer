import pandas as pd
import joblib

# Load the saved model and vectorizer
model = joblib.load('resume_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Load the dataset
df = pd.read_csv('UpdatedResumeDataSet_Encoded.csv')

# Function to predict the category of a new resume
def predict_resume(resume_text):
    resume_vector = tfidf.transform([resume_text])
    predicted_category = model.predict(resume_vector)
    return predicted_category[0]

# Generate predictions for all resumes in the dataset
df['Predicted_Category'] = df['Resume'].apply(predict_resume)

# Save the predictions to a new CSV file
df.to_csv('Resume_Predictions.csv', index=False)

print("Predictions generated and saved to 'Resume_Predictions.csv'")
