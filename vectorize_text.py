import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Function to clean the resume text
def clean_resume(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Load the dataset (without cleaned_resume)
df = pd.read_csv('UpdatedResumeDataSet_Encoded.csv')

# Clean the 'Resume' column
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Fit and transform the cleaned resumes
X = tfidf.fit_transform(df['cleaned_resume'])

# Convert the TF-IDF matrix to a DataFrame
X_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

# Save the features to a CSV file
X_df.to_csv('tfidf_features.csv', index=False)

# Print shape info
print("TF-IDF Matrix Shape:", X.shape)
