import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_resume(text):
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Apply cleaning to the 'Resume' column
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Check if 'cleaned_resume' is now added
print(df.columns)  # This should now show 'cleaned_resume' as a column

# Display the first few cleaned resumes
print(df[['Resume', 'cleaned_resume']].head())
