import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
df = pd.read_csv('UpdatedResumeDataSet_Encoded.csv')

# Features (Resumes) and target (encoded categories)
X = df['Resume']
y = df['Category_encoded']

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Fit and transform the text data to get the TF-IDF features
X_tfidf = tfidf.fit_transform(X)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'resume_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and Vectorizer saved successfully!")
