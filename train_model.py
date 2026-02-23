import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_resume

df = pd.read_csv("UpdatedResumeDataSet_Encoded.csv")

df["cleaned_resume"] = df["Resume"].astype(str).apply(clean_resume)

X = df["cleaned_resume"]
y = df["Category_encoded"]

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "resume_classifier_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model trained and saved.")
