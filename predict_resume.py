import joblib
from functools import lru_cache

@lru_cache(maxsize=1)
def load_assets():
    model = joblib.load("resume_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    mapping = joblib.load("label_mapping.pkl")
    return model, tfidf, mapping["id_to_label"]

def predict_resume(text, top_k=1):
    model, tfidf, id_to_label = load_assets()
    vec = tfidf.transform([text])
    probs = model.predict_proba(vec)[0]

    ranked = sorted(
        [(id_to_label[i], probs[i]) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]
