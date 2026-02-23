import joblib
from functools import lru_cache

@lru_cache(maxsize=1)
def load_assets():
    """
    Load model, vectorizer, and label mapping safely.
    Supports BOTH old and new label_mapping.pkl formats.
    """
    model = joblib.load("resume_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    mapping = joblib.load("label_mapping.pkl")

    # Backward + forward compatible mapping handling
    # New format: {"label_to_id": {...}, "id_to_label": {...}}
    if isinstance(mapping, dict) and "id_to_label" in mapping:
        id_to_label = mapping["id_to_label"]

    # Old format: {"Data Scientist": 0, "Software Engineer": 1, ...}
    elif isinstance(mapping, dict):
        id_to_label = {v: k for k, v in mapping.items()}

    else:
        raise ValueError("Invalid label_mapping.pkl format")

    return model, tfidf, id_to_label


def predict_resume(text, top_k=1):
    """
    Predict resume category.
    Returns a list of (label, confidence) tuples.
    """
    model, tfidf, id_to_label = load_assets()
    vector = tfidf.transform([text])

    # Safe probability handling
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
    else:
        # Fallback (should rarely happen)
        prediction = model.predict(vector)[0]
        probabilities = [0.0] * len(id_to_label)
        probabilities[prediction] = 1.0

    ranked = sorted(
        [(id_to_label[i], probabilities[i]) for i in range(len(probabilities))],
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


# ---------- OPTIONAL CLI TEST ----------
if __name__ == "__main__":
    sample_resume = "Experienced Data Scientist with strong Python and Machine Learning skills"
    results = predict_resume(sample_resume, top_k=3)

    for role, confidence in results:
        print(f"{role}: {confidence*100:.2f}%")
