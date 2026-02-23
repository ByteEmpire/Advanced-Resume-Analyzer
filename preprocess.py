import re

STOPWORDS = {
    "a","an","the","and","or","is","are","of","to",
    "in","for","on","with","as","by","from"
}

def clean_resume(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)
