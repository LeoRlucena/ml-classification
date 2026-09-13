"""
Loads .joblib model and classifies a given text input, returning the predicted class and confidence score.
"""

from __future__ import annotations

from functools import lru_cache

import joblib

from src.anonimizer import anonymize
from src.config import MODEL_PATH, CONFIDENCE_THRESHOLD

@lru_cache(maxsize=1)
def load_artifact() -> dict:
    """
    Load the model artifact from disk and cache it for future use.
    Returns a dictionary containing the model and its associated classes.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Please train the model first.")
    return joblib.load(MODEL_PATH)

def identify_text(text: str, text_origin: str = "json") -> dict:
    """
    Classify the given text input using the loaded model.
    Returns a dictionary containing the predicted class, confidence score, and origin of the text.
    """
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    per_model_type = artifact["per_model_type"]
    threshold = float(artifact.get("confidence_threshold", CONFIDENCE_THRESHOLD))

    clean_text = anonymize(text or "").strip()
    if not clean_text:
        return {
            "type": "Desconhecido",
            "model": "Desconhecido",
            "confidence": 0.0,
            "needs_review": True,
            "origin": text_origin,
            "error": "Texto vazio. Cole o conteúdo extraído do PDF.",
        }

    model = str(pipeline.predict([clean_text])[0])
    proba = pipeline.predict_proba([clean_text])[0]
    confidence = float(max(proba))
    type = per_model_type.get(model, "Desconhecido")

    return {
        "type": type,
        "model": model,
        "confidence": round(confidence, 4),
        "needs_review": confidence < threshold,
        "origin": text_origin,
    }