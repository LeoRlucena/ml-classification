"""
Trains a machine learning model on a given dataset and saves it to disk.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    MODEL_PATH,
    CLASSES,
    CSV_DATASET_PATH,
    CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    PROCESSED_DIR,
    SEED,
    TEST_SIZE,
)

TYPE_PER_MODEL = dict(CLASSES)  # Create a dictionary mapping model names to their corresponding types

def build_pipeline() -> Pipeline:
    """
    Build a machine learning pipeline consisting of a TF-IDF vectorizer and a logistic regression classifier.
    Returns the constructed pipeline.
    """
    return Pipeline(
        steps = [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )

def main() -> None:
    """
    Main function to train the machine learning model on the dataset and save it to disk.
    """
    df = pd.read_csv(CSV_DATASET_PATH)
    x = df["text"]
    y = df["model"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    print(
        f"Treino: {len(x_train)} docs | Teste: {len(x_test)} docs "
        f"(test_size={TEST_SIZE})"
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)
    confidence = y_proba.max(axis=1)

    print("\n=== Relatório no conjunto de TESTE (não treinado) ===")
    print(classification_report(y_test, y_pred, digits=3))

    needs_review = confidence < CONFIDENCE_THRESHOLD
    n_reviews = int(needs_review.sum())
    print(
        f"Limiar {CONFIDENCE_THRESHOLD:.0%}: {n_reviews}/{len(x_test)} "
        f"({n_reviews / len(x_test):.1%}) iriam para revisão humana."
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    figures = PROCESSED_DIR / "confusion_matrix.png"
    labels = pipeline.named_steps["clf"].classes_
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Matriz de Confusão - TESTE")
    fig.tight_layout()
    fig.savefig(figures, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix figure to {figures}")

    joblib.dump(
        {
            "pipeline": pipeline,
            "per_model_type": TYPE_PER_MODEL,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        MODEL_PATH
    )
    print(f"Saved trained model to {MODEL_PATH}")

    report = classification_report(y_test, y_pred, output_dict=True)
    (PROCESSED_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "review_fraction": n_reviews / len(x_test),
                "report": report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()