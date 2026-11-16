"""Fungsi inferensi dan explainability untuk model klasifikasi teks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from src.preprocess import basic_clean, normalize_slang, stem_text


def prepare_text(text: str, slang_map: dict[str, str], use_stemming: bool) -> str:
    cleaned = basic_clean(text)
    cleaned = normalize_slang(cleaned, slang_map)
    return stem_text(cleaned) if use_stemming else cleaned


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(values)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def _class_scores(pipeline: Pipeline, features) -> np.ndarray:
    classifier = pipeline.named_steps["clf"]
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(features)[0]
    return _softmax(classifier.decision_function(features))[0]


def _word_feature_names(pipeline: Pipeline) -> list[str]:
    feature_union = pipeline.named_steps["features"]
    word_vectorizer = feature_union.transformer_list[0][1]
    return list(word_vectorizer.get_feature_names_out())


def _contributions(
    pipeline: Pipeline, features, predicted_index: int
) -> tuple[np.ndarray, list[str]]:
    """Kontribusi fitur kata terhadap kelas prediksi."""
    classifier = pipeline.named_steps["clf"]
    word_names = _word_feature_names(pipeline)
    n_words = len(word_names)

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_
        if coefficients.shape[0] == 1:
            weights = coefficients[0]
        else:
            weights = coefficients[predicted_index]
        contributions = features.toarray()[0][:n_words] * weights[:n_words]
    elif hasattr(classifier, "feature_log_prob_"):
        log_prob = classifier.feature_log_prob_
        weights = log_prob[predicted_index] - log_prob.mean(axis=0)
        contributions = features.toarray()[0][:n_words] * weights[:n_words]
    else:
        contributions = np.zeros(n_words)

    return contributions, word_names


def predict_text(
    pipeline: Pipeline,
    text: str,
    slang_map: dict[str, str],
    use_stemming: bool = True,
    top_k: int = 6,
) -> dict[str, Any]:
    """Prediksi satu teks beserta skor kelas dan kata yang berpengaruh."""
    final_text = prepare_text(text, slang_map, use_stemming)
    features = pipeline.named_steps["features"].transform([final_text])
    scores = _class_scores(pipeline, features)
    classes = [str(label) for label in pipeline.named_steps["clf"].classes_]
    predicted_index = int(np.argmax(scores))
    predicted_label = classes[predicted_index]

    contributions, word_names = _contributions(pipeline, features, predicted_index)
    present = np.flatnonzero(features.toarray()[0][: len(word_names)] > 0)
    ranked = present[np.argsort(contributions[present])[::-1]]
    positive = [
        {"token": word_names[index], "weight": round(float(contributions[index]), 6)}
        for index in ranked[:top_k]
        if contributions[index] > 0
    ]
    negative = [
        {"token": word_names[index], "weight": round(float(contributions[index]), 6)}
        for index in ranked[::-1][:top_k]
        if contributions[index] < 0
    ]

    probabilities = {
        label: round(float(score), 6) for label, score in zip(classes, scores)
    }
    return {
        "label": predicted_label,
        "confidence": round(float(scores[predicted_index]), 6),
        "probabilities": probabilities,
        "cleaned_text": final_text,
        "supporting_tokens": positive,
        "opposing_tokens": negative,
    }
