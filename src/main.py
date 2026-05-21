from dataclasses import dataclass
from typing import List


@dataclass
class PredictionResult:
    input_vector: List[float]
    predicted_class: str
    confidence: float
    explanation: str


def classify_profile(scores: List[float]) -> PredictionResult:
    """
    Simule une mini logique de classification.

    scores[0] = orientation recherche
    scores[1] = orientation production
    scores[2] = orientation produit
    """
    if len(scores) != 3:
        raise ValueError("The input vector must contain exactly 3 scores.")

    labels = [
        "Research-oriented AI profile",
        "Production-oriented ML engineer profile",
        "Product-oriented AI builder profile"
    ]

    explanations = [
        "The highest score is linked to experimentation, modeling and research work.",
        "The highest score is linked to deployment, monitoring and production reliability.",
        "The highest score is linked to user value, product design and practical AI use cases."
    ]

    max_score = max(scores)
    predicted_index = scores.index(max_score)

    return PredictionResult(
        input_vector=scores,
        predicted_class=labels[predicted_index],
        confidence=max_score,
        explanation=explanations[predicted_index]
    )


def main() -> None:
    sample = [0.82, 0.15, 0.03]
    result = classify_profile(sample)

    print(f"Input vector: {result.input_vector}")
    print(f"Predicted class: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    main()
