import numpy as np
from enum import Enum


class SimilarityMethod(Enum):
    COSINE = "cosine"
    L2     = "l2"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(similarity, -1.0, 1.0))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_similarity(
    a: np.ndarray,
    b: np.ndarray,
    method: SimilarityMethod = SimilarityMethod.COSINE
) -> float:
    if method == SimilarityMethod.COSINE:
        return cosine_similarity(a, b)
    elif method == SimilarityMethod.L2:
        dist = l2_distance(a, b)
        return max(0.0, 1.0 - dist / 2.0)
    else:
        raise ValueError(f"Bilinmeyen metot: {method}")