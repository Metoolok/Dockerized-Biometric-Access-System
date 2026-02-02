from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from face_access_system.config.settings import SIMILARITY_THRESHOLD
from face_access_system.recognition.similarity import (
    compute_similarity,
    SimilarityMethod,
)
from face_access_system.database.crud import get_all_users
from face_access_system.database.models import User


@dataclass
class RecognitionResult:
    matched_user:  Optional[User]
    confidence:    float
    is_recognized: bool
    all_scores:    List[tuple]


class FaceRecognizer:
    def __init__(
        self,
        threshold: float = SIMILARITY_THRESHOLD,
        method: SimilarityMethod = SimilarityMethod.COSINE
    ):
        self.threshold = threshold
        self.method    = method
        print(f"[Recognizer] Threshold={threshold}, Method={method.value}")

    def recognize(self, query_embedding: np.ndarray) -> RecognitionResult:
        all_users = get_all_users()

        if not all_users:
            return RecognitionResult(
                matched_user=None,
                confidence=0.0,
                is_recognized=False,
                all_scores=[]
            )

        scores: List[tuple] = []

        for user in all_users:
            score = compute_similarity(
                query_embedding,
                user.embedding,
                method=self.method
            )
            scores.append((user, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_user, best_score = scores[0]

        lower_threshold = 0.9
        upper_threshold = 1.0
        is_recognized = lower_threshold <= best_score <= upper_threshold

        return RecognitionResult(
            matched_user=best_user if is_recognized else None,
            confidence=best_score,
            is_recognized=is_recognized,
            all_scores=scores
        )

    def recognize_batch(
        self,
        embeddings: List[np.ndarray]
    ) -> List[RecognitionResult]:
        return [self.recognize(emb) for emb in embeddings]