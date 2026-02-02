import numpy as np
from typing import Optional

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from face_access_system.config.settings import (
    DLIB_RECOGNITION_MODEL_PATH,
    EMBEDDING_DIM,
)


class EmbeddingExtractor:
    def __init__(self):
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        if DLIB_AVAILABLE:
            self._model = dlib.face_recognition_model_v1(
                DLIB_RECOGNITION_MODEL_PATH
            )
            print("[EmbeddingExtractor] Dlib ResNet-29 recognition model yüklendi.")
        else:
            print("[EmbeddingExtractor] ⚠️  Dlib model bulunamadı. Fallback aktif.")

    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        if face_image is None:
            return None

        if self._model is not None:
            return self._extract_dlib(face_image)
        else:
            return self._extract_fallback(face_image)

    def _extract_dlib(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            descriptor = self._model.compute_face_descriptor(face_image)
            embedding = np.array(descriptor, dtype=np.float32)

            assert embedding.shape == (EMBEDDING_DIM,), (
                f"Beklenmeyen embedding boyutu: {embedding.shape}"
            )

            return embedding

        except Exception as e:
            print(f"[EmbeddingExtractor] Dlib extraction hata: {e}")
            return None

    def _extract_fallback(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        import cv2

        try:
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image

            resized = cv2.resize(gray, (32, 32))
            flat = resized.flatten().astype(np.float32)
            flat = flat - flat.mean()

            rng = np.random.RandomState(42)
            projection_matrix = rng.randn(1024, EMBEDDING_DIM).astype(np.float32)
            projection_matrix /= np.linalg.norm(projection_matrix, axis=0)

            embedding = flat @ projection_matrix

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            print(f"[EmbeddingExtractor] Fallback extraction hata: {e}")
            return None