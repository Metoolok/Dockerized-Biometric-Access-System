import numpy as np
import cv2
from typing import List, Tuple, Optional

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from face_access_system.config.settings import (
    DLIB_PREDICTOR_PATH,
    EMBEDDING_DIM,
)

FaceROI = Tuple[np.ndarray, Tuple[int, int, int, int]]


class FaceDetector:
    TARGET_SIZE: int = 150

    def __init__(self):
        self._detector  = None
        self._predictor = None
        self._haar      = None
        self._init_detectors()

    def _init_detectors(self) -> None:
        if DLIB_AVAILABLE:
            self._init_dlib()
        else:
            self._init_opencv_fallback()

    def _init_dlib(self) -> None:
        self._detector  = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
        print("[FaceDetector] Dlib HOG + 68-landmark yüklendi.")

    def _init_opencv_fallback(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        print("[FaceDetector] ⚠️  OpenCV Haar Cascade fallback aktif.")

    def detect_faces(self, frame: np.ndarray) -> List[FaceROI]:
        if self._detector is not None:
            return self._detect_dlib(frame)
        elif self._haar is not None:
            return self._detect_opencv(frame)
        else:
            raise RuntimeError("Hiçbir yüz algılayıcı başlatılamadı.")

    def _detect_dlib(self, frame: np.ndarray) -> List[FaceROI]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._detector(rgb, 1)
        results: List[FaceROI] = []

        for d in detections:
            x1, y1 = d.left(), d.top()
            x2, y2 = d.right(), d.bottom()
            w, h   = x2 - x1, y2 - y1

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue

            shape = self._predictor(rgb, d)
            aligned = self._align_face(rgb, shape)

            if aligned is not None:
                results.append((aligned, (x1, y1, w, h)))

        return results

    def _align_face(self, rgb: np.ndarray, shape) -> Optional[np.ndarray]:
        points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        left_eye  = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center = points.mean(axis=0)
        tuple(center.astype(int))

        M = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), angle, 1.0)
        rotated = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]))

        margin = 30
        ones  = np.ones((68, 1))
        pts_h = np.hstack([points, ones]).T
        new_pts = (M @ pts_h).T

        x_min = max(0, int(new_pts[:, 0].min()) - margin)
        x_max = min(rotated.shape[1], int(new_pts[:, 0].max()) + margin)
        y_min = max(0, int(new_pts[:, 1].min()) - margin)
        y_max = min(rotated.shape[0], int(new_pts[:, 1].max()) + margin)

        if x_max - x_min < 10 or y_max - y_min < 10:
            return None

        cropped = rotated[y_min:y_max, x_min:x_max]
        resized  = cv2.resize(cropped, (self.TARGET_SIZE, self.TARGET_SIZE))

        return resized

    def _detect_opencv(self, frame: np.ndarray) -> List[FaceROI]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        results: List[FaceROI] = []

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            resized  = cv2.resize(face_rgb, (self.TARGET_SIZE, self.TARGET_SIZE))
            results.append((resized, (x, y, w, h)))

        return results