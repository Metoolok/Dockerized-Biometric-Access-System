
import sys
import os
import cv2
import numpy as np
import time

# Proje kök dizini
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
)
from vision.face_detector import FaceDetector
from vision.embedding_extractor import EmbeddingExtractor
from recognition.recognizer import FaceRecognizer
from app_logging.access_logger import AccessLogger, AccessStatus, AccessDecision
from scripts.init_db import create_tables

# --- GÖRSEL AYARLAR ---
COLORS = {
    "GRANTED": (0, 255, 0),
    "DENIED": (0, 0, 255),
    "UNKNOWN": (0, 100, 255),
    "INFO": (200, 200, 200),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- OPTİMİZASYON AYARLARI ---
PROCESS_EVERY_N_FRAME = 3  # Her 3 karede bir AI çalıştır
SCALE_FACTOR = 0.5  # AI analizi için görüntüyü %50 küçült (Hızı 2 kat artırır)


def draw_box_with_info(frame, x, y, w, h, name, confidence, status):
    color = {
        AccessStatus.GRANTED: COLORS["GRANTED"],
        AccessStatus.DENIED: COLORS["DENIED"],
        AccessStatus.UNKNOWN: COLORS["UNKNOWN"],
    }.get(status, COLORS["INFO"])

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    panel_h = 70
    panel_y = y - panel_h if y - panel_h > 0 else y + h + 5

    # Bilgi Paneli (Yarı saydam efekt yerine düz siyah maske)
    cv2.rectangle(frame, (x, panel_y), (x + w, panel_y + panel_h), COLORS["BLACK"], -1)
    cv2.rectangle(frame, (x, panel_y), (x + w, panel_y + panel_h), color, 2)

    text_x = x + 8
    display_name = name if name else "Unknown"
    cv2.putText(frame, f"N: {display_name}", (text_x, panel_y + 20), FONT, 0.5, COLORS["WHITE"], 2)
    cv2.putText(frame, f"S: {confidence:.1%}", (text_x, panel_y + 42), FONT, 0.5, COLORS["INFO"], 1)
    cv2.putText(frame, status.value, (text_x, panel_y + 62), FONT, 0.5, color, 2)


def main() -> None:
    create_tables()
    detector, extractor, recognizer, logger = FaceDetector(), EmbeddingExtractor(), FaceRecognizer(), AccessLogger()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    prev_time = time.time()
    frame_count = 0
    cached_faces = []  # AI sonuçlarını saklamak için

    last_decision_time = {}
    COOLDOWN_SEC = 3.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. FPS Hesaplama
        curr_time = time.time()
        fps_actual = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # 2. ANALİZ (Sadece belirli karelerde ve küçük boyutta)
        if frame_count % PROCESS_EVERY_N_FRAME == 0:
            cached_faces = []
            # Görüntüyü küçülterek işle (Hız kazancı)
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            detected_faces = detector.detect_faces(small_frame)

            for face_img, (sx, sy, sw, sh) in detected_faces:
                # Koordinatları orijinal boyuta geri getir
                x, y, w, h = int(sx / SCALE_FACTOR), int(sy / SCALE_FACTOR), int(sw / SCALE_FACTOR), int(
                    sh / SCALE_FACTOR)

                embedding = extractor.extract(face_img)
                if embedding is None: continue

                result = recognizer.recognize(embedding)
                user_key = result.matched_user.id if result.matched_user else "unknown"

                # Cooldown ve Loglama Mantığı
                now = time.time()
                if user_key not in last_decision_time or (now - last_decision_time[user_key] > COOLDOWN_SEC):
                    decision = logger.log_access(result)
                    last_decision_time[user_key] = now
                else:
                    status = AccessStatus.UNKNOWN
                    if result.is_recognized:
                        status = AccessStatus.GRANTED if result.matched_user.is_authorized else AccessStatus.DENIED
                    decision = AccessDecision(status=status, user=result.matched_user, confidence=result.confidence,
                                              message="")

                cached_faces.append({"coords": (x, y, w, h), "dec": decision})

        # 3. ÇİZİM (Her karede cached sonuçları bas)
        for face in cached_faces:
            x, y, w, h = face["coords"]
            d = face["dec"]
            draw_box_with_info(frame, x, y, w, h, name=d.user.name if d.user else "", confidence=d.confidence,
                               status=d.status)

        # Header ve UI
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(frame, f"AI ACCESS | FPS: {fps_actual:.1f}", (15, 22), FONT, 0.6, (255, 255, 255), 2)

        cv2.imshow("Biometric Access Control", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()