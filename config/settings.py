"""
face_access_system / config / settings.py
─────────────────────────────────────────
Sistem genelinde kullanılan sabitler ve ayarlar.
Tek bir yerden yönetilir → kolay bakım, kolay test.
"""

import os

# ─── Kamera ────────────────────────────────────────
CAMERA_INDEX: int = 0          # Sistem kamerası (0 = varsayılan)
FRAME_WIDTH: int  = 640
FRAME_HEIGHT: int = 480
FPS: int = 30

# ─── Proje dizinleri ───────────────────────────────
# face_access_system/ klasörü
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PythonProject/ kök dizini (bir üst klasör)
PROJECT_ROOT: str = os.path.dirname(BASE_DIR)

# ─── Yüz Algılama ──────────────────────────────────
# Dlib model dosyaları PythonProject/ kök dizininde
DLIB_PREDICTOR_PATH: str = os.path.join(PROJECT_ROOT, "shape_predictor_68_face_landmarks.dat")
DLIB_RECOGNITION_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "dlib_face_recognition_resnet_model_v1.dat")

# ─── Embedding & Benzerlik ─────────────────────────
EMBEDDING_DIM: int = 128       # Dlib → 128 boyutlu vektör
SIMILARITY_THRESHOLD: float = 0.6   # Cosine similarity eşiği

# ─── Veritabanı ────────────────────────────────────
DATABASE_PATH: str = os.path.join(BASE_DIR, "data", "biometric.db")

# ─── Loglama ───────────────────────────────────────
LOG_FILE: str = os.path.join(BASE_DIR, "data", "access.log")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"