import os

# ─── Kamera ────────────────────────────────────────
CAMERA_INDEX: int = 0
FRAME_WIDTH: int  = 640
FRAME_HEIGHT: int = 480
FPS: int = 30

# ─── Dizini Hesaplama ──────────────────────────────
# 1. Bu dosyanın yeri: PythonProject/face_access_system/config/settings.py
# 2. Bir üstü: PythonProject/face_access_system/
# 3. İki üstü: PythonProject/ (İşte modellerin olduğu yer)
CURRENT_FILE_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

# ─── Yüz Algılama (Modeller PythonProject klasöründe) ───
DLIB_PREDICTOR_PATH: str = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
DLIB_RECOGNITION_MODEL_PATH: str = os.path.join(BASE_DIR, "dlib_face_recognition_resnet_model_v1.dat")

# ─── Embedding & Benzerlik ─────────────────────────
EMBEDDING_DIM: int = 128
SIMILARITY_THRESHOLD: float = 0.6

# ─── Veritabanı & Log (Veriler face_access_system/data içinde kalsın) ───
# Verileri proje klasörünün içine kaydedelim
PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))
DATABASE_PATH: str = os.path.join(PROJECT_DIR, "data", "biometric.db")
LOG_FILE: str = os.path.join(PROJECT_DIR, "data", "access.log")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"

# Debug: Dosyaları bulabiliyor mu kontrol etmek için print ekleyelim
if not os.path.exists(DLIB_PREDICTOR_PATH):
    print(f"⚠️ KRİTİK HATA: Model dosyası bulunamadı! Aranan konum: {DLIB_PREDICTOR_PATH}")