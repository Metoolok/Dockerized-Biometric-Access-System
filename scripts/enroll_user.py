import sys
import os
import time
import cv2
import numpy as np


# Dosyanın konumundan bağımsız olarak kök dizini (PythonProject) sisteme tanıtır.
current_file = os.path.abspath(__file__)
# scripts -> face_access_system -> PythonProject (3 basamak yukarı)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --------------------------------------------------

from face_access_system.config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
from face_access_system.vision.face_detector import FaceDetector
from face_access_system.vision.embedding_extractor import EmbeddingExtractor
from face_access_system.database.crud import create_user, get_all_users
from face_access_system.scripts.init_db import create_tables

def get_user_name() -> str:
    print("\n" + "=" * 50)
    print("  🎯 YUZ KAYIT SISTEMI (Enrollment)")
    print("=" * 50)
    name = input("\n👤 Kaydedilecek Kullanıcı Adı: ").strip()
    if not name:
        print("❌ Ad boş bırakılamaz.")
        sys.exit(1)
    return name

def capture_face(detector: FaceDetector) -> np.ndarray:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ Kamera açılamadı! Lütfen bağlantıyı kontrol edin.")
        sys.exit(1)

    print("\n📷 Kamera açıldı. Lütfen yüzünüzü net gösterin...")
    print("   (Yüz algılandığında 2 saniye içinde otomatik çekilecek)\n")

    face_detected_time = None
    captured_face = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        faces = detector.detect_faces(frame)
        display_frame = frame.copy()

        if faces:
            # Sadece ilk tespit edilen yüzü al
            face_img, (x, y, w, h) = faces[0]

            if face_detected_time is None:
                face_detected_time = time.time()
                print("   ✅ Yüz tespit edildi! Sabit bekleyin...")

            # Görsel geri bildirim
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display_frame, "BEKLEYIN...", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 2 saniye dolduysa yakala
            if time.time() - face_detected_time >= 2.0:
                captured_face = face_img
                cv2.putText(display_frame, "YAKALANDI!", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.imshow("Enrollment", display_frame)
                cv2.waitKey(1000) # 1 saniye sonucu göster
                break
        else:
            face_detected_time = None
            cv2.putText(display_frame, "Yuz aranıyor...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Enrollment", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    return captured_face

def enroll() -> None:
    # Veritabanı tabloları yoksa oluştur
    create_tables()

    name = get_user_name()

    print("\n🔧 AI Modelleri yukleniyor (Dlib)...")
    detector  = FaceDetector()
    extractor = EmbeddingExtractor()

    face_img = capture_face(detector)

    if face_img is None:
        print("❌ Yüz yakalanamadı.")
        return

    print("\n🧠 Biometrik imza (Embedding) çıkarılıyor...")
    embedding = extractor.extract(face_img)

    if embedding is None:
        print("❌ Hata: Embedding oluşturulamadı!")
        sys.exit(1)

    print("\n💾 Veritabanına kaydediliyor...")
    create_user(name=name, embedding=embedding, is_authorized=True)

    print("\n" + "=" * 50)
    print(f"  ✅ KAYIT BAŞARILI!")
    print(f"     Kullanıcı:  {name}")
    print(f"     Durum:      Sisteme Giriş Yetkisi Verildi")
    print("=" * 50)

    try:
        all_users = get_all_users()
        print(f"👥 Sistemdeki toplam kullanıcı sayısı: {len(all_users)}")
    except:
        pass

if __name__ == "__main__":
    enroll()