import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional
from enum import Enum


from config.settings import LOG_FILE, LOG_FORMAT
from recognition.recognizer import RecognitionResult
from database.crud import create_access_log
from database.models import User


class AccessStatus(Enum):
    GRANTED = "ACCESS GRANTED"
    DENIED = "ACCESS DENIED"
    UNKNOWN = "UNKNOWN PERSON"


@dataclass
class AccessDecision:
    status: AccessStatus
    user: Optional[User]
    confidence: float
    message: str


class AccessLogger:
    def __init__(self):
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Log dizinini ve handler'ları ayarlar."""
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        # Orijinal logging kütüphanesinden bir logger nesnesi oluşturuyoruz
        self.logger = logging.getLogger("access_audit")
        self.logger.setLevel(logging.INFO)

        # Eğer handler'lar yoksa (yeniden başlatmada çift log yazmaması için) ekle
        if not self.logger.handlers:
            formatter = logging.Formatter(LOG_FORMAT)

            # Konsol (Terminal) Logu
            console = logging.StreamHandler()
            console.setFormatter(formatter)

            # Dosya (.log) Logu
            file_h = logging.FileHandler(LOG_FILE)
            file_h.setFormatter(formatter)

            self.logger.addHandler(console)
            self.logger.addHandler(file_h)

    def log_access(self, result: RecognitionResult) -> AccessDecision:
        """Tanıma sonucunu değerlendirir, loglar ve veritabanına işler."""
        user = result.matched_user if result.is_recognized else None
        conf = float(result.confidence)

        if result.is_recognized and user:
            if user.is_authorized:
                self.logger.info(f"[GRANTED] User: {user.name} | Score: {conf:.4f}")
                create_access_log(user_id=user.id, confidence=conf, access_granted=True)
                return AccessDecision(AccessStatus.GRANTED, user, conf, f"Welcome, {user.name}!")

            self.logger.warning(f"[DENIED] Unauthorized Attempt: {user.name}")
            create_access_log(user_id=user.id, confidence=conf, access_granted=False)
            return AccessDecision(AccessStatus.DENIED, user, conf, "Access Denied")

        self.logger.warning(f"[UNKNOWN] Not recognized | Score: {conf:.4f}")
        create_access_log(user_id=None, confidence=conf, access_granted=False)
        return AccessDecision(AccessStatus.UNKNOWN, None, conf, "Unknown Face")