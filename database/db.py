import sqlite3
import os
import threading
from contextlib import contextmanager

from config.settings import DATABASE_PATH


class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        self._initialized = True

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


db_manager = DatabaseManager()