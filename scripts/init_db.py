import sys
import os

# Proje kök dizinini path'a ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import db_manager




def create_tables() -> None:
    create_users = """
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL,
        embedding     BLOB NOT NULL,
        is_authorized INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT NOT NULL
    );
    """

    create_logs = """
    CREATE TABLE IF NOT EXISTS access_logs (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id        INTEGER,
        confidence     REAL NOT NULL,
        access_granted INTEGER NOT NULL,
        timestamp      TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """

    create_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON access_logs(timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_logs_user_id   ON access_logs(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_users_name     ON users(name);",
    ]

    with db_manager.get_connection() as conn:
        conn.execute(create_users)
        conn.execute(create_logs)
        for idx_sql in create_indexes:
            conn.execute(idx_sql)

    print("✅ Tüm tablo ve index'ler oluşturuldu.")


def verify_schema() -> None:
    with db_manager.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        print("\n📋 Mevcut tablo")
        print("─" * 40)
        for t in tables:
            print(f"  • {t['name']}")
            columns = conn.execute(f"PRAGMA table_info({t['name']})")
            for col in columns:
                print(f"      {col['name']:20s} {col['type']}")
        print()


if __name__ == "__main__":
    print("🗄️  Veritabanı başlatılıyor...\n")
    create_tables()
    verify_schema()
    print("✅ init_db.py tamamlandı.")