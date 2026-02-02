from datetime import datetime
from typing import Optional, List
import numpy as np

from face_access_system.database.db import db_manager
from face_access_system.database.models import User, AccessLog


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def create_user(name: str, embedding: np.ndarray, is_authorized: bool = True) -> User:
    blob = _embedding_to_blob(embedding)
    now = datetime.now()

    with db_manager.get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (name, embedding, is_authorized, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, blob, int(is_authorized), now.isoformat())
        )
        user_id = cursor.lastrowid

    return User(
        id=user_id,
        name=name,
        embedding=embedding,
        is_authorized=is_authorized,
        created_at=now
    )


def get_user_by_id(user_id: int) -> Optional[User]:
    with db_manager.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()

    if row is None:
        return None

    return User(
        id=row["id"],
        name=row["name"],
        embedding=_blob_to_embedding(row["embedding"]),
        is_authorized=bool(row["is_authorized"]),
        created_at=datetime.fromisoformat(row["created_at"])
    )


def get_all_users() -> List[User]:
    with db_manager.get_connection() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()

    return [
        User(
            id=row["id"],
            name=row["name"],
            embedding=_blob_to_embedding(row["embedding"]),
            is_authorized=bool(row["is_authorized"]),
            created_at=datetime.fromisoformat(row["created_at"])
        )
        for row in rows
    ]


def update_user_authorization(user_id: int, is_authorized: bool) -> None:
    with db_manager.get_connection() as conn:
        conn.execute(
            "UPDATE users SET is_authorized = ? WHERE id = ?",
            (int(is_authorized), user_id)
        )


def delete_user(user_id: int) -> None:
    with db_manager.get_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def create_access_log(
    user_id: Optional[int],
    confidence: float,
    access_granted: bool
) -> AccessLog:
    now = datetime.now()

    with db_manager.get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO access_logs (user_id, confidence, access_granted, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, round(confidence, 4), int(access_granted), now.isoformat())
        )
        log_id = cursor.lastrowid

    return AccessLog(
        id=log_id,
        user_id=user_id,
        confidence=confidence,
        access_granted=access_granted,
        timestamp=now
    )


def get_access_logs(limit: int = 50) -> List[AccessLog]:
    with db_manager.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return [
        AccessLog(
            id=row["id"],
            user_id=row["user_id"],
            confidence=row["confidence"],
            access_granted=bool(row["access_granted"]),
            timestamp=datetime.fromisoformat(row["timestamp"])
        )
        for row in rows
    ]


def get_logs_by_user(user_id: int, limit: int = 20) -> List[AccessLog]:
    with db_manager.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM access_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

    return [
        AccessLog(
            id=row["id"],
            user_id=row["user_id"],
            confidence=row["confidence"],
            access_granted=bool(row["access_granted"]),
            timestamp=datetime.fromisoformat(row["timestamp"])
        )
        for row in rows
    ]