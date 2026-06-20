# storage.py
#
# SQLite-backed persistence for user profiles, conversations, and messages.
# Uses only the Python standard library — no extra service or dependency needed.
#
# Database layout
# ───────────────
#  users         — one row per user, profile stored as JSON blob
#  conversations — one per chat session, belongs to a user
#  messages      — individual turns, belong to a conversation

import json
import sqlite3
import uuid
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/chatbot.db")


class Storage:

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        """Yield a connection with WAL mode (safe for concurrent reads)."""
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    profile     TEXT NOT NULL DEFAULT '{}',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL REFERENCES users(id),
                    title       TEXT NOT NULL DEFAULT 'New conversation',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conv_user
                    ON conversations(user_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_msg_conv
                    ON messages(conversation_id, created_at ASC);
            """)

    # ------------------------------------------------------------------
    # Users / profiles
    # ------------------------------------------------------------------

    def create_user(self, name: str, profile: dict | None = None) -> str:
        """Create a new user and return their ID."""
        uid  = str(uuid.uuid4())
        now  = _now()
        data = json.dumps(profile or {})

        with self._conn() as con:
            con.execute(
                "INSERT INTO users (id, name, profile, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (uid, name, data, now, now),
            )

        log.info("Created user '%s' (%s)", name, uid)
        return uid

    def get_user(self, user_id: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()

        if row is None:
            return None

        return {**dict(row), "profile": json.loads(row["profile"])}

    def list_users(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, name, created_at FROM users ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_profile(self, user_id: str, profile: dict) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE users SET profile = ?, updated_at = ? WHERE id = ?",
                (json.dumps(profile), _now(), user_id),
            )

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def create_conversation(self, user_id: str, title: str = "New conversation") -> str:
        cid = str(uuid.uuid4())
        now = _now()

        with self._conn() as con:
            con.execute(
                "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, user_id, title, now, now),
            )

        return cid

    def list_conversations(self, user_id: str) -> list[dict]:
        """Return all conversations for a user, newest first."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations WHERE user_id = ? "
                "ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title[:80], _now(), conversation_id),
            )

    def touch_conversation(self, conversation_id: str) -> None:
        """Update `updated_at` so the conversation floats to the top of the list."""
        with self._conn() as con:
            con.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn() as con:
            con.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            con.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(self, conversation_id: str, role: str, content: str) -> str:
        mid = str(uuid.uuid4())
        now = _now()

        with self._conn() as con:
            con.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (mid, conversation_id, role, content, now),
            )

        self.touch_conversation(conversation_id)
        return mid

    def get_messages(self, conversation_id: str) -> list[dict]:
        """Return all messages in a conversation, oldest first."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat()