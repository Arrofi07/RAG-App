# storage.py
#
# SQLite-backed persistence for user profiles, conversations, messages,
# auth credentials, and document deduplication records.
#
# WHAT CHANGED (v10)
# ──────────────────
# 1. AUTH COLUMNS on the users table:
#      email         — unique, used for login
#      password_hash — bcrypt hash, never plain text
#      is_verified   — False until email verification is added (SMTP TODO)
#      is_active     — soft-disable accounts without deleting data
#
#    The existing `name` + `profile` columns are unchanged. All existing
#    rows get NULL for the new auth columns and remain readable; auth is
#    only enforced on new registrations and API endpoints.
#
#    Migration strategy: ALTER TABLE ... ADD COLUMN is used to non-destructively
#    add columns to an existing DB. Safe to run on a DB that already has users.
#
# 2. DOCUMENTS TABLE for deduplication:
#      filename_hash  — SHA-256 of the lowercase filename
#      content_hash   — SHA-256 of the raw file bytes
#    Both are checked independently before ingestion. The table also stores
#    metadata (chunks ingested, file size, upload time) useful for the UI.
#
# DATABASE LAYOUT (v10)
# ─────────────────────
#  users         — auth + profile, one row per registered user
#  conversations — one per chat session, belongs to a user
#  messages      — individual turns, belong to a conversation
#  documents     — one row per successfully ingested PDF (for deduplication)

import json
import sqlite3
import uuid
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/chatbot.db")


class Storage:

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._migrate()       # safely adds new columns to existing DBs

    # ──────────────────────────────────────────────────────────────────────────
    # Connection helper
    # ──────────────────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Yield a connection with WAL mode and foreign keys enabled."""
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

    # ──────────────────────────────────────────────────────────────────────────
    # Schema
    # ──────────────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """
        Create tables that don't exist yet. Safe to call on an existing DB.

        IMPORTANT: auth columns (email, password_hash, is_verified, is_active)
        are NOT listed here — they are added by _migrate() which runs immediately
        after. This means both code paths (fresh DB and existing DB) go through
        _migrate(), so there's no race between CREATE TABLE and ALTER TABLE.

        The idx_users_email partial index is also created in _migrate() after
        the email column is guaranteed to exist.
        """
        with self._conn() as con:
            con.executescript("""
                -- Users table: original columns only.
                -- Auth columns added by _migrate() below.
                CREATE TABLE IF NOT EXISTS users (
                    id         TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    profile    TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL REFERENCES users(id),
                    title       TEXT NOT NULL DEFAULT 'New conversation',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conv_user
                    ON conversations(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id              TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_msg_conv
                    ON messages(conversation_id, created_at ASC);

                -- Documents table for deduplication.
                -- filename_hash and content_hash are checked independently
                -- before any ingestion (see check_duplicate()).
                CREATE TABLE IF NOT EXISTS documents (
                    id             TEXT PRIMARY KEY,
                    filename       TEXT NOT NULL,
                    filename_hash  TEXT NOT NULL UNIQUE,
                    content_hash   TEXT NOT NULL UNIQUE,
                    file_size      INTEGER NOT NULL,
                    chunks_count   INTEGER NOT NULL,
                    category       TEXT,
                    author         TEXT,
                    year           INTEGER,
                    tags           TEXT,
                    uploaded_by    TEXT REFERENCES users(id),
                    ingested_at    TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_docs_content_hash
                    ON documents(content_hash);
                CREATE INDEX IF NOT EXISTS idx_docs_filename_hash
                    ON documents(filename_hash);
            """)

    def _migrate(self) -> None:
        """
        Non-destructive migration: add auth columns to the users table if they
        don't exist yet. This is the ONLY place these columns are created —
        runs on both fresh DBs (immediately after _init_schema creates the
        bare table) and existing DBs (adds columns to the already-existing table).

        SQLite doesn't support IF NOT EXISTS on ALTER TABLE, so we check
        the column list with PRAGMA table_info before each ALTER.
        """
        with self._conn() as con:
            existing = {
                row[1]
                for row in con.execute("PRAGMA table_info(users)").fetchall()
            }

            new_columns = {
                "email":         "TEXT",
                "password_hash": "TEXT",
                "is_verified":   "INTEGER DEFAULT 0",
                "is_active":     "INTEGER DEFAULT 1",
            }
            for col, definition in new_columns.items():
                if col not in existing:
                    con.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
                    log.info("DB migration: added column users.%s", col)

            # Partial unique index on email — created here (not in _init_schema)
            # so the email column is guaranteed to exist first.
            con.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email
                    ON users(email)
                    WHERE email IS NOT NULL
            """)

    # ──────────────────────────────────────────────────────────────────────────
    # Auth — registration and login
    # ──────────────────────────────────────────────────────────────────────────

    def register_user(
        self,
        name:          str,
        email:         str,
        password_hash: str,
        profile:       dict | None = None,
    ) -> str:
        """
        Create a new user with auth credentials. Returns the new user ID.
        Raises ValueError if the email is already registered.

        The password must be hashed BEFORE calling this method — never pass
        a plain-text password to the DB layer.
        """
        email = email.lower().strip()  # normalise email to lowercase
        uid   = str(uuid.uuid4())
        now   = _now()

        try:
            with self._conn() as con:
                con.execute(
                    """INSERT INTO users
                       (id, name, email, password_hash, is_verified, is_active,
                        profile, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 0, 1, ?, ?, ?)""",
                    (uid, name, email, password_hash,
                     json.dumps(profile or {}), now, now),
                )
        except sqlite3.IntegrityError:
            # UNIQUE constraint on email failed
            raise ValueError(f"Email '{email}' is already registered.")

        log.info("Registered new user '%s' (%s)", name, uid)
        return uid

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """
        Fetch a user row by email (used during login).
        Returns None if not found. Includes password_hash for verification.
        """
        email = email.lower().strip()
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

        if row is None:
            return None

        return {**dict(row), "profile": json.loads(row["profile"])}

    def get_user(self, user_id: str) -> Optional[dict]:
        """Fetch a user by ID. Returns None if not found."""
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()

        if row is None:
            return None

        return {**dict(row), "profile": json.loads(row["profile"])}

    def list_users(self) -> list[dict]:
        """Return all users (ID, name, email, created_at). No password hashes."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, name, email, created_at FROM users ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_profile(self, user_id: str, profile: dict) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE users SET profile = ?, updated_at = ? WHERE id = ?",
                (json.dumps(profile), _now(), user_id),
            )

    def set_user_verified(self, user_id: str) -> None:
        """Mark email as verified (call this from the email verification flow later)."""
        with self._conn() as con:
            con.execute(
                "UPDATE users SET is_verified = 1, updated_at = ? WHERE id = ?",
                (_now(), user_id),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Conversations
    # ──────────────────────────────────────────────────────────────────────────

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

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
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
        """Bump updated_at so the conversation floats to the top of the list."""
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

    # ──────────────────────────────────────────────────────────────────────────
    # Messages
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Document deduplication (v10)
    # ──────────────────────────────────────────────────────────────────────────

    def check_duplicate(
        self,
        filename:     str,
        content_hash: str,
    ) -> Optional[dict]:
        """
        Check whether a file is a duplicate BEFORE ingesting it.

        Returns a dict describing the conflict if a duplicate is found, else None.

        Two independent checks:
          1. filename_hash — same lowercase filename was ingested before
          2. content_hash  — identical file bytes (even under a different name)

        We check content_hash first because it's the stronger signal —
        re-uploading the same content under a different name is more
        confusing than re-uploading by filename.

        The returned dict always includes a human-readable 'reason' string
        suitable for showing in the UI.
        """
        filename_hash = _sha256(filename.lower())

        with self._conn() as con:
            # Check 1: same content bytes (strongest duplicate signal)
            row = con.execute(
                "SELECT filename, ingested_at FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if row:
                return {
                    "duplicate":    True,
                    "type":         "content",
                    "reason":       (
                        f"This file's content is identical to "
                        f"'{row['filename']}' which was ingested on "
                        f"{row['ingested_at'][:10]}."
                    ),
                    "original_filename": row["filename"],
                    "ingested_at":       row["ingested_at"],
                }

            # Check 2: same filename (case-insensitive)
            row = con.execute(
                "SELECT filename, ingested_at FROM documents WHERE filename_hash = ?",
                (filename_hash,),
            ).fetchone()
            if row:
                return {
                    "duplicate":    True,
                    "type":         "filename",
                    "reason":       (
                        f"A file named '{row['filename']}' was already ingested "
                        f"on {row['ingested_at'][:10]}. "
                        f"Delete the existing document first if you want to replace it."
                    ),
                    "original_filename": row["filename"],
                    "ingested_at":       row["ingested_at"],
                }

        return None   # no duplicate found — safe to ingest

    def record_document(
        self,
        filename:     str,
        content_hash: str,
        file_size:    int,
        chunks_count: int,
        category:     Optional[str] = None,
        author:       Optional[str] = None,
        year:         Optional[int] = None,
        tags:         Optional[list] = None,
        uploaded_by:  Optional[str] = None,
    ) -> str:
        """
        Record a successfully ingested document for future deduplication checks.
        Call this AFTER Qdrant upsert succeeds — not before, so a failed
        ingestion doesn't block future re-tries.

        Returns the new document record ID.
        """
        doc_id        = str(uuid.uuid4())
        filename_hash = _sha256(filename.lower())

        with self._conn() as con:
            con.execute(
                """INSERT INTO documents
                   (id, filename, filename_hash, content_hash, file_size,
                    chunks_count, category, author, year, tags,
                    uploaded_by, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id, filename, filename_hash, content_hash,
                    file_size, chunks_count, category, author, year,
                    json.dumps(tags or []),
                    uploaded_by, _now(),
                ),
            )

        log.info(
            "Recorded document '%s' (%d chunks, content_hash=%s…)",
            filename, chunks_count, content_hash[:12],
        )
        return doc_id

    def delete_document_record(self, filename: str) -> bool:
        """
        Remove a document's deduplication record so it can be re-ingested.
        Does NOT remove the chunks from Qdrant — that's the caller's job.
        Returns True if a record was deleted, False if not found.
        """
        filename_hash = _sha256(filename.lower())
        with self._conn() as con:
            cur = con.execute(
                "DELETE FROM documents WHERE filename_hash = ?",
                (filename_hash,),
            )
        return cur.rowcount > 0

    def list_document_records(self) -> list[dict]:
        """Return all ingested document records, newest first."""
        with self._conn() as con:
            rows = con.execute(
                """SELECT id, filename, file_size, chunks_count,
                          category, author, year, tags, uploaded_by, ingested_at
                   FROM documents ORDER BY ingested_at DESC"""
            ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d["tags"] or "[]")
            result.append(d)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _sha256(data: str | bytes) -> str:
    """Return the hex SHA-256 digest of a string or bytes object."""
    import hashlib
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# Expose _sha256 for use in main.py (computing content hash from file bytes)
sha256 = _sha256