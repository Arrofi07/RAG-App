# tests/test_storage.py
#
# Tests for storage.py — user/profile/conversation/message CRUD.
# Uses an in-memory SQLite file via the `in_memory_storage` fixture.
# No network calls, no external services.

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

class TestUsers:

    def test_create_and_get_user(self, in_memory_storage):
        db  = in_memory_storage
        uid = db.create_user("Alice", {"nationality": "German"})
        user = db.get_user(uid)
        assert user is not None
        assert user["name"] == "Alice"
        assert user["profile"]["nationality"] == "German"

    def test_get_nonexistent_user_returns_none(self, in_memory_storage):
        db = in_memory_storage
        assert db.get_user("does-not-exist") is None

    def test_list_users_returns_all(self, in_memory_storage):
        db = in_memory_storage
        db.create_user("Alice")
        db.create_user("Bob")
        users = db.list_users()
        names = [u["name"] for u in users]
        assert "Alice" in names
        assert "Bob" in names

    def test_update_profile(self, in_memory_storage):
        db  = in_memory_storage
        uid = db.create_user("Carol", {"field_of_study": "Physics"})
        db.update_profile(uid, {"field_of_study": "Computer Science"})
        user = db.get_user(uid)
        assert user["profile"]["field_of_study"] == "Computer Science"

    def test_update_profile_replaces_entirely(self, in_memory_storage):
        """update_profile stores the whole new dict, not a merge."""
        db  = in_memory_storage
        uid = db.create_user("Dave", {"field_of_study": "Physics", "german_level": "B2"})
        db.update_profile(uid, {"field_of_study": "Biology"})   # no german_level
        user = db.get_user(uid)
        assert "german_level" not in user["profile"]

    def test_two_users_have_different_ids(self, in_memory_storage):
        db = in_memory_storage
        uid1 = db.create_user("Eve")
        uid2 = db.create_user("Frank")
        assert uid1 != uid2


# ─────────────────────────────────────────────────────────────────────────────
# Conversations
# ─────────────────────────────────────────────────────────────────────────────

class TestConversations:

    def _user(self, db, name="TestUser"):
        return db.create_user(name)

    def test_create_conversation(self, in_memory_storage):
        db  = in_memory_storage
        uid = self._user(db)
        cid = db.create_conversation(uid, "My first chat")
        assert cid is not None

    def test_list_conversations_by_user(self, in_memory_storage):
        db   = in_memory_storage
        uid  = self._user(db)
        cid1 = db.create_conversation(uid, "Chat A")
        cid2 = db.create_conversation(uid, "Chat B")
        convs = db.list_conversations(uid)
        ids = [c["id"] for c in convs]
        assert cid1 in ids
        assert cid2 in ids

    def test_conversations_ordered_newest_first(self, in_memory_storage):
        import time
        db  = in_memory_storage
        uid = self._user(db)
        cid1 = db.create_conversation(uid, "First")
        time.sleep(0.01)     # ensure different timestamps
        cid2 = db.create_conversation(uid, "Second")
        db.add_message(cid2, "user", "hello")   # touch cid2 to bump its updated_at

        convs = db.list_conversations(uid)
        assert convs[0]["id"] == cid2   # most recently updated first

    def test_update_conversation_title(self, in_memory_storage):
        db  = in_memory_storage
        uid = self._user(db)
        cid = db.create_conversation(uid)
        db.update_conversation_title(cid, "Updated Title")
        conv = db.get_conversation(cid)
        assert conv["title"] == "Updated Title"

    def test_title_truncated_to_80_chars(self, in_memory_storage):
        db  = in_memory_storage
        uid = self._user(db)
        cid = db.create_conversation(uid)
        long_title = "x" * 200
        db.update_conversation_title(cid, long_title)
        conv = db.get_conversation(cid)
        assert len(conv["title"]) <= 80

    def test_delete_conversation(self, in_memory_storage):
        db  = in_memory_storage
        uid = self._user(db)
        cid = db.create_conversation(uid, "To be deleted")
        db.delete_conversation(cid)
        convs = db.list_conversations(uid)
        assert all(c["id"] != cid for c in convs)

    def test_delete_conversation_also_deletes_messages(self, in_memory_storage):
        db  = in_memory_storage
        uid = self._user(db)
        cid = db.create_conversation(uid)
        db.add_message(cid, "user",      "Hello")
        db.add_message(cid, "assistant", "Hi there!")
        db.delete_conversation(cid)
        msgs = db.get_messages(cid)
        assert msgs == []


# ─────────────────────────────────────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────────────────────────────────────

class TestMessages:

    def _conv(self, db):
        uid = db.create_user("MsgUser")
        return db.create_conversation(uid, "Test conversation")

    def test_add_and_get_messages(self, in_memory_storage):
        db  = in_memory_storage
        cid = self._conv(db)
        db.add_message(cid, "user",      "What is a Studienkolleg?")
        db.add_message(cid, "assistant", "A Studienkolleg is a preparatory course...")
        msgs = db.get_messages(cid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_messages_in_chronological_order(self, in_memory_storage):
        db  = in_memory_storage
        cid = self._conv(db)
        db.add_message(cid, "user",      "First message")
        db.add_message(cid, "user",      "Second message")
        db.add_message(cid, "assistant", "Reply")
        msgs = db.get_messages(cid)
        assert msgs[0]["content"] == "First message"
        assert msgs[-1]["content"] == "Reply"

    def test_add_message_updates_conversation_timestamp(self, in_memory_storage):
        import time
        db  = in_memory_storage
        uid = db.create_user("TsUser")
        cid = db.create_conversation(uid)
        ts_before = db.get_conversation(cid)["updated_at"]
        time.sleep(0.02)
        db.add_message(cid, "user", "hello")
        ts_after = db.get_conversation(cid)["updated_at"]
        assert ts_after > ts_before

    def test_messages_for_different_conversations_do_not_mix(self, in_memory_storage):
        db   = in_memory_storage
        uid  = db.create_user("IsoUser")
        cid1 = db.create_conversation(uid, "Conv 1")
        cid2 = db.create_conversation(uid, "Conv 2")
        db.add_message(cid1, "user", "Message in conv 1")
        db.add_message(cid2, "user", "Message in conv 2")
        assert len(db.get_messages(cid1)) == 1
        assert len(db.get_messages(cid2)) == 1
        assert db.get_messages(cid1)[0]["content"] == "Message in conv 1"