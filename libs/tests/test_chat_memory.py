"""
Chat memory tests.

Covers:
- Basic add / clear operations on an unfrozen (live) memory.
- Frozen (pinned) memory: mutations raise RuntimeError; reads still work.
- Snapshot produces an independent frozen copy.
- Rolling window via max_messages.
- to_dict serialisation round-trip.
- Edge cases: empty memory, single message, invalid arguments.
"""

import pytest

from libs.ragsearch.chat_memory import ChatMemory, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill(mem: ChatMemory, n: int) -> None:
    for i in range(n):
        mem.add("user" if i % 2 == 0 else "assistant", f"message {i}")


# ---------------------------------------------------------------------------
# Unfrozen (live) memory – basic operations
# ---------------------------------------------------------------------------

class TestUnfrozenMemory:
    def test_new_memory_is_empty(self):
        mem = ChatMemory()
        assert len(mem) == 0
        assert mem.messages() == []

    def test_add_single_message(self):
        mem = ChatMemory()
        mem.add("user", "Hello")
        assert len(mem) == 1
        assert mem.messages()[0].role == "user"
        assert mem.messages()[0].content == "Hello"

    def test_add_multiple_messages_preserves_order(self):
        mem = ChatMemory()
        mem.add("user", "first")
        mem.add("assistant", "second")
        mem.add("user", "third")
        roles = [m.role for m in mem.messages()]
        assert roles == ["user", "assistant", "user"]

    def test_clear_removes_all_messages(self):
        mem = ChatMemory()
        _fill(mem, 5)
        mem.clear()
        assert len(mem) == 0

    def test_not_frozen_by_default(self):
        assert ChatMemory().frozen is False

    def test_messages_returns_copy(self):
        mem = ChatMemory()
        mem.add("user", "hi")
        copy = mem.messages()
        copy.clear()
        assert len(mem) == 1  # original unaffected

    def test_system_role_accepted(self):
        mem = ChatMemory()
        mem.add("system", "You are a helpful assistant.")
        assert mem.messages()[0].role == "system"


# ---------------------------------------------------------------------------
# Frozen / pinned memory
# ---------------------------------------------------------------------------

class TestFrozenMemory:
    def test_frozen_memory_raises_on_add(self):
        mem = ChatMemory(frozen=True)
        with pytest.raises(RuntimeError, match="frozen"):
            mem.add("user", "test")

    def test_frozen_memory_raises_on_clear(self):
        mem = ChatMemory(frozen=True)
        with pytest.raises(RuntimeError, match="frozen"):
            mem.clear()

    def test_frozen_property_is_true(self):
        mem = ChatMemory(frozen=True)
        assert mem.frozen is True

    def test_freeze_makes_live_memory_immutable(self):
        mem = ChatMemory()
        mem.add("user", "before freeze")
        mem.freeze()
        assert mem.frozen is True
        with pytest.raises(RuntimeError, match="frozen"):
            mem.add("user", "after freeze")

    def test_freeze_preserves_existing_messages(self):
        mem = ChatMemory()
        mem.add("user", "kept")
        mem.freeze()
        assert len(mem) == 1
        assert mem.messages()[0].content == "kept"

    def test_frozen_memory_can_read_messages(self):
        mem = ChatMemory(frozen=True)
        # Reading must not raise
        assert mem.messages() == []
        assert len(mem) == 0

    def test_freeze_is_idempotent(self):
        mem = ChatMemory()
        mem.freeze()
        mem.freeze()  # must not raise
        assert mem.frozen is True


# ---------------------------------------------------------------------------
# Snapshot (pinned copy)
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_is_frozen(self):
        mem = ChatMemory()
        mem.add("user", "original")
        snap = mem.snapshot()
        assert snap.frozen is True

    def test_snapshot_contains_messages_at_time_of_call(self):
        mem = ChatMemory()
        mem.add("user", "m1")
        mem.add("assistant", "m2")
        snap = mem.snapshot()
        assert len(snap) == 2
        assert snap.messages()[0].content == "m1"
        assert snap.messages()[1].content == "m2"

    def test_snapshot_is_independent_of_original(self):
        mem = ChatMemory()
        mem.add("user", "before")
        snap = mem.snapshot()
        mem.add("user", "after")  # live memory grows
        assert len(snap) == 1  # snapshot unchanged

    def test_snapshot_of_empty_memory_is_empty_and_frozen(self):
        mem = ChatMemory()
        snap = mem.snapshot()
        assert snap.frozen is True
        assert len(snap) == 0

    def test_snapshot_raises_on_add(self):
        snap = ChatMemory().snapshot()
        with pytest.raises(RuntimeError, match="frozen"):
            snap.add("user", "no")

    def test_snapshot_raises_on_clear(self):
        snap = ChatMemory().snapshot()
        with pytest.raises(RuntimeError, match="frozen"):
            snap.clear()

    def test_multiple_snapshots_are_independent(self):
        mem = ChatMemory()
        mem.add("user", "one")
        snap1 = mem.snapshot()
        mem.add("user", "two")
        snap2 = mem.snapshot()
        assert len(snap1) == 1
        assert len(snap2) == 2


# ---------------------------------------------------------------------------
# Rolling window (max_messages)
# ---------------------------------------------------------------------------

class TestMaxMessages:
    def test_max_messages_limits_history(self):
        mem = ChatMemory(max_messages=3)
        _fill(mem, 5)
        assert len(mem) == 3

    def test_max_messages_keeps_most_recent(self):
        mem = ChatMemory(max_messages=2)
        mem.add("user", "first")
        mem.add("user", "second")
        mem.add("user", "third")
        contents = [m.content for m in mem.messages()]
        assert contents == ["second", "third"]

    def test_max_messages_none_means_unlimited(self):
        mem = ChatMemory(max_messages=None)
        _fill(mem, 100)
        assert len(mem) == 100

    def test_max_messages_one_keeps_last(self):
        mem = ChatMemory(max_messages=1)
        mem.add("user", "a")
        mem.add("user", "b")
        assert mem.messages()[0].content == "b"

    def test_invalid_max_messages_raises(self):
        with pytest.raises(ValueError):
            ChatMemory(max_messages=0)

    def test_negative_max_messages_raises(self):
        with pytest.raises(ValueError):
            ChatMemory(max_messages=-1)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_returns_list(self):
        mem = ChatMemory()
        assert isinstance(mem.to_dict(), list)

    def test_to_dict_empty_memory(self):
        assert ChatMemory().to_dict() == []

    def test_to_dict_has_role_and_content_keys(self):
        mem = ChatMemory()
        mem.add("user", "hi")
        result = mem.to_dict()
        assert result[0].keys() == {"role", "content"}

    def test_to_dict_preserves_order_and_values(self):
        mem = ChatMemory()
        mem.add("user", "q1")
        mem.add("assistant", "a1")
        result = mem.to_dict()
        assert result == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]

    def test_to_dict_snapshot(self):
        mem = ChatMemory()
        mem.add("user", "snap me")
        snap = mem.snapshot()
        result = snap.to_dict()
        assert result == [{"role": "user", "content": "snap me"}]


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

class TestMessageDataclass:
    def test_message_has_role_and_content(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_message_equality(self):
        assert Message("user", "x") == Message("user", "x")
        assert Message("user", "x") != Message("assistant", "x")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_content_allowed(self):
        mem = ChatMemory()
        mem.add("user", "")
        assert mem.messages()[0].content == ""

    def test_long_content_stored_correctly(self):
        long_text = "word " * 10_000
        mem = ChatMemory()
        mem.add("user", long_text)
        assert mem.messages()[0].content == long_text

    def test_add_after_clear_works(self):
        mem = ChatMemory()
        mem.add("user", "before")
        mem.clear()
        mem.add("user", "after")
        assert len(mem) == 1
        assert mem.messages()[0].content == "after"

    def test_unicode_content(self):
        mem = ChatMemory()
        mem.add("user", "こんにちは 🌍")
        assert mem.messages()[0].content == "こんにちは 🌍"
