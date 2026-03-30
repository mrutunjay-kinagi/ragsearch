"""
chat_memory.py – In-process chat history with frozen / pinned snapshot support.

Usage::

    from libs.ragsearch.chat_memory import ChatMemory

    mem = ChatMemory(max_messages=20)
    mem.add("user", "What is RAG?")
    mem.add("assistant", "RAG stands for Retrieval-Augmented Generation.")

    # Freeze a snapshot (pinned; immutable)
    pinned = mem.snapshot()

    # Further messages go into the live memory only
    mem.add("user", "How does it work?")

    # Frozen memory raises on mutation
    pinned.add("user", "...")  # → RuntimeError
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single turn in a conversation."""

    role: str    # "user" | "assistant" | "system"
    content: str


class ChatMemory:
    """Maintains an ordered conversation history.

    When *frozen* is ``True`` the instance acts as a read-only pinned
    snapshot: any attempt to add or clear messages raises :class:`RuntimeError`.

    Args:
        frozen:       Start in frozen/pinned mode.  Defaults to ``False``.
        max_messages: Rolling window size.  When set, only the last
                      *max_messages* messages are kept after each :meth:`add`.
                      ``None`` means unlimited.
    """

    def __init__(
        self,
        frozen: bool = False,
        max_messages: Optional[int] = None,
    ) -> None:
        if max_messages is not None and max_messages <= 0:
            raise ValueError("max_messages must be a positive integer or None")
        self._messages: List[Message] = []
        self._frozen = frozen
        self.max_messages = max_messages

    # ------------------------------------------------------------------
    # Freeze / thaw
    # ------------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        """``True`` when this memory is pinned (immutable)."""
        return self._frozen

    def freeze(self) -> None:
        """Pin this memory; no further mutations are allowed."""
        self._frozen = True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a new message.

        Raises:
            RuntimeError: If the memory is frozen.
        """
        if self._frozen:
            raise RuntimeError("Cannot add messages to a frozen ChatMemory.")
        self._messages.append(Message(role=role, content=content))
        if self.max_messages is not None:
            self._messages = self._messages[-self.max_messages :]
        logger.debug("ChatMemory: added %s message (%d total)", role, len(self._messages))

    def clear(self) -> None:
        """Remove all messages.

        Raises:
            RuntimeError: If the memory is frozen.
        """
        if self._frozen:
            raise RuntimeError("Cannot clear a frozen ChatMemory.")
        self._messages.clear()

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def messages(self) -> List[Message]:
        """Return a shallow copy of the current message list."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def to_dict(self) -> List[dict]:
        """Return messages as a list of ``{"role": ..., "content": ...}`` dicts."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> "ChatMemory":
        """Return a *frozen* copy of the current memory (pinned snapshot).

        The snapshot is independent of this instance; future mutations to
        this memory do not affect the snapshot.
        """
        snap = ChatMemory(frozen=True)
        snap._messages = list(self._messages)
        return snap
