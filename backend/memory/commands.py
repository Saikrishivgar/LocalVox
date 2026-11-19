"""Conversational memory inspection, explicit remember, update, forget, and delete commands."""

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from backend.config import AppConfig, get_config
from backend.memory.models import (
    MemoryCandidate,
    MemoryCategory,
    MemoryDatabaseError,
    MemoryPrivacyError,
    MemoryRecord,
    MemorySource,
    normalize_subject,
)
from backend.memory.retriever import FtsMemoryRetriever, extract_query_tokens
from backend.memory.store import MemoryStore


@dataclass(frozen=True)
class MemoryCommandResult:
    """Structured result of handling a user memory command."""

    handled: bool
    action: str
    response_text: str
    affected_records: tuple[MemoryRecord, ...] = ()
    requires_confirmation: bool = False


_CLEAR_ALL_RE = re.compile(
    r"^(?:clear|delete|wipe|erase|forget)\s+(?:all|every)(?:\s+stored)?\s+memor(?:y|ies)$|"
    r"^forget\s+everything(?:\s+about\s+me)?$",
    flags=re.IGNORECASE,
)
_CONFIRM_CLEAR_RE = re.compile(
    r"^(?:yes\s+)?(?:confirm|yes,?\s*clear|yes,?\s*delete)\s+(?:all\s+memor(?:y|ies)|it|everything)$|"
    r"^(?:yes,?\s*)?confirm\s+clear\s+all\s+memories$",
    flags=re.IGNORECASE,
)
_CANCEL_CLEAR_RE = re.compile(
    r"^(?:no|cancel|never\s*mind|keep\s+(?:them|memories))$",
    flags=re.IGNORECASE,
)
_INSPECT_ALL_RE = re.compile(
    r"^what\s+do\s+you\s+(?:remember|know)\s+about\s+me$|"
    r"^(?:list|show)\s+(?:all\s+|my\s+)?memories$",
    flags=re.IGNORECASE,
)
_INSPECT_PROJECTS_RE = re.compile(
    r"^what\s+do\s+you\s+(?:remember|know)\s+about\s+my\s+projects?$|"
    r"^(?:list|show)\s+my\s+projects?$",
    flags=re.IGNORECASE,
)
_REMEMBER_THIS_RE = re.compile(
    r"^(?:please\s+)?remember\s+(?:this|that)(?:\s*[:,-]?\s*(.*))?$",
    flags=re.IGNORECASE,
)
_REMEMBER_DIRECT_RE = re.compile(
    r"^(?:please\s+)?remember\s+(.+)$",
    flags=re.IGNORECASE,
)
_UPDATE_MEM_RE = re.compile(
    r"^(?:please\s+)?update\s+(?:this\s+memory|that\s+memory|memory)\s*(?:about\s+([^:,-]+?))?\s*"
    r"(?:to|with|:|-)\s*(.+)$|^update\s+this\s+memory\s*[:,-]?\s*(.+)$",
    flags=re.IGNORECASE,
)
_DELETE_BY_ID_RE = re.compile(
    r"^(?:delete|remove)\s+(?:the\s+)?memory\s+(?:id\s+)?([0-9a-f]{12,64})$",
    flags=re.IGNORECASE,
)
_FORGET_RE = re.compile(
    r"^(?:please\s+)?(?:forget|delete|remove)\s+(?:this\s+information|that\s+information|"
    r"(?:the\s+)?memor(?:y|ies)\s+(?:about\s+)?|about\s+)?(.+)?$",
    flags=re.IGNORECASE,
)


def _normalize_cmd(text: str) -> str:
    cleaned = " ".join(text.strip().rstrip(".!?").split())
    low = cleaned.lower()
    for prefix in ("hey localvox,", "hey localvox", "localvox,", "localvox"):
        if low.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip(" ,:-")
            break
    return cleaned


_QUESTION_REMEMBER_PREFIXES = (
    "remember when ",
    "remember what ",
    "remember why ",
    "remember how ",
    "remember where ",
    "remember who ",
    "remember if ",
    "remember whether ",
)


def _infer_category_from_text(text: str) -> MemoryCategory:
    low = text.lower()
    if any(w in low for w in ("project", "building", "app", "repo", "yolo", "fastapi", "next")):
        return MemoryCategory.ONGOING_PROJECT
    if any(
        w in low for w in ("prefer", "favorite", "favourite", "like ", "dislike", "always explain")
    ):
        return MemoryCategory.PERSONAL_PREFERENCE
    if any(w in low for w in ("hobby", "hobbies", "enjoy", "chess", "music", "reading")):
        return MemoryCategory.INTEREST_HOBBY
    if any(w in low for w in ("decided", "decision", "chose", "choose")):
        return MemoryCategory.IMPORTANT_DECISION
    if any(w in low for w in ("learning", "studying", "exam", "course", "syllabus", "chapter")):
        return MemoryCategory.LEARNING_PROGRESS
    if any(w in low for w in ("todo", "task", "deadline", "finish", "submit", "tomorrow")):
        return MemoryCategory.PENDING_TASK
    return MemoryCategory.USER_FACT


class MemoryCommandHandler:
    """Handle explicit voice and CLI memory inspection, retention, update, and deletion."""

    def __init__(
        self,
        store: MemoryStore,
        retriever: FtsMemoryRetriever,
        config: AppConfig | None = None,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._config = config if config is not None else get_config()
        self._pending_clear_until: float | None = None
        self._last_referenced_memory_id: str | None = None

    @property
    def has_pending_clear_confirmation(self) -> bool:
        """Return True if a destructive clear-all confirmation window is currently active."""
        if self._pending_clear_until is None:
            return False
        if time.monotonic() > self._pending_clear_until:
            self._pending_clear_until = None
            return False
        return True

    def _format_memories_speech(self, records: Sequence[MemoryRecord], header: str) -> str:
        if not records:
            return header
        snippets = "; ".join(f"{r.content}" for r in records[:5])
        if len(records) > 5:
            return f"{header} Showing top 5 of {len(records)}: {snippets}."
        return f"{header} {snippets}"

    def handle(
        self,
        text: str,
        *,
        recent_turns: Sequence[tuple[str, str]] = (),
        session_id: str | None = None,
    ) -> MemoryCommandResult | None:
        """Interpret and execute memory control commands; returns None for normal dialogue."""
        cmd = _normalize_cmd(text)
        if not cmd:
            return None

        # 1. Handle active confirmation for destructive bulk deletion.
        if self.has_pending_clear_confirmation:
            if _CONFIRM_CLEAR_RE.match(cmd):
                self._pending_clear_until = None
                try:
                    deleted = self._store.clear_all(confirmed=True)
                except MemoryDatabaseError as exc:
                    return MemoryCommandResult(
                        handled=True,
                        action="clear_all_failed",
                        response_text=(
                            "I could not clear your stored memories because the "
                            f"database failed: {exc}"
                        ),
                    )
                self._last_referenced_memory_id = None
                if deleted == 0:
                    return MemoryCommandResult(
                        handled=True,
                        action="clear_all",
                        response_text="There were no stored memories to delete.",
                    )
                return MemoryCommandResult(
                    handled=True,
                    action="clear_all",
                    response_text=f"Confirmed. Permanently deleted all {deleted} stored memories.",
                )
            if _CANCEL_CLEAR_RE.match(cmd):
                self._pending_clear_until = None
                return MemoryCommandResult(
                    handled=True,
                    action="clear_cancelled",
                    response_text="Cancelled. All stored memories remain intact.",
                )
            # Clear stale pending confirmation when user moves on to another topic.
            self._pending_clear_until = None

        # 2. Request bulk clear -> require explicit confirmation first.
        if _CLEAR_ALL_RE.match(cmd):
            count = self._store.count()
            if count == 0:
                return MemoryCommandResult(
                    handled=True,
                    action="clear_all_empty",
                    response_text="You currently have no stored memories.",
                )
            self._pending_clear_until = (
                time.monotonic() + self._config.memory.pending_confirmation_timeout_s
            )
            return MemoryCommandResult(
                handled=True,
                action="confirm_clear_all",
                requires_confirmation=True,
                response_text=(
                    f"You have {count} stored memories. Are you sure you want to delete all "
                    "of them? Say confirm clear all memories to proceed, or cancel to keep them."
                ),
            )

        # 3. Inspect user's projects.
        if _INSPECT_PROJECTS_RE.match(cmd):
            projects = self._retriever.filter_by_category(MemoryCategory.ONGOING_PROJECT, limit=10)
            if not projects:
                return MemoryCommandResult(
                    handled=True,
                    action="inspect_projects",
                    response_text="I do not have any saved memories about your projects yet.",
                )
            self._last_referenced_memory_id = projects[0].id
            return MemoryCommandResult(
                handled=True,
                action="inspect_projects",
                affected_records=tuple(projects),
                response_text=self._format_memories_speech(
                    projects,
                    f"Here is what I remember about your projects ({len(projects)}):",
                ),
            )

        # 4. Inspect all memories about the user.
        if _INSPECT_ALL_RE.match(cmd):
            records = self._store.list_memories(limit=10)
            if not records:
                return MemoryCommandResult(
                    handled=True,
                    action="inspect_all",
                    response_text="I do not have any saved memories about you yet.",
                )
            self._last_referenced_memory_id = records[0].id
            return MemoryCommandResult(
                handled=True,
                action="inspect_all",
                affected_records=tuple(records),
                response_text=self._format_memories_speech(
                    records,
                    f"Here is what I remember about you ({len(records)}):",
                ),
            )

        # 5. Delete specific memory by ID.
        m_id = _DELETE_BY_ID_RE.match(cmd)
        if m_id:
            target_id = m_id.group(1)
            existing = self._store.get(target_id)
            if existing is None:
                return MemoryCommandResult(
                    handled=True,
                    action="delete_not_found",
                    response_text=(
                        f"No memory record with ID {target_id} was found, so nothing was deleted."
                    ),
                )
            try:
                removed = self._store.delete(target_id)
            except MemoryDatabaseError as exc:
                return MemoryCommandResult(
                    handled=True,
                    action="delete_failed",
                    response_text=f"Database deletion failed; memory was not deleted: {exc}",
                )
            if not removed:
                return MemoryCommandResult(
                    handled=True,
                    action="delete_failed",
                    response_text="The memory could not be deleted from the database.",
                )
            if self._last_referenced_memory_id == target_id:
                self._last_referenced_memory_id = None
            return MemoryCommandResult(
                handled=True,
                action="delete_by_id",
                affected_records=(existing,),
                response_text=f"Deleted memory: {existing.content}",
            )

        # 6. Explicit update memory command.
        m_upd = _UPDATE_MEM_RE.match(cmd)
        if m_upd:
            query_hint = (m_upd.group(1) or "").strip()
            new_text = (m_upd.group(2) or m_upd.group(3) or "").strip()
            if not new_text:
                return MemoryCommandResult(
                    handled=True,
                    action="update_invalid",
                    response_text="Please tell me the new information to update the memory with.",
                )
            target_rec: MemoryRecord | None = None
            if query_hint:
                hits = self._retriever.search(query_hint, limit=1, min_score=0.1)
                if hits:
                    target_rec = hits[0].record
            elif self._last_referenced_memory_id is not None:
                target_rec = self._store.get(self._last_referenced_memory_id)
            if target_rec is None:
                hits = self._retriever.search(new_text, limit=1, min_score=0.15)
                if hits:
                    target_rec = hits[0].record
            if target_rec is None:
                recent = self._store.list_memories(limit=1)
                target_rec = recent[0] if recent else None
            if target_rec is None:
                return MemoryCommandResult(
                    handled=True,
                    action="update_not_found",
                    response_text="I could not find an existing memory to update.",
                )
            try:
                updated = self._store.update(
                    target_rec.id,
                    content=new_text,
                    keywords=extract_query_tokens(new_text),
                    source=MemorySource.USER_EXPLICIT,
                    confirmed_by_user=True,
                )
            except MemoryPrivacyError as exc:
                return MemoryCommandResult(
                    handled=True,
                    action="privacy_blocked",
                    response_text=str(exc),
                )
            except MemoryDatabaseError as exc:
                return MemoryCommandResult(
                    handled=True,
                    action="update_failed",
                    response_text=(
                        f"I could not update that memory because the database failed: {exc}"
                    ),
                )
            self._last_referenced_memory_id = updated.id
            return MemoryCommandResult(
                handled=True,
                action="update_memory",
                affected_records=(updated,),
                response_text=f"Updated memory to: {updated.content}",
            )

        # 7. Explicit remember ("Remember this" / "Remember that ...").
        raw_stripped = text.strip()
        cmd_lower = cmd.lower()
        is_remember_question = raw_stripped.endswith("?") or cmd_lower.startswith(
            _QUESTION_REMEMBER_PREFIXES
        )
        m_rem = (
            None
            if is_remember_question
            else (_REMEMBER_THIS_RE.match(cmd) or _REMEMBER_DIRECT_RE.match(cmd))
        )
        if m_rem:
            body = (m_rem.group(1) or "").strip() if m_rem.lastindex else ""
            if not body:
                if recent_turns:
                    prev_user, _prev_assistant = recent_turns[-1]
                    body = prev_user.strip()
                if not body:
                    return MemoryCommandResult(
                        handled=True,
                        action="remember_empty",
                        response_text="Please tell me what you want me to remember.",
                    )
            try:
                category = _infer_category_from_text(body)
                tokens = extract_query_tokens(body)
                subject = normalize_subject(" ".join(tokens[:3])) if tokens else ""
                cand = MemoryCandidate(
                    category=category,
                    subject=subject,
                    content=body,
                    keywords=tokens[:10],
                    importance=0.85,
                    confidence=1.0,
                    source=MemorySource.USER_EXPLICIT,
                    confirmed_by_user=True,
                    session_id=session_id,
                )
                saved, action, _ = self._store.upsert(cand)
            except MemoryPrivacyError as exc:
                return MemoryCommandResult(
                    handled=True,
                    action="privacy_blocked",
                    response_text=str(exc),
                )
            except MemoryDatabaseError as exc:
                return MemoryCommandResult(
                    handled=True,
                    action="remember_failed",
                    response_text=f"I could not save that memory due to a database error: {exc}",
                )
            self._last_referenced_memory_id = saved.id
            verb = "Updated" if action == "updated" else "Saved"
            return MemoryCommandResult(
                handled=True,
                action=f"remember_{action}",
                affected_records=(saved,),
                response_text=f"{verb} to memory: {saved.content}",
            )

        # 8. Forget / delete matching memory.
        if not raw_stripped.endswith("?") and cmd_lower.startswith(
            ("forget ", "delete memory", "remove memory", "delete that information")
        ):
            m_fgt = _FORGET_RE.match(cmd)
            topic = (m_fgt.group(1) or "").strip() if m_fgt else ""
            targets: list[MemoryRecord] = []
            if not topic or topic.lower() in (
                "this information",
                "that information",
                "that",
                "this",
            ):
                if self._last_referenced_memory_id is not None:
                    rec = self._store.get(self._last_referenced_memory_id)
                    if rec is not None:
                        targets = [rec]
                elif recent_turns:
                    last_u = recent_turns[-1][0]
                    hits = self._retriever.search(last_u, limit=1, min_score=0.25)
                    targets = [h.record for h in hits]
            else:
                hits = self._retriever.search(topic, limit=3, min_score=0.25)
                targets = [h.record for h in hits]

            if not targets:
                return MemoryCommandResult(
                    handled=True,
                    action="forget_not_found",
                    response_text="No matching memory was found, so nothing was deleted.",
                )

            deleted_records: list[MemoryRecord] = []
            try:
                for rec in targets[:1]:
                    if self._store.delete(rec.id):
                        deleted_records.append(rec)
                        if self._last_referenced_memory_id == rec.id:
                            self._last_referenced_memory_id = None
            except MemoryDatabaseError as exc:
                return MemoryCommandResult(
                    handled=True,
                    action="forget_failed",
                    response_text=f"Failed to delete memory from the database: {exc}",
                )

            if not deleted_records:
                return MemoryCommandResult(
                    handled=True,
                    action="forget_not_deleted",
                    response_text="The database reported zero deleted records.",
                )
            return MemoryCommandResult(
                handled=True,
                action="forget_memory",
                affected_records=tuple(deleted_records),
                response_text=f"Forgot memory: {deleted_records[0].content}",
            )

        return None
