"""Session-scoped conversation history manager independent of pipeline engines."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:
    """Single user/assistant exchange."""

    user: str
    assistant: str


class ConversationManager:
    """Manage bounded multi-turn conversation history for local LLM prompts."""

    def __init__(
        self,
        system_prompt: str,
        *,
        max_turns: int = 10,
        max_chars: int = 12000,
    ) -> None:
        if max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns}")
        if max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._turns: list[ConversationTurn] = []

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt."""
        return self._system_prompt

    @property
    def turn_count(self) -> int:
        """Return number of stored user/assistant turn pairs."""
        return len(self._turns)

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        """Return an immutable snapshot of stored conversation turns."""
        return tuple(self._turns)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a completed turn and trim oldest turns to stay within bounds."""
        clean_user = user_text.strip()
        clean_assistant = assistant_text.strip()
        if not clean_user or not clean_assistant:
            return
        self._turns.append(ConversationTurn(user=clean_user, assistant=clean_assistant))
        self._trim()

    def _total_chars(self) -> int:
        return sum(len(t.user) + len(t.assistant) for t in self._turns)

    def _trim(self) -> None:
        while len(self._turns) > self._max_turns:
            self._turns.pop(0)
        while len(self._turns) > 1 and self._total_chars() > self._max_chars:
            self._turns.pop(0)

    def build_messages(
        self,
        current_user_text: str | None = None,
        *,
        extra_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages from system prompt, bounded history, and optional new input."""
        sys_content = self._system_prompt
        if extra_context and extra_context.strip():
            sys_content = f"{sys_content}\n\nContext:\n{extra_context.strip()}"

        messages: list[dict[str, str]] = [{"role": "system", "content": sys_content}]
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        if current_user_text and current_user_text.strip():
            messages.append({"role": "user", "content": current_user_text.strip()})
        return messages

    def clear(self) -> None:
        """Reset all conversation history for the current session."""
        self._turns.clear()

