"""Privacy filters blocking storage of passwords, secrets, keys, and credentials."""

import re

from backend.memory.models import MemoryPrivacyError

_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", flags=re.IGNORECASE),
    ),
    (
        "openai_or_api_prefix",
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16})\b"),
    ),
    (
        "bearer_or_jwt",
        re.compile(r"\bBearer\s+[A-Za-z0-9._-]{12,}\b|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    (
        "connection_string_secret",
        re.compile(r"\b[a-z]+://[^:\s/]+:[^@\s/]+@[^\s/]+", flags=re.IGNORECASE),
    ),
    (
        "password_or_passphrase",
        re.compile(
            r"\b(?:my\s+|the\s+|wifi\s+|wi-fi\s+|account\s+|root\s+|sudo\s+|login\s+|db\s+)?"
            r"(?:password|passphrase|passcode|pin\s+code|atm\s+pin|secret\s+key|api\s*key|access\s+token|auth\s+token)"
            r"(?:\s+is|\s*[:=])\s*[^\s,;]+",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "generic_credential_kv",
        re.compile(
            r"\b(?:api_key|secret_key|client_secret|access_token|private_key)\s*[:=]\s*['\"]?[^\s'\";]+",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "payment_card_number",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    ),
    (
        "ssn",
        re.compile(
            r"\b(?:ssn|social\s+security(?:\s+number)?)\s*(?:is|[:=])?\s*\d{3}-?\d{2}-?\d{4}\b",
            flags=re.IGNORECASE,
        ),
    ),
)


def _passes_luhn(digits_only: str) -> bool:
    if not digits_only.isdigit() or not (13 <= len(digits_only) <= 19):
        return False
    total = 0
    reverse_digits = digits_only[::-1]
    for idx, ch in enumerate(reverse_digits):
        n = ord(ch) - 48
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def contains_sensitive_data(text: str) -> tuple[bool, str | None]:
    """Return (True, reason) if text appears to contain secrets, credentials, or card/SSN PII."""
    if not text:
        return False, None
    for reason, pattern in _SENSITIVE_PATTERNS:
        for match in pattern.finditer(text):
            if reason == "payment_card_number":
                digits = re.sub(r"[^0-9]", "", match.group(0))
                if not _passes_luhn(digits):
                    continue
            return True, reason
    return False, None


def validate_memory_privacy(text: str, *, enabled: bool = True) -> None:
    """Raise MemoryPrivacyError when privacy filtering blocks sensitive text."""
    if not enabled:
        return
    blocked, reason = contains_sensitive_data(text)
    if blocked:
        raise MemoryPrivacyError(
            f"Refused to store sensitive credential/secret data ({reason})."
        )
