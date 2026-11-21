"""Credentials masked out of text before it is shown, logged or recorded: named credential fields, authorization
schemes, URL userinfo, JWTs and Dream Labs / MakerMods keys by their prefix."""

from __future__ import annotations

import re

_NAME = (r"[A-Za-z0-9_-]*?(?:api[_-]?key|api[_-]?token|access[_-]?token|refresh[_-]?token|auth[_-]?token"
         r"|id[_-]?token|token|secret|password|passwd|authorization|credentials?|cookie|signature)")
_NAME_RE = re.compile(r"\A" + _NAME + r"\Z", re.IGNORECASE)
_SCHEME = r"(?:Bearer|Basic|Digest|Token|ApiKey)"
_RULES = (
    # name=value, name: value, "name": "value" (the scheme of an authorization value is kept)
    (re.compile(r"(?<![A-Za-z0-9_-])(" + _NAME + r")([\"']?\s*[:=]\s*[\"']?)(" + _SCHEME + r"\s+)?"
                r"[^\s\"',;&)\]}>]+", re.IGNORECASE), lambda m: f"{m[1]}{m[2]}{m[3] or ''}***"),
    (re.compile(r"(?<![A-Za-z0-9_-])(Bearer|Basic|Digest|Token)\s+(?!\*\*\*)[^\s\"',;&)\]}>]+", re.IGNORECASE),
     lambda m: f"{m[1]} ***"),
    (re.compile(r"\b([a-z][a-z0-9+.-]*://)([^\s/?#@:\"']*)(:[^\s?#@\"']*)?@", re.IGNORECASE),
     lambda m: f"{m[1]}***@" if m[3] is None else f"{m[1]}{m[2]}:***@"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*"), lambda m: "eyJ***"),
    (re.compile(r"(dl_(?:live|test)_)[A-Za-z0-9_-]+", re.IGNORECASE), lambda m: f"{m[1]}***"),
    (re.compile(r"(?<![A-Za-z0-9])(mm_)[A-Za-z0-9]{16,}(?![A-Za-z0-9_-])"), lambda m: f"{m[1]}***"),
)


def mask_secrets(text) -> str:
    text = str(text)
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text


def redact_secrets(value):
    """A copy of a JSON value with credential fields' values replaced by ``***`` and every string masked."""
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, (list, tuple)):
        return [redact_secrets(v) for v in value]
    if isinstance(value, dict):
        return {k: "***" if isinstance(k, str) and _NAME_RE.match(k) and v is not None else redact_secrets(v)
                for k, v in value.items()}
    return value
