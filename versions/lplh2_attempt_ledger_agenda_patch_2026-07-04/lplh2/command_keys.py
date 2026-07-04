"""Shared text normalization helpers for command/location keys."""

from __future__ import annotations

import re
from typing import Any


_DIRECTIONS = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "u": "up",
    "d": "down",
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "northeast": "northeast",
    "northwest": "northwest",
    "southeast": "southeast",
    "southwest": "southwest",
    "up": "up",
    "down": "down",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_location_key(location: Any) -> str:
    return normalize_text(location)


def normalize_command_key(command: Any) -> str:
    text = clean_text(command).lower()
    if text in _DIRECTIONS:
        return _DIRECTIONS[text]
    words = text.split()
    if len(words) >= 2 and words[0] in {"go", "walk", "head", "travel", "move"}:
        if words[1] in _DIRECTIONS:
            return _DIRECTIONS[words[1]]
    return normalize_text(text)

