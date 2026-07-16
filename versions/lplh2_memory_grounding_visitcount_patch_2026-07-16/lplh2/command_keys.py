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


def commands_equivalent(a: Any, b: Any) -> bool:
    """Return whether two parser commands express the same local attempt."""
    left = normalize_command_key(a)
    right = normalize_command_key(b)
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return False
    left_verb, right_verb = left_tokens[0], right_tokens[0]
    fillers = {"up", "down", "through", "into", "in", "to", "at", "the", "a", "an"}
    left_object = [token for token in left_tokens[1:] if token not in fillers]
    right_object = [token for token in right_tokens[1:] if token not in fillers]
    if not left_object or left_object != right_object:
        return False
    movement = {"go", "climb", "enter", "walk", "crawl", "step", "get"}
    return left_verb == right_verb or (
        left_verb in movement and right_verb in movement
    )
