"""Situation memory for LPLH2.

This module stores active unresolved situations outside the experience
retrieval backend. It is intentionally small: the LLM decides whether a
step contains a new future-return situation, and this class parses,
normalizes, and deduplicates the resulting records.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


class SituationMemory:
    """Stores unresolved return-later situations."""

    def __init__(self):
        self._situations: list[dict[str, str]] = []
        self._keys: set[str] = set()

    def reset(self):
        self._situations = []
        self._keys = set()

    def active_situations(self) -> list[dict[str, str]]:
        return [dict(item) for item in self._situations]

    def format_for_prompt(self) -> str:
        if not self._situations:
            return "[]"
        return json.dumps(self._situations, ensure_ascii=False)

    def parse_response(self, text: str) -> tuple[Optional[dict[str, str]], Optional[str]]:
        """Parse the LLM detector response.

        Returns:
            (situation, error). If the response means "none", situation and
            error are both None. If parsing fails, situation is None and error
            is a short reason.
        """
        body = self._extract_body(text)
        if not body:
            return None, "empty response"
        if self._is_none(body):
            return None, None

        parsed: Any
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = self._parse_loose_json(body)
            if parsed is None:
                return None, "response was not valid JSON"

        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None
        if not isinstance(parsed, dict):
            return None, "response did not contain an object"

        location = self._clean_field(parsed.get("location"))
        situation = self._clean_field(parsed.get("situation"))
        possible_solution = self._clean_field(parsed.get("possible_solution"))
        if not location or not situation:
            return None, "response missing location or situation"

        return {
            "location": location,
            "situation": situation,
            "possible_solution": possible_solution,
        }, None

    def parse_resolution_response(self, text: str) -> tuple[list[dict[str, str]], Optional[str]]:
        """Parse a list of active situations that the LLM says are now solved."""
        body = self._extract_body(text)
        if not body:
            return [], "empty response"
        if self._is_none(body):
            return [], None

        parsed: Any
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = self._parse_loose_resolution_json(body)
            if parsed is None:
                return [], "response was not valid JSON"

        if isinstance(parsed, dict) and isinstance(parsed.get("resolved_situations"), list):
            parsed = parsed["resolved_situations"]
        elif isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return [], "response did not contain a list"

        resolved: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            location = self._clean_field(item.get("location"))
            situation = self._clean_field(item.get("situation"))
            if location and situation:
                resolved.append({"location": location, "situation": situation})
        return resolved, None

    def add(self, situation: dict[str, str]) -> tuple[bool, dict[str, str]]:
        """Add a situation if not already present."""
        normalized = {
            "location": self._clean_field(situation.get("location")) or "unknown",
            "situation": self._clean_field(situation.get("situation")) or "",
            "possible_solution": self._clean_field(situation.get("possible_solution")) or "",
        }
        key = self._key(normalized)
        if not normalized["situation"] or key in self._keys:
            return False, normalized

        self._situations.append(normalized)
        self._keys.add(key)
        return True, normalized

    def remove(self, situation: dict[str, str]) -> bool:
        """Remove a stored situation. Reserved for the later solver step."""
        key = self._key(situation)
        if key not in self._keys:
            return False
        self._situations = [s for s in self._situations if self._key(s) != key]
        self._keys.remove(key)
        return True

    def key_for(self, situation: dict[str, str]) -> str:
        """Return the normalized internal key for equality checks."""
        return self._key(situation)

    def _extract_body(self, text: str) -> str:
        raw = str(text or "").strip()
        match = re.search(r"\|start\|\s*(.*?)\s*\|end\|", raw, re.DOTALL | re.IGNORECASE)
        return (match.group(1) if match else raw).strip()

    def _is_none(self, body: str) -> bool:
        normalized = re.sub(r"[\s.]+", "", body.strip().lower())
        return normalized in {"none", "null", "no", "nosituation", "nonefound"}

    def _parse_loose_json(self, body: str) -> Optional[dict[str, str]]:
        """Best-effort parser for minor formatting drift."""
        location_match = re.search(
            r'"?location"?\s*:\s*"([^"]+)"',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        situation_match = re.search(
            r'"?situation"?\s*:\s*"([^"]+)"',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        possible_solution_match = re.search(
            r'"?possible_solution"?\s*:\s*"([^"]*)"',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not location_match or not situation_match:
            return None
        return {
            "location": location_match.group(1),
            "situation": situation_match.group(1),
            "possible_solution": (
                possible_solution_match.group(1)
                if possible_solution_match else ""
            ),
        }

    def _parse_loose_resolution_json(self, body: str) -> Optional[list[dict[str, str]]]:
        parsed = self._parse_loose_json(body)
        return [parsed] if parsed else None

    def _clean_field(self, value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _key(self, situation: dict[str, str]) -> str:
        location = self._normalize(situation.get("location", ""))
        text = self._normalize(situation.get("situation", ""))
        return f"{location}|{text}"

    def _normalize(self, text: str) -> str:
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
