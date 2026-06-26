"""Affordance brainstorming for LPLH2.

This module does not choose actions. It parses and formats suggestions from an
LLM pass that asks: given the current room, inventory, recent failures, and
stored situations, what concrete text-game commands are worth trying?
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


class AffordanceBrainstormer:
    """Parses local/inventory affordance suggestions from the LLM."""

    def __init__(self):
        self._cache_location_key = ""
        self._cache_state_key = ""
        self._cached_ideas: list[dict[str, Any]] = []
        self._unproductive_by_state: dict[str, set[str]] = {}

    def reset(self):
        """Clear short-lived same-location affordance carryover."""
        self._cache_location_key = ""
        self._cache_state_key = ""
        self._cached_ideas = []
        self._unproductive_by_state = {}

    def parse_response(self, text: str) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Parse the LLM brainstorm response.

        Returns:
            (ideas, error). Empty ideas with no error means the LLM intentionally
            found no useful command ideas.
        """
        body = self._extract_body(text)
        if not body:
            return [], "empty response"
        if self._is_none(body):
            return [], None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = self._parse_loose_json(body)
            if parsed is None:
                return [], "response was not valid JSON"

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return [], "response did not contain a list"

        ideas: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            location = self._clean_field(item.get("location"))
            situation = self._clean_field(item.get("situation"))
            reason = self._clean_field(item.get("reason"))
            commands = self._clean_commands(item.get("commands_to_try"))
            if commands:
                ideas.append({
                    "location": location or "current location",
                    "situation": situation or self._situation_from_commands(commands),
                    "reason": reason,
                    "commands_to_try": commands,
                })

        return ideas, None

    def failure_context(self, recent_failed_commands: list,
                        known_failed_commands_here: str) -> dict[str, list[str]]:
        """Extract exact failed commands and their verbs for prompt context."""
        commands: list[str] = []
        for command in recent_failed_commands or []:
            clean = self._clean_command(command)
            if clean:
                commands.append(clean)

        known = self._parse_known_failed_commands(known_failed_commands_here)
        commands.extend(known)
        commands = self._dedupe_commands(commands)

        verbs: list[str] = []
        seen_verbs: set[str] = set()
        for command in commands:
            verb = command.split(" ", 1)[0].strip().lower()
            if verb and verb not in seen_verbs:
                verbs.append(verb)
                seen_verbs.add(verb)
        return {
            "failed_commands": commands,
            "failed_verbs": verbs,
        }

    def record_attempt_result(self, location: str, state_signature: dict[str, Any],
                              command: str, useful: bool):
        """Remember valid-but-unproductive commands for the exact unchanged state."""
        clean = self._clean_command(command)
        if not clean or useful:
            return
        state_key = self._attempt_state_key(location, state_signature)
        if not state_key:
            return
        self._unproductive_by_state.setdefault(state_key, set()).add(
            self._command_key(clean)
        )

    def unproductive_commands(self, location: str,
                              state_signature: dict[str, Any] | None) -> list[str]:
        """Return commands already tried without progress in this exact state."""
        state_key = self._attempt_state_key(location, state_signature)
        if not state_key:
            return []
        return sorted(self._unproductive_by_state.get(state_key, set()))

    def cached_ideas_for_state(self, location: str,
                               state_signature: dict[str, Any] | None,
                               failed_commands: list[str] | None = None) -> list[dict[str, Any]]:
        """Return cached affordance ideas when the exact compact state still matches."""
        location_key = self._normalize(location)
        state_key = self._state_key(state_signature)
        same_context = (
            bool(location_key)
            and location_key == self._cache_location_key
            and state_key == self._cache_state_key
        )
        if not same_context:
            return []
        ideas = self._copy_ideas(self._cached_ideas)
        if failed_commands:
            ideas = self.filter_failed_commands(ideas, failed_commands)
        return ideas

    def state_signature(self, location: str, visible_objects: list,
                        inventory: list, score: int) -> dict[str, Any]:
        """Compact state used to decide whether same-location carryover is stale."""
        return {
            "location": self._normalize(location),
            "visible_objects": sorted(self._normalize(x) for x in (visible_objects or []) if self._normalize(x)),
            "inventory": sorted(self._normalize(x) for x in (inventory or []) if self._normalize(x)),
            "score": score,
        }

    def merge_with_carryover(self, location: str, fresh_ideas: list[dict[str, Any]],
                             failed_commands: list[str],
                             state_signature: dict[str, Any] | None = None,
                             reset_cache: bool = False) -> dict[str, Any]:
        """Merge fresh LLM ideas with untried ideas from the same location.

        The carryover is intentionally short-lived: it survives only while the
        agent remains in the same location and the compact visible state is
        unchanged. Returning to a location later asks the LLM to brainstorm again.
        """
        location_key = self._normalize(location)
        state_key = self._state_key(state_signature)
        same_context = (
            bool(location_key)
            and location_key == self._cache_location_key
            and state_key == self._cache_state_key
            and not reset_cache
        )
        carried_before = self._copy_ideas(self._cached_ideas) if same_context else []

        merged = self._merge_ideas(carried_before + self._copy_ideas(fresh_ideas))
        filtered = self.filter_failed_commands(merged, failed_commands)

        self._cache_location_key = location_key
        self._cache_state_key = state_key
        self._cached_ideas = self._copy_ideas(filtered)

        return {
            "carried_ideas_before": carried_before,
            "fresh_ideas": self._copy_ideas(fresh_ideas),
            "filtered_failed_commands": self._dedupe_commands(failed_commands or []),
            "merged_ideas": filtered,
            "carried_ideas_after": self._copy_ideas(self._cached_ideas),
            "cache_reset": not same_context,
        }

    def filter_failed_commands(self, ideas: list[dict[str, Any]],
                               failed_commands: list[str]) -> list[dict[str, Any]]:
        """Remove exact failed commands from brainstorm ideas."""
        failed_keys = {self._command_key(command) for command in (failed_commands or [])}
        output: list[dict[str, Any]] = []
        for idea in ideas or []:
            commands = [
                command for command in idea.get("commands_to_try", [])
                if self._command_key(command) not in failed_keys
            ]
            if not commands:
                continue
            cleaned = dict(idea)
            cleaned["commands_to_try"] = commands[:4]
            output.append(cleaned)
        return output[:5]

    def format_for_prompt(self, ideas: list[dict[str, Any]],
                          include_reason: bool = True) -> str:
        if not ideas:
            return "[]"
        if include_reason:
            return json.dumps(ideas, ensure_ascii=False)
        compact = []
        for idea in ideas:
            compact.append({
                "location": idea.get("location", "current location"),
                "situation": idea.get("situation", ""),
                "commands_to_try": idea.get("commands_to_try", []),
            })
        return json.dumps(compact, ensure_ascii=False)

    def _extract_body(self, text: str) -> str:
        raw = str(text or "").strip()
        match = re.search(r"\|start\|\s*(.*?)\s*\|end\|", raw, re.DOTALL | re.IGNORECASE)
        return (match.group(1) if match else raw).strip()

    def _is_none(self, body: str) -> bool:
        normalized = re.sub(r"[\s.]+", "", body.strip().lower())
        return normalized in {"none", "null", "no", "noideas", "noidea", "[]"}

    def _parse_loose_json(self, body: str) -> Optional[list[dict[str, Any]]]:
        """Best-effort parser for minor formatting drift."""
        complete_items = self._parse_complete_object_fragments(body)
        command_items = self._parse_command_array_fragments(body)
        items = self._merge_ideas(complete_items + command_items)
        return items or None

    def _parse_complete_object_fragments(self, body: str) -> list[dict[str, Any]]:
        """Recover every complete JSON object embedded in a truncated list."""
        fragments: list[dict[str, Any]] = []
        depth = 0
        start = None
        in_string = False
        escape = False
        for idx, char in enumerate(body):
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    fragment = body[start:idx + 1]
                    try:
                        parsed = json.loads(fragment)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        fragments.append(parsed)
                    start = None
        return fragments

    def _parse_command_array_fragments(self, body: str) -> list[dict[str, Any]]:
        """Recover complete commands_to_try arrays even when surrounding JSON is broken."""
        items: list[dict[str, Any]] = []
        for commands_match in re.finditer(
            r'"?commands_to_try"?\s*:\s*\[(.*?)\]',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            commands = re.findall(r'"([^"]+)"', commands_match.group(1))
            if not commands:
                continue

            segment_start = body.rfind("{", 0, commands_match.start())
            segment = body[segment_start:commands_match.end()] if segment_start >= 0 else body[:commands_match.end()]
            location_match = list(re.finditer(
                r'"?location"?\s*:\s*"([^"]+)"',
                segment,
                flags=re.IGNORECASE | re.DOTALL,
            ))
            situation_match = list(re.finditer(
                r'"?situation"?\s*:\s*"([^"]+)"',
                segment,
                flags=re.IGNORECASE | re.DOTALL,
            ))
            reason_match = list(re.finditer(
                r'"?reason"?\s*:\s*"([^"]+)"',
                segment,
                flags=re.IGNORECASE | re.DOTALL,
            ))
            items.append({
                "location": location_match[-1].group(1) if location_match else "current location",
                "situation": situation_match[-1].group(1) if situation_match else "",
                "reason": reason_match[-1].group(1) if reason_match else "",
                "commands_to_try": commands,
            })
        return items

    def _parse_known_failed_commands(self, text: str) -> list[str]:
        body = str(text or "").strip()
        if not body or self._is_none(body):
            return []
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        commands = []
        for item in parsed:
            if isinstance(item, dict):
                command = self._clean_command(item.get("command"))
                if command:
                    commands.append(command)
            elif isinstance(item, str):
                command = self._clean_command(item)
                if command:
                    commands.append(command)
        return commands

    def _clean_field(self, value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _clean_commands(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        commands: list[str] = []
        seen: set[str] = set()
        for command in value:
            clean = self._clean_command(command)
            key = clean.lower()
            if clean and key not in seen:
                commands.append(clean)
                seen.add(key)
        return commands[:4]

    def _clean_command(self, value: Any) -> str:
        text = self._clean_field(value)
        text = re.sub(r"^<com>\s*|\s*</com>$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:120]

    def _situation_from_commands(self, commands: list[str]) -> str:
        """Fallback label when the LLM gives useful commands but omits situation."""
        first = commands[0] if commands else "suggested command"
        return f"commands suggested around {first}"

    def _merge_ideas(self, ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        by_key: dict[str, dict[str, Any]] = {}
        for idea in ideas or []:
            situation = self._clean_field(idea.get("situation"))
            location = self._clean_field(idea.get("location")) or "current location"
            key = f"{self._normalize(location)}|{self._normalize(situation)}"
            if key not in by_key:
                clean = {
                    "location": location,
                    "situation": situation or self._situation_from_commands(idea.get("commands_to_try", [])),
                    "reason": self._clean_field(idea.get("reason")),
                    "commands_to_try": [],
                }
                by_key[key] = clean
                merged.append(clean)
            existing = by_key[key]
            for command in self._clean_commands(idea.get("commands_to_try")):
                if self._command_key(command) not in {
                    self._command_key(c) for c in existing["commands_to_try"]
                }:
                    existing["commands_to_try"].append(command)
            existing["commands_to_try"] = existing["commands_to_try"][:4]
            if not existing.get("reason") and idea.get("reason"):
                existing["reason"] = self._clean_field(idea.get("reason"))
        return [idea for idea in merged if idea.get("commands_to_try")][:5]

    def _dedupe_commands(self, commands: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for command in commands or []:
            clean = self._clean_command(command)
            key = self._command_key(clean)
            if clean and key not in seen:
                output.append(clean)
                seen.add(key)
        return output

    def _copy_ideas(self, ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return json.loads(json.dumps(ideas or [], ensure_ascii=False))

    def _state_key(self, signature: dict[str, Any] | None) -> str:
        if not isinstance(signature, dict):
            return ""
        return json.dumps(signature, sort_keys=True, ensure_ascii=False)

    def _attempt_state_key(self, location: str, signature: dict[str, Any] | None) -> str:
        state_key = self._state_key(signature)
        location_key = self._normalize(location)
        if not state_key and not location_key:
            return ""
        return f"{location_key}|{state_key}"

    def _command_key(self, command: Any) -> str:
        text = self._clean_command(command).lower()
        directions = {
            "n": "north", "s": "south", "e": "east", "w": "west",
            "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
            "u": "up", "d": "down",
            "north": "north", "south": "south", "east": "east", "west": "west",
            "northeast": "northeast", "northwest": "northwest",
            "southeast": "southeast", "southwest": "southwest",
            "up": "up", "down": "down",
        }
        if text in directions:
            return directions[text]
        words = text.split()
        if len(words) >= 2 and words[0] in {"go", "walk", "head", "travel", "move"}:
            if words[1] in directions:
                return directions[words[1]]
        return self._normalize(text)

    def _normalize(self, value: Any) -> str:
        text = self._clean_field(value).lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
