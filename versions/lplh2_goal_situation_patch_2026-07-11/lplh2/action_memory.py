"""Action failure memory for LPLH2.

This memory stores exact command failures at exact locations, with a compact
world-state snapshot. The visible records stay simple; deterministic keys are
kept only inside this class for deduplication and removal when a command later
works.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .command_keys import normalize_command_key, normalize_location_key, normalize_text


class FailedActionMemory:
    """Location-scoped memory of failed commands."""

    def __init__(self):
        self._records_by_location: dict[str, list[dict[str, Any]]] = {}
        self._index: dict[str, dict[str, Any]] = {}

    def reset(self):
        self._records_by_location = {}
        self._index = {}

    def record(self, location: str, command: str, observation: str,
               failure_reason: str, world_signature: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Store or refresh a location+command failure record."""
        record = {
            "location": self._clean(location) or "unknown",
            "command": self._clean(command),
            "observation": self._clean_observation(observation),
            "failure_reason": self._clean(failure_reason) or "The game rejected this command.",
            "world_signature": self._clean_signature(world_signature),
            "attempt_count": 1,
            "previous_failures": [],
        }
        if not record["command"]:
            return "duplicate", record

        key = self._key(record["location"], record["command"])
        if key in self._index:
            stored = self._index[key]
            stored["attempt_count"] = int(stored.get("attempt_count", 1) or 1) + 1
            if self._materially_different_failure(stored, record):
                previous = {
                    "observation": stored.get("observation", ""),
                    "failure_reason": stored.get("failure_reason", ""),
                }
                history = list(stored.get("previous_failures", []))
                if previous not in history:
                    history.append(previous)
                stored["previous_failures"] = history[-3:]
                stored["observation"] = record["observation"]
                stored["failure_reason"] = record["failure_reason"]
                stored["world_signature"] = record["world_signature"]
                return "updated_duplicate", dict(stored)
            return "duplicate", dict(stored)

        loc_key = self._location_key(record["location"])
        self._records_by_location.setdefault(loc_key, []).append(record)
        self._index[key] = record
        return "stored", dict(record)

    def remove(self, location: str, command: str) -> dict[str, Any] | None:
        """Remove a failure record when the same command later succeeds."""
        key = self._key(location, command)
        record = self._index.pop(key, None)
        if record is None:
            return None

        loc_key = self._location_key(record.get("location", location))
        self._records_by_location[loc_key] = [
            item for item in self._records_by_location.get(loc_key, [])
            if self._key(item.get("location", ""), item.get("command", "")) != key
        ]
        if not self._records_by_location.get(loc_key):
            self._records_by_location.pop(loc_key, None)
        return dict(record)

    def records_for_location(self, location: str) -> list[dict[str, Any]]:
        loc_key = self._location_key(location)
        return [dict(item) for item in self._records_by_location.get(loc_key, [])]

    def format_for_prompt(self, location: str, max_items: int = 12) -> str:
        records = self.records_for_location(location)[-max_items:]
        if not records:
            return "[]"
        return json.dumps(records, ensure_ascii=False)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            loc: [dict(item) for item in records]
            for loc, records in self._records_by_location.items()
        }

    def _key(self, location: str, command: str) -> str:
        return f"{self._location_key(location)}|{self._command_key(command)}"

    def _materially_different_failure(self, stored: dict[str, Any],
                                      new_record: dict[str, Any]) -> bool:
        old_reason = self._normalize(stored.get("failure_reason", ""))
        new_reason = self._normalize(new_record.get("failure_reason", ""))
        if new_reason and old_reason and new_reason != old_reason:
            return True
        old_obs = self._normalize(stored.get("observation", ""))
        new_obs = self._normalize(new_record.get("observation", ""))
        return bool(new_obs and old_obs and new_obs != old_obs)

    def _location_key(self, location: str) -> str:
        return normalize_location_key(location)

    def _command_key(self, command: str) -> str:
        return normalize_command_key(command)

    def _clean_signature(self, signature: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(signature, dict):
            return {}
        cleaned = {
            "location": self._clean(signature.get("location")) or "unknown",
            "inventory": self._clean_list(signature.get("inventory")),
            "visible_objects": self._clean_list(signature.get("visible_objects")),
        }
        if "score" in signature:
            cleaned["score"] = signature.get("score")
        return cleaned

    def _clean_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        output = []
        seen = set()
        for value in values:
            clean = self._clean(value)
            key = clean.lower()
            if clean and key not in seen:
                output.append(clean)
                seen.add(key)
        return output

    def _clean(self, value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _clean_command(self, value: Any) -> str:
        return self._clean(value).lower()

    def _clean_observation(self, value: Any) -> str:
        return self._clean(value)[:500]

    def _normalize(self, value: Any) -> str:
        return normalize_text(value)


class StateScopedActionMemory:
    """Exact-state memory of commands that did not help.

    This memory is deliberately advisory. It stores what happened in the
    state where the action was selected, then exposes matching records to the
    main LLM prompt when the agent reaches the same state snapshot again.
    """

    def __init__(self):
        self._records_by_state: dict[str, list[dict[str, Any]]] = {}
        self._index: dict[str, dict[str, Any]] = {}

    def reset(self):
        self._records_by_state = {}
        self._index = {}

    def make_state_snapshot(self, location: str, observation: str, inventory: list,
                            visible_objects: list, score: int) -> dict[str, Any]:
        return {
            "location": self._clean(location) or "unknown",
            "observation": self._clean_observation(observation),
            "inventory": self._clean_list(inventory),
            "visible_objects": self._clean_list(visible_objects),
            "score": score,
        }

    def record(self, state_snapshot: dict[str, Any], command: str,
               result_observation: str, reason: str, source: str) -> tuple[bool, dict[str, Any]]:
        snapshot = self._clean_snapshot(state_snapshot)
        record = {
            "location": snapshot.get("location") or "unknown",
            "command": self._clean(command),
            "state_snapshot": snapshot,
            "result_observation": self._clean_observation(result_observation),
            "reason": self._clean(reason) or "This command did not help in this exact state.",
            "source": self._clean(source) or "unknown",
        }
        if not record["command"]:
            return False, record

        state_key = self._state_key(snapshot)
        if not state_key:
            return False, record

        key = f"{state_key}|{self._command_key(record['command'])}"
        if key in self._index:
            return False, dict(self._index[key])

        self._records_by_state.setdefault(state_key, []).append(record)
        self._index[key] = record
        return True, dict(record)

    def remove_command_records(self, location: str, command: str) -> list[dict[str, Any]]:
        """Remove stale advisory records when the same command later helps."""
        loc_key = self._location_key(location)
        cmd_key = self._command_key(command)
        if not loc_key or not cmd_key:
            return []

        removed: list[dict[str, Any]] = []
        for key, record in list(self._index.items()):
            if (self._location_key(record.get("location", "")) == loc_key
                    and self._command_key(record.get("command", "")) == cmd_key):
                removed.append(dict(record))
                self._index.pop(key, None)

        if not removed:
            return []

        for state_key, records in list(self._records_by_state.items()):
            kept = [
                item for item in records
                if not (
                    self._location_key(item.get("location", "")) == loc_key
                    and self._command_key(item.get("command", "")) == cmd_key
                )
            ]
            if kept:
                self._records_by_state[state_key] = kept
            else:
                self._records_by_state.pop(state_key, None)
        return removed

    def records_for_state(self, state_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        state_key = self._state_key(self._clean_snapshot(state_snapshot))
        return [dict(item) for item in self._records_by_state.get(state_key, [])]

    def commands_for_state(self, state_snapshot: dict[str, Any]) -> list[str]:
        commands = []
        seen = set()
        for record in self.records_for_state(state_snapshot):
            command = self._clean(record.get("command"))
            key = self._command_key(command)
            if command and key not in seen:
                commands.append(command)
                seen.add(key)
        return commands

    def format_for_prompt(self, state_snapshot: dict[str, Any], max_items: int = 10) -> str:
        records = self.records_for_state(state_snapshot)[-max_items:]
        if not records:
            return "[]"

        visible_records = []
        for record in records:
            snapshot = record.get("state_snapshot", {})
            visible_records.append({
                "location": record.get("location", "unknown"),
                "command": record.get("command", ""),
                "result": record.get("result_observation", ""),
                "reason": record.get("reason", ""),
                "state_observation": snapshot.get("observation", ""),
                "inventory": snapshot.get("inventory", []),
                "visible_objects": snapshot.get("visible_objects", []),
                "score": snapshot.get("score"),
            })
        return json.dumps(visible_records, ensure_ascii=False)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            state: [dict(item) for item in records]
            for state, records in self._records_by_state.items()
        }

    def _clean_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(snapshot, dict):
            snapshot = {}
        cleaned = {
            "location": self._clean(snapshot.get("location")) or "unknown",
            "observation": self._clean_observation(snapshot.get("observation")),
            "inventory": self._clean_list(snapshot.get("inventory")),
            "visible_objects": self._clean_list(snapshot.get("visible_objects")),
        }
        if "score" in snapshot:
            cleaned["score"] = snapshot.get("score")
        return cleaned

    def _state_key(self, snapshot: dict[str, Any]) -> str:
        if not snapshot:
            return ""
        key_payload = {
            "location": self._location_key(snapshot.get("location")),
            "observation": self._normalize(snapshot.get("observation")),
            "inventory": sorted(self._normalize(x) for x in snapshot.get("inventory", []) if self._normalize(x)),
            "visible_objects": sorted(self._normalize(x) for x in snapshot.get("visible_objects", []) if self._normalize(x)),
            "score": snapshot.get("score"),
        }
        return json.dumps(key_payload, sort_keys=True, ensure_ascii=False)

    def _location_key(self, location: str) -> str:
        return normalize_location_key(location)

    def _command_key(self, command: str) -> str:
        return normalize_command_key(command)

    def _clean_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        output = []
        seen = set()
        for value in values:
            clean = self._clean(value)
            key = clean.lower()
            if clean and key not in seen:
                output.append(clean)
                seen.add(key)
        return output

    def _clean(self, value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _clean_observation(self, value: Any) -> str:
        return self._clean(value)[:700]

    def _normalize(self, value: Any) -> str:
        return normalize_text(value)
