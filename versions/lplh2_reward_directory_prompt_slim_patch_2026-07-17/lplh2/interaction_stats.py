"""Persistent observational statistics for repeated object interaction."""

from __future__ import annotations

import re
from typing import Any

from .command_keys import normalize_command_key, normalize_text


class InteractionStats:
    """Track cross-epoch object outcomes without constraining action choice."""

    def __init__(self):
        self._records: dict[tuple[str, str], dict[str, Any]] = {}

    def record(
        self,
        registry_room_id: str,
        object_noun: str,
        command: str,
        outcome_class: str,
        reward_change: int,
        epoch: int,
    ) -> dict:
        room_id = str(registry_room_id or "").strip()
        noun = self._noun_key(object_noun)
        command_key = normalize_command_key(command)
        if not room_id or not noun or not command_key:
            return {}
        key = (room_id, noun)
        record = self._records.setdefault(
            key,
            {
                "registry_room_id": room_id,
                "object_noun": str(object_noun or "").strip().lower(),
                "attempts": 0,
                "state_changes": 0,
                "score_gained": 0,
                "epochs": set(),
                "distinct_commands": set(),
            },
        )
        record["attempts"] += 1
        if str(outcome_class or "") == "state_change":
            record["state_changes"] += 1
        if int(reward_change or 0) > 0:
            record["score_gained"] += int(reward_change)
        record["epochs"].add(int(epoch))
        record["distinct_commands"].add(command_key)
        return self._public_record(record)

    def notes_for_visible_objects(
        self,
        registry_room_id: str,
        visible_objects: list,
    ) -> str:
        room_id = str(registry_room_id or "").strip()
        if not room_id:
            return ""
        notes = []
        seen = set()
        for value in visible_objects or []:
            if isinstance(value, dict):
                label = str(value.get("name") or value.get("object") or "").strip()
            else:
                label = str(value or "").strip()
            noun = self._noun_key(label)
            if not noun or noun in seen:
                continue
            seen.add(noun)
            record = self._records.get((room_id, noun))
            if not self._is_futile(record):
                continue
            notes.append(
                f"{label}: {record['attempts']} commands tried across "
                f"{len(record['epochs'])} epoch(s), no score and no lasting change. "
                "Treat it as exhausted for now, re-engage only if you carry a new "
                "item or the observation shows it changed."
            )
        return "\n".join(notes)

    def records(self) -> list[dict]:
        return [self._public_record(record) for record in self._records.values()]

    def reset_epoch(self):
        """Statistics intentionally survive ordinary epoch resets."""

    def full_reset(self):
        self._records = {}

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _is_futile(record: dict | None) -> bool:
        return bool(
            record
            and int(record.get("attempts", 0)) >= 8
            and int(record.get("score_gained", 0)) == 0
            and int(record.get("state_changes", 0)) <= 1
        )

    @staticmethod
    def _noun_key(value: Any) -> str:
        normalized = normalize_text(value)
        normalized = re.sub(r"\b(?:the|a|an)\b", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            **record,
            "epochs": sorted(record.get("epochs", set())),
            "distinct_commands": sorted(record.get("distinct_commands", set())),
        }
