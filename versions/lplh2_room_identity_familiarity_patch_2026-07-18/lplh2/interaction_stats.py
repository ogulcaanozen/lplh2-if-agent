"""Persistent observational statistics for repeated object interaction."""

from __future__ import annotations

import re
from typing import Any

from .command_keys import normalize_command_key, normalize_text
from . import config


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
                "attempts_this_epoch": 0,
                "state_changes": 0,
                "state_changes_this_epoch": 0,
                "score_gained": 0,
                "epochs": set(),
                "distinct_commands": set(),
                "exhausted_in_prior_epoch": False,
            },
        )
        record["attempts"] += 1
        record["attempts_this_epoch"] += 1
        if str(outcome_class or "") == "state_change":
            record["state_changes"] += 1
            record["state_changes_this_epoch"] += 1
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
            tier = self.tier(room_id, label)
            if tier["tier"] == "FRESH":
                continue
            if tier["tier"] == "EXHAUSTED":
                notes.append(
                    f"{label} [EXHAUSTED]: {record['attempts']} attempts across "
                    f"{len(record['epochs'])} epoch(s), no score and no lasting "
                    "change. Closed for now; re-engage only if you carry a new "
                    "item, the observation shows it changed, or a known reward "
                    "names it."
                )
            elif int(record.get("attempts", 0)) >= 3:
                notes.append(
                    f"{label} [COVERED]: {record['attempts']} attempts, "
                    f"{record['state_changes']} lasting change(s), "
                    f"{record['score_gained']} score gained."
                )
        return "\n".join(notes)

    def tier(self, registry_room_id: str, object_noun: str) -> dict:
        room_id = str(registry_room_id or "").strip()
        noun = self._noun_key(object_noun)
        record = self._records.get((room_id, noun))
        if not record:
            return {"tier": "FRESH", "attempts": 0}
        attempts = int(record.get("attempts", 0))
        attempts_epoch = int(record.get("attempts_this_epoch", 0))
        state_changes = int(record.get("state_changes", 0))
        changes_epoch = int(record.get("state_changes_this_epoch", 0))
        score = int(record.get("score_gained", 0))
        productive = score > 0 or state_changes > 1
        exhausted = (
            not productive
            and (
                attempts >= config.OBJECT_EXHAUSTED_ATTEMPTS
                or (
                    record.get("exhausted_in_prior_epoch")
                    and attempts_epoch >= 3
                    and changes_epoch == 0
                )
            )
        )
        return {
            "tier": "EXHAUSTED" if exhausted else "COVERED",
            "attempts": attempts,
            "attempts_this_epoch": attempts_epoch,
            "state_changes": state_changes,
            "score_gained": score,
            "exhausted_in_prior_epoch": bool(
                record.get("exhausted_in_prior_epoch")
            ),
        }

    def untouched_objects(
        self,
        registry_room_id: str,
        visible_objects: list,
    ) -> list[str]:
        output = []
        for value in visible_objects or []:
            label = str(
                value.get("name") if isinstance(value, dict) else value
            ).strip()
            if label and self.tier(registry_room_id, label)["tier"] == "FRESH":
                output.append(label)
        return output

    def records(self) -> list[dict]:
        return [self._public_record(record) for record in self._records.values()]

    def reset_epoch(self):
        """Keep totals while rolling per-epoch exhaustion forward."""
        for record in self._records.values():
            record["exhausted_in_prior_epoch"] = (
                self.tier(
                    record.get("registry_room_id", ""),
                    record.get("object_noun", ""),
                )["tier"] == "EXHAUSTED"
            )
            record["attempts_this_epoch"] = 0
            record["state_changes_this_epoch"] = 0

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
