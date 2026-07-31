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
        observation: str = "",
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
                "information_gains": 0,
                "score_gained": 0,
                "epochs": set(),
                "distinct_commands": set(),
                "no_progress_commands": set(),
                "information_signatures": set(),
                "last_progress_kind": "",
                "exhausted_in_prior_epoch": False,
            },
        )
        record["attempts"] += 1
        record["attempts_this_epoch"] += 1
        outcome = str(outcome_class or "").strip().lower()
        information_signature = self._information_signature(observation)
        new_information = bool(
            outcome == "info"
            and information_signature
            and information_signature not in record["information_signatures"]
        )
        if outcome == "state_change":
            record["state_changes"] += 1
            record["state_changes_this_epoch"] += 1
        if new_information:
            record["information_gains"] += 1
            record["information_signatures"].add(information_signature)
        if int(reward_change or 0) > 0:
            record["score_gained"] += int(reward_change)
        record["epochs"].add(int(epoch))
        record["distinct_commands"].add(command_key)
        progress_kind = ""
        if int(reward_change or 0) > 0 or outcome == "scored":
            progress_kind = "score"
        elif outcome == "state_change":
            progress_kind = "state_change"
        elif new_information:
            progress_kind = "new_information"
        if progress_kind:
            record["no_progress_commands"].clear()
            record["last_progress_kind"] = progress_kind
            record["exhausted_in_prior_epoch"] = False
        else:
            record["no_progress_commands"].add(command_key)
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
                    f"{label} [EXHAUSTED]: "
                    f"{len(record['no_progress_commands'])} distinct commands "
                    f"since the last useful result; {record['attempts']} attempts across "
                    f"{len(record['epochs'])} epoch(s). Closed for now; "
                    "re-engage only if a relevant item, explicit object change, "
                    "known reward, or stored goal provides new evidence."
                )
            elif int(record.get("attempts", 0)) >= 3:
                notes.append(
                    f"{label} [COVERED]: {record['attempts']} attempts, "
                    f"{record['state_changes']} lasting change(s), "
                    f"{record['information_gains']} new information result(s), "
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
        information_gains = int(record.get("information_gains", 0))
        score = int(record.get("score_gained", 0))
        no_progress_count = len(record.get("no_progress_commands", set()))
        # New information resets the no-progress streak when it occurs, but it
        # must not make the object permanently immune to later exhaustion.
        established_progress = bool(score > 0 or state_changes > 0)
        exhausted = (
            not established_progress
            and (
                no_progress_count >= config.OBJECT_EXHAUSTED_ATTEMPTS
                or (
                    record.get("exhausted_in_prior_epoch")
                    and no_progress_count >= 3
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
            "information_gains": information_gains,
            "score_gained": score,
            "distinct_no_progress_commands": no_progress_count,
            "last_progress_kind": record.get("last_progress_kind", ""),
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

    def tiers_for_objects(self, registry_room_id: str,
                          visible_objects: list) -> dict[str, dict]:
        output: dict[str, dict] = {}
        for value in visible_objects or []:
            label = str(
                value.get("name") if isinstance(value, dict) else value
            ).strip()
            key = self._noun_key(label)
            if key:
                output[key] = self.tier(registry_room_id, label)
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
            and len(record.get("no_progress_commands", set())) >= 5
        )

    @staticmethod
    def _information_signature(observation: str) -> str:
        normalized = normalize_text(observation)
        return normalized[:500]

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
            "no_progress_commands": sorted(
                record.get("no_progress_commands", set())
            ),
            "information_signatures": sorted(
                record.get("information_signatures", set())
            ),
        }
