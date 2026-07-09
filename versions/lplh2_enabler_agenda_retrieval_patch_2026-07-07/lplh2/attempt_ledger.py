"""Attempt ledger for LPLH2.

The ledger records factual command counts and outcomes by location. It does not
ban or judge commands; it makes repetition visible to the LLM.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from .command_keys import clean_text, normalize_command_key, normalize_location_key, normalize_text


class AttemptLedger:
    """Durable per-location command attempt counts."""

    def __init__(self):
        self._records: dict[str, dict[str, Any]] = {}
        self._room_visits: dict[str, dict[str, Any]] = {}

    def reset(self):
        self._records = {}
        self._room_visits = {}

    def record_room_visit(self, location: str, step: int, epoch: int = 1) -> dict[str, Any]:
        display = clean_text(location) or "unknown"
        key = normalize_location_key(display)
        if not key:
            key = "unknown"
        record = self._room_visits.get(key)
        if not record:
            record = {
                "location": display,
                "count": 0,
                "first_step": step,
                "last_step": step,
                "last_epoch": epoch,
            }
            self._room_visits[key] = record
        record["count"] = int(record.get("count", 0)) + 1
        record["last_step"] = step
        record["last_epoch"] = epoch
        if not record.get("location"):
            record["location"] = display
        return dict(record)

    def record_step(self, location: str, command: str, observation: str,
                    action_valid: bool | None, reward_change: int,
                    location_changed: bool, destination: str,
                    inventory_changed: bool, environment_changed: bool,
                    repetition_status: str, terminal_defeat: bool,
                    state_key: str,
                    step: int, epoch: int = 1) -> dict[str, Any]:
        loc_display = clean_text(location) or "unknown"
        cmd_display = clean_text(command)
        loc_key = normalize_location_key(loc_display) or "unknown"
        cmd_key = normalize_command_key(cmd_display)
        if not cmd_key:
            return {
                "status": "skipped_empty_command",
                "location": loc_display,
                "command": cmd_display,
            }

        key = f"{loc_key}|{cmd_key}"
        record = self._records.get(key)
        if not record:
            record = {
                "location": loc_display,
                "command": cmd_display,
                "command_key": cmd_key,
                "count": 0,
                "first_step": step,
                "last_step": step,
                "last_epoch": epoch,
                "outcomes": {},
                "destinations": {},
                "last_outcome": "",
                "last_observation": "",
                "distinct_outcomes": 0,
                "_outcome_hashes": [],
                "last_state_key": "",
            }
            self._records[key] = record

        outcome_class = self._outcome_class(
            action_valid=action_valid,
            reward_change=reward_change,
            location_changed=location_changed,
            inventory_changed=inventory_changed,
            environment_changed=environment_changed,
            repetition_status=repetition_status,
            terminal_defeat=terminal_defeat,
        )
        outcome_text = self._outcome_text(
            outcome_class=outcome_class,
            observation=observation,
            reward_change=reward_change,
            destination=destination,
        )

        record["command"] = cmd_display or record.get("command", "")
        record["count"] = int(record.get("count", 0)) + 1
        record["last_step"] = step
        record["last_epoch"] = epoch
        record["last_outcome"] = outcome_text
        record["last_observation"] = clean_text(observation)[:160]
        record["last_state_key"] = clean_text(state_key)
        outcomes = Counter(record.get("outcomes", {}))
        outcomes[outcome_class] += 1
        record["outcomes"] = dict(outcomes)
        if location_changed and destination:
            destinations = Counter(record.get("destinations", {}))
            destinations[clean_text(destination) or "unknown"] += 1
            record["destinations"] = dict(destinations)

        outcome_hashes = list(record.get("_outcome_hashes", []))
        digest = self._observation_digest(observation)
        if digest and digest not in outcome_hashes:
            outcome_hashes.append(digest)
            outcome_hashes = outcome_hashes[-6:]
        record["_outcome_hashes"] = outcome_hashes
        record["distinct_outcomes"] = len(outcome_hashes)

        return {
            "status": "recorded",
            "location": loc_display,
            "command": cmd_display,
            "command_key": cmd_key,
            "outcome_class": outcome_class,
            "count_after": record["count"],
            "distinct_outcomes": record["distinct_outcomes"],
            "room_visits_here": self.room_visit_count(loc_display),
            "record": self._public_record(record),
        }

    def count(self, location: str, command: str) -> int:
        record = self._records.get(
            f"{normalize_location_key(location) or 'unknown'}|{normalize_command_key(command)}"
        )
        return int(record.get("count", 0)) if record else 0

    def counts_for_location(self, location: str) -> dict[str, dict[str, Any]]:
        loc_key = normalize_location_key(location) or "unknown"
        output = {}
        prefix = f"{loc_key}|"
        for key, record in self._records.items():
            if key.startswith(prefix):
                output[record.get("command_key", key[len(prefix):])] = self._public_record(record)
        return output

    def room_visit_count(self, location: str) -> int:
        record = self._room_visits.get(normalize_location_key(location) or "unknown")
        return int(record.get("count", 0)) if record else 0

    def format_room_block(self, location: str, current_state_key: str = "",
                          max_items: int = 10) -> str:
        loc_display = clean_text(location) or "unknown"
        loc_key = normalize_location_key(loc_display) or "unknown"
        room = self._room_visits.get(loc_key, {})
        records = [
            record for key, record in self._records.items()
            if key.startswith(f"{loc_key}|")
        ]
        records.sort(key=lambda item: (-int(item.get("count", 0)), -int(item.get("last_step", 0))))

        lines = []
        visits = int(room.get("count", 0))
        if room:
            lines.append(
                f"{room.get('location', loc_display)}: visited {visits} time(s) "
                f"(first step {room.get('first_step')}, last step {room.get('last_step')})."
            )
        else:
            lines.append(f"{loc_display}: no prior visits recorded.")

        if not records:
            lines.append("Commands already executed from this room: none.")
            lines.append("Never tried from this room: any command not listed above.")
            return "\n".join(lines)

        lines.append("Commands already executed from this room:")
        current_state_key = clean_text(current_state_key)
        for record in records[:max_items]:
            count = int(record.get("count", 0))
            marker = " *" if current_state_key and record.get("last_state_key") == current_state_key else ""
            outcome_note = self._render_outcome_note(record)
            lines.append(
                f"- {record.get('command', '')} x{count} {outcome_note} "
                f"[last step {record.get('last_step')}]"
                f"{marker}"
            )
        if len(records) > max_items:
            lines.append(f"... +{len(records) - max_items} more command(s) tried here.")
        lines.append("(* = most recent attempt was in the current compact state.)")
        lines.append("Never tried from this room: any command not listed above.")
        return "\n".join(lines)

    def problem_attempts_for_location(self, location: str,
                                      max_items: int = 8) -> list[dict[str, Any]]:
        """Return compact invalid/unproductive/fatal attempts for prompt context.

        This complements failed-command memory: the ledger is factual attempt
        history, while failed-command memory stores a concise semantic reason.
        """
        loc_key = normalize_location_key(location) or "unknown"
        records = [
            record for key, record in self._records.items()
            if key.startswith(f"{loc_key}|")
        ]
        output: list[dict[str, Any]] = []
        for record in records:
            outcomes = record.get("outcomes", {}) or {}
            if not (
                int(outcomes.get("invalid", 0) or 0)
                or int(outcomes.get("unproductive", 0) or 0)
                or int(outcomes.get("fatal", 0) or 0)
            ):
                continue
            entry = {
                "command": record.get("command", ""),
                "count": int(record.get("count", 0) or 0),
                "outcome": clean_text(record.get("last_outcome", ""))[:160],
                "last_observation": clean_text(record.get("last_observation", ""))[:160],
                "last_step": int(record.get("last_step", 0) or 0),
            }
            output.append(entry)
        output.sort(key=lambda item: (-int(item.get("last_step", 0)), -int(item.get("count", 0))))
        return output[:max_items]

    def format_problem_attempts_for_prompt(self, location: str,
                                           max_items: int = 8) -> str:
        records = self.problem_attempts_for_location(location, max_items=max_items)
        if not records:
            return "[]"
        return json.dumps(records, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [self._public_record(record) for record in self._records.values()],
            "room_visits": {key: dict(value) for key, value in self._room_visits.items()},
        }

    def _outcome_class(self, action_valid: bool | None, reward_change: int,
                       location_changed: bool, inventory_changed: bool,
                       environment_changed: bool, repetition_status: str,
                       terminal_defeat: bool = False) -> str:
        if terminal_defeat:
            return "fatal"
        if action_valid is False:
            return "invalid"
        if reward_change:
            return "scored"
        if location_changed:
            return "moved"
        if inventory_changed or environment_changed:
            return "state_change"
        if repetition_status in {"stored_unproductive", "duplicate_unproductive", "stored_invalid", "duplicate_invalid"}:
            return "unproductive"
        return "info"

    def _outcome_text(self, outcome_class: str, observation: str,
                      reward_change: int, destination: str) -> str:
        if outcome_class == "fatal":
            return f"FATAL: {clean_text(observation)[:80]}"
        if outcome_class == "scored":
            sign = "+" if reward_change > 0 else ""
            return f"scored {sign}{reward_change}"
        if outcome_class == "moved":
            dest = clean_text(destination) or "another location"
            return f"moved to {dest}"
        if outcome_class == "invalid":
            return f"INVALID: {clean_text(observation)[:80]}"
        if outcome_class == "state_change":
            return f"changed state: {clean_text(observation)[:80]}"
        if outcome_class == "unproductive":
            return f"no new effect: {clean_text(observation)[:80]}"
        return f"gave info: {clean_text(observation)[:80]}"

    def _render_outcome_note(self, record: dict[str, Any]) -> str:
        distinct = int(record.get("distinct_outcomes", 0))
        if distinct > 1:
            return f"outcomes varied across attempts ({distinct} different results)"
        return f"{record.get('last_outcome', '')} - same result every time"

    def _observation_digest(self, observation: str) -> str:
        normalized = normalize_text(observation)[:120]
        if not normalized:
            return ""
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in record.items()
            if not key.startswith("_")
        }
