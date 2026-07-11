"""Situation memory for unresolved observations and preparation goals.

Observation situations remain epoch-local world state. Goal situations are
learned, non-local preparation requirements inferred from repeated deaths and
persist across epochs. Episodic memories stay place-bound; open goals are the
deliberate exception because their purpose is to change behaviour before the
agent reaches the hazardous room.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


class SituationMemory:
    """Store epoch-local observations and persistent preparation goals."""

    MAX_OPEN_GOALS = 5

    def __init__(self):
        self._situations: list[dict[str, str]] = []
        self._keys: set[str] = set()
        self._goal_situations: dict[str, dict[str, Any]] = {}
        self._next_goal_number = 1
        self.last_add_status = ""

    def reset(self, full: bool = False):
        """Reset situation state.

        Observation situations describe the current epoch's mutable world and
        are always cleared. Goal situations are learned knowledge keyed to
        stable room fingerprints, so they survive ordinary epoch resets. A
        full reset clears both stores and is used only when experiences are not
        being preserved.
        """
        self._situations = []
        self._keys = set()
        self.last_add_status = ""
        if full:
            self._goal_situations = {}
            self._next_goal_number = 1

    def active_situations(self) -> list[dict[str, str]]:
        output = [dict(item) for item in self._situations]
        for goal in self._goal_situations.values():
            if goal.get("declined") or goal.get("status") == "confirmed":
                continue
            output.append(self._render_goal(goal))
        return output

    def format_for_prompt(self) -> str:
        active = self.active_situations()
        if not active:
            return "[]"
        return json.dumps(active, ensure_ascii=False)

    def goal_situations(self) -> list[dict[str, Any]]:
        """Return complete goal records for logging and tests."""
        return [self._copy_goal(goal) for goal in self._goal_situations.values()]

    def open_goal_count(self) -> int:
        return sum(
            1 for goal in self._goal_situations.values()
            if not goal.get("declined") and goal.get("status") != "confirmed"
        )

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
        if self.find_goal_for_room(normalized["location"], "", open_only=True):
            self.last_add_status = "suppressed_by_goal"
            return False, normalized
        key = self._key(normalized)
        if not normalized["situation"] or key in self._keys:
            self.last_add_status = "duplicate"
            return False, normalized

        self._situations.append(normalized)
        self._keys.add(key)
        self.last_add_status = "stored"
        return True, normalized

    def add_goal_situation(
        self,
        hazard_location: str,
        hazard_fingerprint: str = "",
        fatal_action: str = "",
        gateway: Optional[dict[str, str]] = None,
        hazard_text: str = "",
        requires: Optional[list[str]] = None,
        advice: str = "",
        last_death_inventory: Optional[list[str]] = None,
        created_epoch: int = 1,
        deaths: int = 1,
    ) -> tuple[bool, Optional[dict[str, Any]], str]:
        """Create or merge a persistent goal for one hazardous room."""
        location = self._clean_field(hazard_location) or "unknown"
        fingerprint = self._clean_field(hazard_fingerprint)
        existing = self.find_goal_for_room(location, fingerprint, open_only=True)
        if existing:
            self._append_unique(existing["fatal_actions"], fatal_action)
            self._append_gateway(existing["gateways"], gateway)
            existing["deaths"] = max(
                int(existing.get("deaths", 0)),
                int(deaths or 0),
            )
            existing["last_death_inventory"] = self._clean_list(
                last_death_inventory
            )
            if hazard_text:
                existing["hazard_text"] = self._clean_field(hazard_text)
            if requires:
                existing["requires"] = self._merge_clean_lists(
                    existing.get("requires", []), requires
                )
            if advice:
                existing["advice"] = self._clean_field(advice)
            return False, self._copy_goal(existing), "merged"

        if self.open_goal_count() >= self.MAX_OPEN_GOALS:
            return False, None, "refused_cap"

        goal_id = f"goal_{self._next_goal_number:03d}"
        self._next_goal_number += 1
        goal = {
            "goal_id": goal_id,
            "hazard_location": location,
            "hazard_fingerprint": fingerprint,
            "fatal_actions": [],
            "gateways": [],
            "hazard_text": self._clean_field(hazard_text),
            "requires": self._clean_list(requires),
            "advice": self._clean_field(advice),
            "status": "untested",
            "deaths": max(1, int(deaths or 1)),
            "refutations": [],
            "declined": False,
            "last_death_inventory": self._clean_list(last_death_inventory),
            "confirmed_inventory": [],
            "created_epoch": int(created_epoch or 1),
        }
        self._append_unique(goal["fatal_actions"], fatal_action)
        self._append_gateway(goal["gateways"], gateway)
        self._goal_situations[goal_id] = goal
        return True, self._copy_goal(goal), "created"

    def find_goal_for_room(self, location: str, fingerprint: str = "",
                           open_only: bool = False) -> Optional[dict[str, Any]]:
        """Find a goal by stable hazard-room identity."""
        for goal in self._goal_situations.values():
            if open_only and (
                goal.get("declined") or goal.get("status") == "confirmed"
            ):
                continue
            if self._same_room(
                goal.get("hazard_location", ""),
                goal.get("hazard_fingerprint", ""),
                location,
                fingerprint,
            ):
                return goal
        return None

    def record_goal_death(self, goal_id: str, inventory: Optional[list[str]],
                          epoch: int = 0, step: int = 0) -> dict[str, Any]:
        """Record another death and report whether its inventory is new."""
        goal = self._goal_situations.get(goal_id)
        if not goal:
            return {"status": "not_found", "new_evidence": False}
        cleaned_inventory = self._clean_list(inventory)
        new_evidence = self._inventory_key(cleaned_inventory) != self._inventory_key(
            goal.get("last_death_inventory", [])
        )
        goal["deaths"] = int(goal.get("deaths", 0)) + 1
        if new_evidence:
            goal["last_death_inventory"] = cleaned_inventory
        return {
            "status": "recorded",
            "goal_id": goal_id,
            "new_evidence": new_evidence,
            "deaths": goal["deaths"],
            "epoch": int(epoch or 0),
            "step": int(step or 0),
        }

    def merge_goal_evidence(self, goal_id: str, fatal_action: str = "",
                            gateway: Optional[dict[str, str]] = None,
                            hazard_text: str = "") -> dict[str, Any]:
        """Merge a newly observed fatal action/gateway into an existing goal."""
        goal = self._goal_situations.get(goal_id)
        if not goal:
            return {"status": "not_found"}
        self._append_unique(goal["fatal_actions"], fatal_action)
        self._append_gateway(goal["gateways"], gateway)
        if hazard_text:
            goal["hazard_text"] = self._clean_field(hazard_text)
        return {
            "status": "merged",
            "goal_id": goal_id,
            "fatal_actions": list(goal["fatal_actions"]),
            "gateways": [dict(item) for item in goal["gateways"]],
        }

    def refute_goal(self, goal_id: str, inventory: Optional[list[str]],
                    epoch: int = 0, step: int = 0) -> dict[str, Any]:
        """Record observed failure despite carrying a hypothesized requirement."""
        goal = self._goal_situations.get(goal_id)
        if not goal:
            return {"status": "not_found"}
        evidence = {
            "inventory": self._clean_list(inventory),
            "epoch": int(epoch or 0),
            "step": int(step or 0),
        }
        goal["refutations"].append(evidence)
        if len(goal["refutations"]) >= 3:
            goal["status"] = "avoid"
        return {
            "status": "avoid" if goal["status"] == "avoid" else "refuted",
            "goal_id": goal_id,
            "refutations": len(goal["refutations"]),
        }

    def confirm_goal(self, goal_id: str, inventory: Optional[list[str]]) -> dict[str, Any]:
        goal = self._goal_situations.get(goal_id)
        if not goal:
            return {"status": "not_found"}
        goal["status"] = "confirmed"
        goal["confirmed_inventory"] = self._clean_list(inventory)
        return {"status": "confirmed", "goal_id": goal_id}

    def decline_goal(self, goal_id: str) -> dict[str, Any]:
        goal = self._goal_situations.get(goal_id)
        if not goal:
            return {"status": "not_found"}
        goal["declined"] = True
        return {"status": "declined", "goal_id": goal_id}

    def update_goal_hypothesis(self, goal_id: str,
                               requires: Optional[list[str]] = None,
                               advice: str = "", hazard_text: str = "") -> dict[str, Any]:
        """Apply a revised LLM hypothesis without incrementing death count."""
        goal = self._goal_situations.get(goal_id)
        if not goal:
            return {"status": "not_found"}
        if requires is not None:
            goal["requires"] = self._clean_list(requires)
        if advice:
            goal["advice"] = self._clean_field(advice)
        if hazard_text:
            goal["hazard_text"] = self._clean_field(hazard_text)
        return {"status": "updated", "goal_id": goal_id}

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

    def _render_goal(self, goal: dict[str, Any]) -> dict[str, str]:
        location = goal.get("hazard_location", "unknown")
        deaths = int(goal.get("deaths", 0))
        gateways = goal.get("gateways", [])
        gateway_text = ""
        if gateways:
            gateway = gateways[-1]
            gateway_text = (
                f" (entered via {gateway.get('command', 'unknown')} from "
                f"{gateway.get('room', 'unknown')})"
            )
        hazard = goal.get("hazard_text") or "entry has repeatedly been fatal"
        if goal.get("status") == "avoid":
            situation = (
                f"AVOID: entering this hazard has been fatal {deaths}x; "
                "no working preparation found"
            )
            solution = "Seek another route or new evidence before entering."
        else:
            situation = (
                f"PREPARATION REQUIRED (died {deaths}x): {hazard}{gateway_text}"
            )
            required = ", ".join(goal.get("requires", [])) or "a suitable counter"
            advice = goal.get("advice") or "Do not enter unprepared."
            solution = f"obtain first: {required}. {advice}"
        return {
            "location": location,
            "situation": situation,
            "possible_solution": solution,
        }

    def _same_room(self, left_location: str, left_fingerprint: str,
                   right_location: str, right_fingerprint: str) -> bool:
        left_fp = self._clean_field(left_fingerprint)
        right_fp = self._clean_field(right_fingerprint)
        if left_fp and right_fp:
            return left_fp == right_fp
        return self._base_location(left_location) == self._base_location(right_location)

    def _base_location(self, location: str) -> str:
        text = re.sub(r"\s+#\d+\s*$", "", self._clean_field(location))
        return self._normalize(text)

    def _clean_list(self, values: Optional[list[Any]]) -> list[str]:
        output: list[str] = []
        for value in values or []:
            cleaned = self._clean_field(value)
            if cleaned and self._normalize(cleaned) not in {
                self._normalize(item) for item in output
            }:
                output.append(cleaned)
        return output

    def _merge_clean_lists(self, left: list[Any], right: list[Any]) -> list[str]:
        return self._clean_list(list(left or []) + list(right or []))

    def _append_unique(self, values: list[str], value: Any):
        cleaned = self._clean_field(value)
        if cleaned and self._normalize(cleaned) not in {
            self._normalize(item) for item in values
        }:
            values.append(cleaned)

    def _append_gateway(self, gateways: list[dict[str, str]], gateway: Optional[dict]):
        if not isinstance(gateway, dict):
            return
        cleaned = {
            "room": self._clean_field(gateway.get("room")) or "unknown",
            "fingerprint": self._clean_field(gateway.get("fingerprint")),
            "command": self._clean_field(gateway.get("command")) or "unknown",
        }
        key = (
            self._base_location(cleaned["room"]),
            cleaned["fingerprint"],
            self._normalize(cleaned["command"]),
        )
        existing = {
            (
                self._base_location(item.get("room", "")),
                self._clean_field(item.get("fingerprint", "")),
                self._normalize(item.get("command", "")),
            )
            for item in gateways
        }
        if key not in existing:
            gateways.append(cleaned)

    def _inventory_key(self, inventory: list[str]) -> tuple[str, ...]:
        return tuple(sorted(self._normalize(item) for item in inventory if item))

    def _copy_goal(self, goal: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(goal, ensure_ascii=False))
