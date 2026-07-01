"""Advisory situation plans for LPLH2.

The plan memory is intentionally small and non-controlling. It stores at most
one active plan proposed by the auxiliary gate, formats it for the action LLM,
and lets the agent clear it once the player actually tries the target action.
"""

from __future__ import annotations

import json
import re
from typing import Any


class ActivePlanMemory:
    """Stores one advisory plan tied to a previously stored situation."""

    def __init__(self):
        self._plan: dict[str, Any] | None = None

    def reset(self):
        self._plan = None

    def active_plan(self) -> dict[str, Any] | None:
        return dict(self._plan) if self._plan else None

    def has_active_plan(self) -> bool:
        return bool(self._plan)

    def set_plan(self, plan: dict[str, Any], step: int = 0,
                 source: str = "auxiliary_gate") -> dict:
        normalized = self._normalize_plan(plan)
        if not normalized.get("target_location"):
            return {
                "status": "skipped_missing_target",
                "active_plan": self.active_plan(),
                "proposed_plan": normalized,
            }
        related = normalized.get("related_situation") or {}
        if not related.get("situation"):
            return {
                "status": "skipped_missing_related_situation",
                "active_plan": self.active_plan(),
                "proposed_plan": normalized,
            }

        normalized["created_step"] = int(step or 0)
        normalized["source"] = source or "auxiliary_gate"
        normalized["status"] = "active"
        self._plan = normalized
        return {
            "status": "created",
            "active_plan": self.active_plan(),
            "proposed_plan": normalized,
        }

    def clear(self, reason: str = "", step: int = 0) -> dict:
        old = self.active_plan()
        self._plan = None
        return {
            "status": "cleared" if old else "no_active_plan",
            "reason": self._clean(reason),
            "step": int(step or 0),
            "cleared_plan": old,
            "active_plan": None,
        }

    def mark_preparation_done(self, command: str, step: int = 0) -> dict:
        """Remove an exact completed preparation command from the active plan."""
        before = self.active_plan()
        result = {
            "status": "no_active_plan",
            "completed_command": self._clean(command).lower(),
            "removed_preparation": "",
            "active_plan_before": before,
            "active_plan_after": before,
            "step": int(step or 0),
        }
        if not self._plan:
            return result

        normalized_command = self._clean(command).lower()
        preparations = list(self._plan.get("suggested_preparation", []) or [])
        remaining = [cmd for cmd in preparations if cmd != normalized_command]
        if len(remaining) == len(preparations):
            result["status"] = "not_a_preparation_command"
            return result

        self._plan["suggested_preparation"] = remaining
        result["status"] = "updated_preparation_completed"
        result["removed_preparation"] = normalized_command
        result["active_plan_after"] = self.active_plan()
        return result

    def format_for_prompt(self, navigation_hint: dict | None = None) -> str:
        if not self._plan:
            return "null"
        payload = {
            "plan": self.active_plan(),
            "navigation_hint": navigation_hint or {},
            "advisory_note": (
                "This plan is optional. Consider it when it fits the current "
                "observation. If pursuing it and a next_command hint exists, "
                "use that as the immediate navigation command."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def situation_key(self) -> str:
        if not self._plan:
            return ""
        return self._situation_key(self._plan.get("related_situation") or {})

    def matches_situation(self, situation: dict[str, Any]) -> bool:
        return bool(self._plan) and self.situation_key() == self._situation_key(situation)

    def target_commands(self) -> list[str]:
        """Compatibility accessor for older plan records."""
        if not self._plan:
            return []
        return list(self._plan.get("commands_to_try_at_target", []) or [])

    def _normalize_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict):
            plan = {}
        related_raw = plan.get("related_situation") or {}
        if not isinstance(related_raw, dict):
            related_raw = {}
        related = {
            "location": self._clean(related_raw.get("location")),
            "situation": self._clean(related_raw.get("situation")),
        }
        target = self._clean(plan.get("target_location")) or related["location"]
        return {
            "target_location": target,
            "related_situation": related,
            "reason": self._clean(plan.get("reason")),
            "suggested_preparation": self._clean_command_list(
                plan.get("suggested_preparation", [])
            ),
            "target_goal": self._clean(plan.get("target_goal")),
        }

    def _clean_command_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            raw = [value]
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        cleaned: list[str] = []
        for item in raw:
            text = self._clean(item).lower()
            text = re.sub(r"^[`\"']+|[`\"']+$", "", text).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned[:8]

    def _situation_key(self, situation: dict[str, Any]) -> str:
        location = self._normalize(situation.get("location", ""))
        text = self._normalize(situation.get("situation", ""))
        return f"{location}|{text}"

    def _normalize(self, text: str) -> str:
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _clean(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()
