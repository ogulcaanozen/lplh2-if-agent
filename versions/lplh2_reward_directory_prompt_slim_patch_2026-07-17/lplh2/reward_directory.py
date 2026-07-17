"""Persistent, observation-grounded directory of known score opportunities."""

from __future__ import annotations

import re
from typing import Any

from .command_keys import normalize_command_key, normalize_location_key


def compress_epoch_path(
    transitions: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Remove cycles from an observed path while preserving the surviving commands."""
    hops: list[tuple[str, str, str]] = []
    rooms: list[str] = []

    for source, command, destination in transitions or []:
        source = str(source or "").strip()
        command = str(command or "").strip()
        destination = str(destination or "").strip()
        if not source or not command or not destination or source == destination:
            continue

        if not rooms:
            rooms = [source]
        elif rooms[-1] != source:
            rooms = [source]
            hops = []

        destination_key = normalize_location_key(destination)
        existing_index = next(
            (
                index for index, room in enumerate(rooms)
                if normalize_location_key(room) == destination_key
            ),
            None,
        )
        if existing_index is not None:
            hops = hops[:existing_index]
            rooms = rooms[:existing_index + 1]
            continue

        hops.append((source, command, destination))
        rooms.append(destination)

    return hops


def render_route_hint(hops: list[tuple[str, str, str]], max_hops: int = 12) -> str:
    """Render compressed hops as concise route advice."""
    hops = list(hops or [])
    if not hops:
        return ""
    truncated = len(hops) > max_hops
    shown = hops[-max_hops:]
    parts = [f"{shown[0][0]}: {shown[0][1]}"]
    for source, command, destination in shown[1:]:
        parts.append(f"{source}: {command}")
    parts.append(shown[-1][2])
    route = " -> ".join(parts)
    return f"... {route}" if truncated else f"Start: {route}"


class RewardDirectory:
    """Cross-epoch directory keyed by the agent's grounded score event key."""

    def __init__(self):
        self._entries: dict[str, dict[str, Any]] = {}

    def add_or_update(self, entry_fields: dict[str, Any] | None = None, **kwargs) -> dict:
        fields = dict(entry_fields or {})
        fields.update(kwargs)
        event_key = str(fields.get("event_key") or "").strip()
        if not event_key:
            return {}

        existing = self._entries.get(event_key, {})
        setup_commands = self._clean_commands(fields.get("setup_commands", []))
        entry = {
            "event_key": event_key,
            "points": self._as_int(fields.get("points", existing.get("points", 0))),
            "location": str(
                fields.get("location") or existing.get("location") or "unknown"
            ).strip(),
            "scoring_command": str(
                fields.get("scoring_command")
                or existing.get("scoring_command")
                or ""
            ).strip(),
            "setup_commands": (
                setup_commands or list(existing.get("setup_commands", []))
                if "setup_commands" in fields
                else list(existing.get("setup_commands", []))
            ),
            "route_hint": str(existing.get("route_hint") or "").strip(),
            "first_seen_epoch": self._as_int(
                existing.get(
                    "first_seen_epoch",
                    fields.get("first_seen_epoch", 1),
                ),
                default=1,
            ),
        }
        self._entries[event_key] = entry
        if fields.get("route_hint"):
            self.mark_route(event_key, str(fields.get("route_hint")))
        return dict(self._entries[event_key])

    def mark_route(self, event_key: str, route_hint: str) -> bool:
        entry = self._entries.get(str(event_key or "").strip())
        route_hint = str(route_hint or "").strip()
        if not entry or not route_hint:
            return False
        current = str(entry.get("route_hint") or "").strip()
        if current and self._route_hop_count(route_hint) >= self._route_hop_count(current):
            return False
        entry["route_hint"] = route_hint
        return True

    def entries(self) -> list[dict]:
        return [
            dict(entry)
            for entry in sorted(
                self._entries.values(),
                key=lambda item: (
                    -self._as_int(item.get("points", 0)),
                    self._as_int(item.get("first_seen_epoch", 1), default=1),
                    str(item.get("event_key", "")),
                ),
            )
        ]

    def render(self, earned_keys: set | None = None) -> str:
        earned = {str(key) for key in (earned_keys or set())}
        entries = self.entries()
        if not entries:
            return "none known yet"

        unearned = [entry for entry in entries if entry["event_key"] not in earned]
        completed = [entry for entry in entries if entry["event_key"] in earned]
        lines = []
        for entry in unearned[:6]:
            setup = ", ".join(entry.get("setup_commands", [])) or "none"
            route = self._route_with_setup_cross_references(
                entry.get("route_hint", ""),
                owner_event_key=entry["event_key"],
                earned_keys=earned,
            ) or "not recorded yet"
            lines.append(
                f"[+{entry['points']}] NOT EARNED this epoch | "
                f"room: {entry['location']} | "
                f"scoring command: {entry['scoring_command']} | "
                f"setup in that room first: {setup} | route hint: {route}"
            )
        for entry in completed[:4]:
            lines.append(
                f"[+{entry['points']}] already earned this epoch | "
                f"room: {entry['location']} | do not repeat for points"
            )
        return "\n".join(lines) if lines else "none known yet"

    def reset_epoch_flags(self):
        """Earned flags live on the agent, so no directory state changes here."""

    def full_reset(self):
        self._entries = {}

    def __len__(self) -> int:
        return len(self._entries)

    def _route_with_setup_cross_references(
        self,
        route_hint: str,
        owner_event_key: str,
        earned_keys: set[str],
    ) -> str:
        output = str(route_hint or "").strip()
        if not output:
            return ""
        candidates = [
            entry for entry in self.entries()
            if entry.get("event_key") != owner_event_key
            and entry.get("event_key") not in earned_keys
            and entry.get("setup_commands")
        ]
        candidates.sort(
            key=lambda item: len(str(item.get("location", ""))),
            reverse=True,
        )
        for entry in candidates:
            location = str(entry.get("location") or "").strip()
            if not location:
                continue
            setup = ", ".join(entry.get("setup_commands", []))
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(location)}(?=\s*:|\s*(?:->|$))",
                flags=re.IGNORECASE,
            )
            output = pattern.sub(
                lambda match: f"{match.group(0)} (setup: {setup})",
                output,
            )
        return output

    @staticmethod
    def _clean_commands(commands: Any) -> list[str]:
        if not isinstance(commands, (list, tuple)):
            return []
        output = []
        seen = set()
        for command in commands:
            clean = re.sub(r"\s+", " ", str(command or "")).strip()
            key = normalize_command_key(clean)
            if clean and key and key not in seen:
                output.append(clean)
                seen.add(key)
        return output

    @staticmethod
    def _route_hop_count(route_hint: str) -> int:
        return max(1, str(route_hint or "").count("->"))

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default
