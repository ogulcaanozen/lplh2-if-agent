"""Persistent, observation-grounded directory of known score opportunities."""

from __future__ import annotations

import re
from collections import deque
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
            "merged_event_keys": sorted({
                str(value)
                for value in existing.get("merged_event_keys", [])
                if str(value)
            }),
            "points": self._as_int(fields.get("points", existing.get("points", 0))),
            "location": str(
                fields.get("location") or existing.get("location") or "unknown"
            ).strip(),
            "location_registry_id": str(
                fields.get("location_registry_id")
                or existing.get("location_registry_id")
                or ""
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
            "route_hops": [
                list(hop) for hop in existing.get("route_hops", [])
            ],
            "hop_failures": dict(existing.get("hop_failures", {})),
            "first_seen_epoch": self._as_int(
                existing.get(
                    "first_seen_epoch",
                    fields.get("first_seen_epoch", 1),
                ),
                default=1,
            ),
        }
        self._entries[event_key] = entry
        if fields.get("route_hint") or fields.get("route_hops"):
            self.mark_route(
                event_key,
                str(fields.get("route_hint") or ""),
                fields.get("route_hops", []),
            )
        merged_key = self._merge_duplicates(event_key)
        return dict(self._entries.get(merged_key, self._entries.get(event_key, {})))

    def mark_route(
        self,
        event_key: str,
        route_hint: str,
        route_hops: list | None = None,
    ) -> bool:
        entry = self._entries.get(str(event_key or "").strip())
        route_hint = str(route_hint or "").strip()
        clean_hops = self._clean_hops(route_hops)
        if not entry or (not route_hint and not clean_hops):
            return False
        current = str(entry.get("route_hint") or "").strip()
        current_hops = self._clean_hops(entry.get("route_hops", []))
        new_count = len(clean_hops) if clean_hops else self._route_hop_count(route_hint)
        current_count = (
            len(current_hops) if current_hops else self._route_hop_count(current)
        )
        if current and new_count >= current_count:
            return False
        entry["route_hint"] = route_hint
        entry["route_hops"] = clean_hops
        entry["hop_failures"] = {}
        return True

    def next_hop_from(
        self,
        location: str,
        earned_keys: set | None = None,
        extra_edges: dict[str, dict[str, str]] | None = None,
    ) -> dict:
        for entry in self.ranked_unearned(
            location,
            earned_keys=earned_keys,
            extra_edges=extra_edges,
        ):
            path = entry.get("_path", [])
            if not path:
                continue
            source, command, destination = path[0]
            return {
                "event_key": entry["event_key"],
                "command": command,
                "points": self._as_int(entry.get("points", 0)),
                "entry_room": entry.get("location", "unknown"),
                "source": source,
                "destination": destination,
                "distance": int(entry.get("_distance", len(path))),
                "failed_once": self._hop_key(source, command)
                in entry.get("hop_failures", {}),
            }
        return {}

    def nearby_side_rewards(
        self,
        location: str,
        earned_keys: set | None = None,
        max_hops: int = 2,
        extra_edges: dict[str, dict[str, str]] | None = None,
    ) -> list[dict]:
        """Return nearby unearned rewards other than the primary routed target."""
        primary = self.next_hop_from(
            location,
            earned_keys=earned_keys,
            extra_edges=extra_edges,
        )
        primary_key = str(primary.get("event_key") or "")
        output = []
        for entry in self.ranked_unearned(
            location,
            earned_keys=earned_keys,
            extra_edges=extra_edges,
        ):
            distance = entry.get("_distance")
            if distance is None or int(distance) > max(0, int(max_hops)):
                continue
            if entry.get("event_key") == primary_key:
                continue
            path = entry.get("_path", [])
            first_command = (
                path[0][1] if path else entry.get("scoring_command", "")
            )
            output.append({
                "event_key": entry.get("event_key", ""),
                "points": self._as_int(entry.get("points", 0)),
                "room": entry.get("location", "unknown"),
                "scoring_command": entry.get("scoring_command", ""),
                "hop_count": int(distance),
                "first_command": first_command,
            })
        return output

    def matching_hops(
        self,
        location: str,
        command: str,
        earned_keys: set | None = None,
    ) -> list[dict]:
        location_key = normalize_location_key(location)
        command_key = normalize_command_key(command)
        earned = {str(value) for value in (earned_keys or set())}
        output = []
        for entry in self.entries():
            if self._entry_is_earned(entry, earned):
                continue
            for source, hop_command, destination in self._clean_hops(
                entry.get("route_hops", [])
            ):
                if (
                    normalize_location_key(source) == location_key
                    and normalize_command_key(hop_command) == command_key
                ):
                    output.append({
                        "event_key": entry["event_key"],
                        "source": source,
                        "command": hop_command,
                        "destination": destination,
                    })
        return output

    def flag_hop_failure(
        self,
        event_key: str,
        source: str,
        command: str,
    ) -> bool:
        entry = self._entries.get(str(event_key or "").strip())
        if not entry:
            return False
        key = self._hop_key(source, command)
        failures = entry.setdefault("hop_failures", {})
        failures[key] = int(failures.get(key, 0)) + 1
        return True

    def clear_hop_failure(
        self,
        event_key: str,
        source: str,
        command: str,
    ) -> bool:
        entry = self._entries.get(str(event_key or "").strip())
        if not entry:
            return False
        return entry.setdefault("hop_failures", {}).pop(
            self._hop_key(source, command),
            None,
        ) is not None

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

    def unearned_entries(self, earned_keys: set | None = None) -> list[dict]:
        """Return entries whose primary key and merged aliases are all unearned."""
        earned = {str(value) for value in (earned_keys or set())}
        return [
            entry for entry in self.entries()
            if not self._entry_is_earned(entry, earned)
        ]

    def render(
        self,
        earned_keys: set | None = None,
        current_location: str = "",
        extra_edges: dict[str, dict[str, str]] | None = None,
    ) -> str:
        earned = {str(key) for key in (earned_keys or set())}
        entries = self.entries()
        if not entries:
            return "none known yet"

        ranked = self.ranked_unearned(
            current_location,
            earned_keys=earned,
            extra_edges=extra_edges,
        )
        nearby = [
            entry for entry in ranked
            if entry.get("_distance") is not None
            and int(entry["_distance"]) <= 3
        ]
        nearby_keys = {entry["event_key"] for entry in nearby}
        farther = [
            entry for entry in ranked
            if entry["event_key"] not in nearby_keys
        ][:3]
        unearned = (nearby + farther)[:10]
        completed = [
            entry for entry in entries if self._entry_is_earned(entry, earned)
        ]
        lines = []
        for entry in unearned:
            setup = ", ".join(entry.get("setup_commands", [])) or "none"
            route = self._route_with_setup_cross_references(
                self._render_route(entry),
                owner_event_key=entry["event_key"],
                earned_keys=earned,
            ) or "not recorded yet"
            next_hop = ""
            path = entry.get("_path", [])
            if path:
                next_hop = (
                    f" | distance from here: {len(path)} hop(s)"
                    f" | next hop from here: {path[0][1]}"
                )
            elif entry.get("_distance") == 0:
                next_hop = " | in this room"
            lines.append(
                f"[+{entry['points']}] NOT EARNED this epoch | "
                f"room: {entry['location']} | "
                f"scoring command: {entry['scoring_command']} | "
                f"setup in that room first: {setup} | route hint: {route}"
                f"{next_hop}"
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

    def ranked_unearned(
        self,
        location: str,
        earned_keys: set | None = None,
        extra_edges: dict[str, dict[str, str]] | None = None,
    ) -> list[dict]:
        """Rank unearned opportunities by known graph distance, then value."""
        earned = {str(value) for value in (earned_keys or set())}
        output = []
        for entry in self.entries():
            if self._entry_is_earned(entry, earned):
                continue
            path = self._shortest_path(
                location,
                entry.get("location", ""),
                extra_edges=extra_edges,
            )
            item = dict(entry)
            item["_path"] = path or []
            item["_distance"] = (
                0 if normalize_location_key(location)
                == normalize_location_key(entry.get("location", ""))
                else (len(path) if path else None)
            )
            output.append(item)
        output.sort(key=lambda item: (
            item.get("_distance") is None,
            int(item.get("_distance") or 0)
            if item.get("_distance") is not None else 10**6,
            -self._as_int(item.get("points", 0)),
            self._as_int(item.get("first_seen_epoch", 1), default=1),
            str(item.get("event_key", "")),
        ))
        return output

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
            and not self._entry_is_earned(entry, earned_keys)
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

    def _render_route(self, entry: dict) -> str:
        route = str(entry.get("route_hint") or "").strip()
        failures = entry.get("hop_failures", {}) or {}
        if not failures:
            return route
        annotations = []
        for source, command, _ in self._clean_hops(entry.get("route_hops", [])):
            if self._hop_key(source, command) in failures:
                annotations.append(
                    f"{source}: {command} (this hop failed once from a room "
                    "with this name; if this name appears in more than one "
                    "place, the hop may work from the other one, or from a "
                    "neighboring approach)"
                )
        if annotations:
            route = f"{route} | " if route else ""
            route += " | ".join(annotations)
        return route

    def _merge_duplicates(self, added_event_key: str = "") -> str:
        """Merge only registry-grounded copies of the same scoring event."""
        groups: dict[tuple[int, str, str], list[dict]] = {}
        for entry in self._entries.values():
            registry_id = str(entry.get("location_registry_id") or "").strip()
            command_key = normalize_command_key(entry.get("scoring_command", ""))
            if not registry_id or not command_key:
                continue
            key = (
                self._as_int(entry.get("points", 0)),
                command_key,
                registry_id,
            )
            groups.setdefault(key, []).append(entry)

        retained_for_added = str(added_event_key or "")
        for duplicates in groups.values():
            if len(duplicates) < 2:
                continue

            def route_rank(entry: dict) -> tuple:
                hops = self._clean_hops(entry.get("route_hops", []))
                return (
                    0 if hops else 1,
                    len(hops) if hops else 10**6,
                    self._as_int(entry.get("first_seen_epoch", 1), default=1),
                    str(entry.get("event_key", "")),
                )

            keeper = min(duplicates, key=route_rank)
            setup_commands = []
            aliases = set(keeper.get("merged_event_keys", []))
            first_seen = self._as_int(
                keeper.get("first_seen_epoch", 1), default=1
            )
            for entry in duplicates:
                setup_commands.extend(entry.get("setup_commands", []))
                aliases.add(str(entry.get("event_key") or ""))
                aliases.update(
                    str(value)
                    for value in entry.get("merged_event_keys", [])
                    if str(value)
                )
                first_seen = min(
                    first_seen,
                    self._as_int(entry.get("first_seen_epoch", 1), default=1),
                )
            aliases.discard(str(keeper.get("event_key") or ""))
            aliases.discard("")
            keeper["setup_commands"] = self._clean_commands(setup_commands)
            keeper["merged_event_keys"] = sorted(aliases)
            keeper["first_seen_epoch"] = first_seen

            for entry in duplicates:
                event_key = str(entry.get("event_key") or "")
                if entry is keeper:
                    continue
                self._entries.pop(event_key, None)
                if event_key == retained_for_added:
                    retained_for_added = str(keeper.get("event_key") or "")
        return retained_for_added

    def _shortest_path(
        self,
        source: str,
        target: str,
        extra_edges: dict[str, dict[str, str]] | None = None,
    ) -> list[list[str]] | None:
        source_key = normalize_location_key(source)
        target_key = normalize_location_key(target)
        if not source_key or not target_key:
            return None
        if source_key == target_key:
            return []

        graph: dict[str, list[tuple[str, str, str]]] = {}

        def add_edge(edge_source: str, command: str, destination: str):
            src_key = normalize_location_key(edge_source)
            dest_key = normalize_location_key(destination)
            if not src_key or not dest_key or not command:
                return
            edge = (str(edge_source), str(command), str(destination))
            if edge not in graph.setdefault(src_key, []):
                graph[src_key].append(edge)

        for entry in self._entries.values():
            for edge_source, command, destination in self._clean_hops(
                entry.get("route_hops", [])
            ):
                add_edge(edge_source, command, destination)
        for edge_source, directions in (extra_edges or {}).items():
            if not isinstance(directions, dict):
                continue
            for command, destination in directions.items():
                add_edge(edge_source, command, destination)

        queue = deque([(source_key, [])])
        visited = {source_key}
        while queue:
            room_key, path = queue.popleft()
            for edge_source, command, destination in graph.get(room_key, []):
                dest_key = normalize_location_key(destination)
                next_path = path + [[edge_source, command, destination]]
                if dest_key == target_key:
                    return next_path
                if dest_key not in visited:
                    visited.add(dest_key)
                    queue.append((dest_key, next_path))
        return None

    @staticmethod
    def _entry_is_earned(entry: dict, earned_keys: set[str]) -> bool:
        keys = {str(entry.get("event_key") or "")}
        keys.update(
            str(value)
            for value in entry.get("merged_event_keys", [])
            if str(value)
        )
        return bool(keys.intersection(earned_keys))

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
    def _clean_hops(hops: Any) -> list[list[str]]:
        output = []
        for hop in hops or []:
            if not isinstance(hop, (list, tuple)) or len(hop) != 3:
                continue
            source, command, destination = (
                str(value or "").strip() for value in hop
            )
            if source and command and destination:
                output.append([source, command, destination])
        return output

    @staticmethod
    def _hop_key(source: str, command: str) -> str:
        return (
            f"{normalize_location_key(source)}|"
            f"{normalize_command_key(command)}"
        )

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default
