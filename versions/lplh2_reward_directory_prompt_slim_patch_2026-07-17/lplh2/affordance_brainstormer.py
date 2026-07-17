"""Affordance brainstorming for LPLH2.

This module does not choose actions. It parses and formats suggestions from an
LLM pass that asks: given the current room, inventory, recent failures, and
stored situations, what concrete text-game commands are worth trying?
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .command_keys import (
    commands_equivalent,
    normalize_command_key,
    normalize_location_key,
    normalize_text,
)


class AffordanceBrainstormer:
    """Parses local/inventory affordance suggestions from the LLM."""

    def __init__(self):
        self._cache_location_key = ""
        self._cache_state_key = ""
        self._cached_ideas: list[dict[str, Any]] = []
        self._ideas_by_location: dict[str, dict[str, Any]] = {}
        self._unproductive_by_state: dict[str, set[str]] = {}

    def reset(self):
        """Clear short-lived same-location affordance carryover."""
        self._cache_location_key = ""
        self._cache_state_key = ""
        self._cached_ideas = []
        self._ideas_by_location = {}
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
            kind = self._clean_kind(item.get("kind"))
            preparation_for = self._clean_field(item.get("preparation_for"))
            commands = self._clean_commands(item.get("commands_to_try"))
            if commands:
                idea = {
                    "location": location or "current location",
                    "situation": situation or self._situation_from_commands(commands),
                    "reason": reason,
                    "commands_to_try": commands,
                }
                if kind:
                    idea["kind"] = kind
                if preparation_for:
                    idea["preparation_for"] = preparation_for
                ideas.append(idea)

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
                               failed_commands: list[str] | None = None,
                               attempt_counts: dict[str, dict[str, Any]] | None = None,
                               active_situation_present: bool = False,
                               active_condition_present: bool = False) -> list[dict[str, Any]]:
        """Return cached affordance ideas that still apply at this location."""
        location_key = self._normalize(location)
        state_key = self._state_key(state_signature)
        cached = self._ideas_by_location.get(location_key, {})
        if not cached:
            return []
        ideas = self._copy_ideas(cached.get("ideas", []))
        if self._cache_location_key and self._cache_location_key != location_key:
            ideas = self._drop_generic_condition_carryover(
                ideas,
                allow_condition=active_condition_present,
            )
        if cached.get("state_key") != state_key:
            ideas = self._ideas_still_relevant(
                ideas,
                state_signature,
                preserve_preparation=active_situation_present,
            )
        if failed_commands:
            ideas = self.filter_failed_commands(ideas, failed_commands)
        if attempt_counts:
            ideas = self.consume_completed_commands(ideas, attempt_counts)
        return ideas

    def cache_status(self, location: str,
                     state_signature: dict[str, Any] | None) -> dict[str, Any]:
        """Describe whether a per-location affordance agenda exists."""
        location_key = self._normalize(location)
        state_key = self._state_key(state_signature)
        cached = self._ideas_by_location.get(location_key)
        if not cached:
            return {
                "status": "none",
                "cached_ideas": 0,
                "exact_state": False,
            }
        return {
            "status": "exact" if cached.get("state_key") == state_key else "stale",
            "cached_ideas": len(cached.get("ideas", [])),
            "exact_state": cached.get("state_key") == state_key,
            "cached_state_key": cached.get("state_key", ""),
            "current_state_key": state_key,
        }

    def state_signature(self, location: str, visible_objects: list,
                        inventory: list, score: int,
                        observation: str = "") -> dict[str, Any]:
        """Compact state used to decide whether same-location carryover is stale."""
        return {
            "location": self._normalize(location),
            "visible_objects": sorted(self._normalize(x) for x in (visible_objects or []) if self._normalize(x)),
            "inventory": sorted(self._normalize(x) for x in (inventory or []) if self._normalize(x)),
            "observation": self._normalize(observation)[:500],
            "score": score,
        }

    def merge_with_carryover(self, location: str, fresh_ideas: list[dict[str, Any]],
                             failed_commands: list[str],
                             state_signature: dict[str, Any] | None = None,
                             reset_cache: bool = False,
                             attempt_counts: dict[str, dict[str, Any]] | None = None,
                             active_situation_present: bool = False,
                             active_condition_present: bool = False) -> dict[str, Any]:
        """Merge fresh LLM ideas with still-relevant ideas from this location.

        Exact-state ideas carry over directly. If the compact state changed, keep
        only ideas still anchored to visible objects, inventory, or condition
        context. This preserves unfinished local business without creating a
        separate planning module.
        """
        location_key = self._normalize(location)
        state_key = self._state_key(state_signature)
        cached = self._ideas_by_location.get(location_key, {})
        same_context = (
            bool(location_key)
            and cached
            and cached.get("state_key") == state_key
            and not reset_cache
        )
        if same_context:
            carried_before = self._copy_ideas(cached.get("ideas", []))
        elif cached and not reset_cache:
            carried_before = self._ideas_still_relevant(
                self._copy_ideas(cached.get("ideas", [])),
                state_signature,
                preserve_preparation=active_situation_present,
            )
        else:
            carried_before = []
        if self._cache_location_key and self._cache_location_key != location_key:
            carried_before = self._drop_generic_condition_carryover(
                carried_before,
                allow_condition=active_condition_present,
            )

        merged = self._merge_ideas(self._copy_ideas(fresh_ideas) + carried_before)
        filtered = self.filter_failed_commands(merged, failed_commands)
        filtered = self.consume_completed_commands(filtered, attempt_counts or {})

        self._cache_location_key = location_key
        self._cache_state_key = state_key
        self._cached_ideas = self._copy_ideas(filtered)
        if location_key:
            self._ideas_by_location[location_key] = {
                "state_key": state_key,
                "ideas": self._copy_ideas(filtered),
            }

        return {
            "carried_ideas_before": carried_before,
            "fresh_ideas": self._copy_ideas(fresh_ideas),
            "filtered_failed_commands": self._dedupe_commands(failed_commands or []),
            "merged_ideas": filtered,
            "carried_ideas_after": self._copy_ideas(self._cached_ideas),
            "cache_reset": not same_context,
            "cache_status": "exact" if same_context else ("reset" if reset_cache else "stale_or_new"),
        }

    def _ideas_still_relevant(self, ideas: list[dict[str, Any]],
                              state_signature: dict[str, Any] | None,
                              preserve_preparation: bool = False) -> list[dict[str, Any]]:
        if not ideas:
            return []
        visible = set((state_signature or {}).get("visible_objects", []) or [])
        observation_text = normalize_text((state_signature or {}).get("observation", ""))
        anchors = {item for item in visible if item}
        output: list[dict[str, Any]] = []
        for idea in ideas:
            if preserve_preparation and idea.get("preparation_for"):
                output.append(idea)
                continue
            if idea.get("kind") == "condition":
                output.append(idea)
                continue
            haystack = normalize_text(
                " ".join([
                    str(idea.get("situation", "")),
                    str(idea.get("reason", "")),
                    " ".join(idea.get("commands_to_try", []) or []),
                ])
            )
            if (
                any(anchor and anchor in haystack for anchor in anchors)
                or (observation_text and self._text_overlap(haystack, observation_text))
            ):
                output.append(idea)
        return output[:5]

    def consume_completed_commands(self, ideas: list[dict[str, Any]],
                                   attempt_counts: dict[str, dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Remove commands whose own room ledger already shows useful success."""
        if not ideas:
            return []
        attempt_counts = attempt_counts or {}
        output: list[dict[str, Any]] = []
        for idea in ideas or []:
            remaining = []
            already_done = list(idea.get("already_done", []) or [])
            prior_done_commands = [
                item.get("command", "")
                for item in already_done
                if isinstance(item, dict)
            ]
            for command in self._clean_commands(idea.get("commands_to_try")):
                if any(
                    commands_equivalent(command, done)
                    for done in prior_done_commands
                ):
                    continue
                ledger = self._ledger_for_equivalent(command, attempt_counts)
                if int(ledger.get("count", 0) or 0) > 0:
                    done_entry = {
                        "command": command,
                        "last_outcome": ledger.get("last_outcome", ""),
                        "last_step": ledger.get("last_step"),
                    }
                    already_done.append(done_entry)
                    prior_done_commands.append(command)
                else:
                    remaining.append(command)
            if not remaining:
                continue
            cleaned = dict(idea)
            cleaned["commands_to_try"] = remaining[:4]
            if already_done:
                cleaned["already_done"] = already_done
            output.append(cleaned)
        return output[:5]

    def _drop_generic_condition_carryover(
        self,
        ideas: list[dict[str, Any]],
        allow_condition: bool = False,
    ) -> list[dict[str, Any]]:
        """Discard generic condition probes after moving to another room."""
        if allow_condition:
            return self._copy_ideas(ideas)
        generic_commands = {
            "listen",
            "wait",
            "rest",
            "concentrate",
            "look around",
            "make noise",
        }
        output = []
        for idea in self._copy_ideas(ideas):
            commands = []
            for command in self._clean_commands(idea.get("commands_to_try")):
                command_key = self._command_key(command)
                if any(
                    command_key == generic
                    or command_key.startswith(f"{generic} ")
                    for generic in generic_commands
                ):
                    continue
                commands.append(command)
            if commands:
                idea["commands_to_try"] = commands
                output.append(idea)
        return output

    def filter_failed_commands(self, ideas: list[dict[str, Any]],
                               failed_commands: list[str]) -> list[dict[str, Any]]:
        """Remove exact failed commands from brainstorm ideas."""
        failed_commands = self._dedupe_commands(failed_commands or [])
        output: list[dict[str, Any]] = []
        for idea in ideas or []:
            commands = [
                command for command in idea.get("commands_to_try", [])
                if not any(commands_equivalent(command, failed) for failed in failed_commands)
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
            item = {
                "location": idea.get("location", "current location"),
                "situation": idea.get("situation", ""),
                "commands_to_try": idea.get("commands_to_try", []),
            }
            if idea.get("kind"):
                item["kind"] = idea.get("kind")
            if idea.get("preparation_for"):
                item["preparation_for"] = idea.get("preparation_for")
            compact.append(item)
        return json.dumps(compact, ensure_ascii=False)

    def pending_commands(self, ideas: list[dict[str, Any]]) -> list[str]:
        """Flatten pending commands from idea records for prompt context."""
        commands: list[str] = []
        for idea in ideas or []:
            commands.extend(idea.get("commands_to_try", []))
        return self._dedupe_commands(commands)

    def build_agenda(self, ideas: list[dict[str, Any]],
                     tried_records: list[dict[str, Any]] | None = None,
                     failed_records: list[dict[str, Any]] | None = None,
                     attempt_counts: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Render pending ideas plus tried/failed context for the main LLM.

        This is advisory context only. It makes carryover state legible without
        forcing the action selector to try every pending command.
        """
        pending_ideas = self.consume_completed_commands(
            self._copy_ideas(ideas),
            attempt_counts,
        )
        attempt_counts = attempt_counts or {}

        agenda: list[dict[str, Any]] = []
        for idea in pending_ideas:
            pending = self._clean_commands(idea.get("commands_to_try"))
            if not pending:
                continue
            command_statuses = []
            for command in pending:
                ledger = self._ledger_for_equivalent(command, attempt_counts)
                count = int(ledger.get("count", 0)) if ledger else 0
                if count:
                    outcome = self._clean_field(
                        ledger.get("last_outcome", "")
                    )[:60]
                    command_statuses.append(
                        f"{command}: tried x{count}, last: "
                        f"{outcome or 'outcome not recorded'}"
                    )
            entry = {
                "location": idea.get("location", "current location"),
                "situation": idea.get("situation", ""),
                "pending_commands": pending,
            }
            if command_statuses:
                entry["command_statuses"] = command_statuses
            if idea.get("already_done"):
                entry["already_done"] = idea.get("already_done", [])[:4]
            if idea.get("kind"):
                entry["kind"] = idea.get("kind")
            if idea.get("preparation_for"):
                entry["preparation_for"] = idea.get("preparation_for")
                entry["agenda_type"] = "PREPARATION"
            if idea.get("reason"):
                entry["reason"] = idea.get("reason", "")
            tried_count = len(command_statuses)
            if pending and tried_count == len(pending):
                entry["agenda_status"] = "all_commands_tried"
            elif tried_count:
                entry["agenda_status"] = "partly_tried"
            else:
                entry["agenda_status"] = "fresh"
            agenda.append(entry)

        agenda.sort(key=lambda item: (
            not bool(item.get("preparation_for")),
            item.get("agenda_status") == "all_commands_tried",
            item.get("agenda_status") == "partly_tried",
        ))

        return agenda[:5]

    def format_agenda_for_prompt(self, agenda: list[dict[str, Any]]) -> str:
        if not agenda:
            return "[]"
        return json.dumps(agenda, ensure_ascii=False)

    def _ledger_command_completed(self, ledger: dict[str, Any] | None) -> bool:
        if not ledger:
            return False
        outcomes = ledger.get("outcomes", {}) or {}
        if any(int(outcomes.get(kind, 0) or 0) > 0 for kind in ("state_change", "moved", "scored")):
            return True
        last = normalize_text(ledger.get("last_outcome", ""))
        return (
            last.startswith("changed state")
            or last.startswith("moved to")
            or last.startswith("scored")
        )

    def _text_overlap(self, left: str, right: str) -> bool:
        left_tokens = {
            tok for tok in self._normalize(left).split()
            if len(tok) >= 4 and tok not in {"with", "from", "that", "this", "there"}
        }
        right_tokens = {
            tok for tok in self._normalize(right).split()
            if len(tok) >= 4 and tok not in {"with", "from", "that", "this", "there"}
        }
        return bool(left_tokens and right_tokens and left_tokens.intersection(right_tokens))

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
            kind_match = list(re.finditer(
                r'"?kind"?\s*:\s*"([^"]+)"',
                segment,
                flags=re.IGNORECASE | re.DOTALL,
            ))
            preparation_match = list(re.finditer(
                r'"?preparation_for"?\s*:\s*"([^"]+)"',
                segment,
                flags=re.IGNORECASE | re.DOTALL,
            ))
            item = {
                "location": location_match[-1].group(1) if location_match else "current location",
                "situation": situation_match[-1].group(1) if situation_match else "",
                "reason": reason_match[-1].group(1) if reason_match else "",
                "commands_to_try": commands,
            }
            kind = self._clean_kind(kind_match[-1].group(1) if kind_match else "")
            if kind:
                item["kind"] = kind
            if preparation_match:
                item["preparation_for"] = preparation_match[-1].group(1)
            items.append(item)
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

    def _clean_kind(self, value: Any) -> str:
        kind = self._clean_field(value).lower()
        if kind in {"object", "condition"}:
            return kind
        return ""

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
                kind = self._clean_kind(idea.get("kind"))
                clean = {
                    "location": location,
                    "situation": situation or self._situation_from_commands(idea.get("commands_to_try", [])),
                    "reason": self._clean_field(idea.get("reason")),
                    "commands_to_try": [],
                }
                if kind:
                    clean["kind"] = kind
                preparation_for = self._clean_field(idea.get("preparation_for"))
                if preparation_for:
                    clean["preparation_for"] = preparation_for
                by_key[key] = clean
                merged.append(clean)
            existing = by_key[key]
            kind = self._clean_kind(idea.get("kind"))
            if kind == "condition" or (kind and not existing.get("kind")):
                existing["kind"] = kind
            if not existing.get("preparation_for") and idea.get("preparation_for"):
                existing["preparation_for"] = self._clean_field(
                    idea.get("preparation_for")
                )
            for command in self._clean_commands(idea.get("commands_to_try")):
                if self._command_key(command) not in {
                    self._command_key(c) for c in existing["commands_to_try"]
                }:
                    existing["commands_to_try"].append(command)
            existing["commands_to_try"] = existing["commands_to_try"][:4]
            if not existing.get("reason") and idea.get("reason"):
                existing["reason"] = self._clean_field(idea.get("reason"))
        return [idea for idea in merged if idea.get("commands_to_try")][:5]

    def _records_to_tried_entries(self, records: list[dict[str, Any]] | None,
                                  source: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for record in records or []:
            command = self._clean_command(record.get("command"))
            if not command:
                continue
            result = (
                record.get("result_observation")
                or record.get("observation")
                or record.get("last_observation")
                or record.get("last_outcome")
                or record.get("outcome")
                or record.get("failure_reason")
                or record.get("reason")
                or ""
            )
            entry = {
                "command": command,
                "result": self._clean_field(result)[:240],
                "source": source,
            }
            if record.get("reason"):
                entry["reason"] = self._clean_field(record.get("reason"))[:180]
            if record.get("failure_reason"):
                entry["reason"] = self._clean_field(record.get("failure_reason"))[:180]
            if record.get("outcome") and not entry.get("reason"):
                entry["reason"] = self._clean_field(record.get("outcome"))[:180]
            if record.get("location"):
                entry["location"] = self._clean_field(record.get("location"))
            entries.append(entry)
        return entries

    def _dedupe_tried_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in entries or []:
            key = self._command_key(entry.get("command", ""))
            if key and key not in seen:
                output.append(entry)
                seen.add(key)
        return output[:12]

    def _matching_tried_entries(self, pending_commands: list[str],
                                tried_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tried_entries:
            return []
        pending_keys = {self._command_key(item) for item in pending_commands}
        pending_tokens = self._command_tokens(pending_commands)
        pending_heads = self._command_head_tokens(pending_commands)
        matches: list[dict[str, Any]] = []
        for entry in tried_entries:
            command = entry.get("command", "")
            command_key = self._command_key(command)
            command_tokens = self._command_tokens([command])
            overlap = pending_tokens.intersection(command_tokens)
            if command_key in pending_keys:
                matches.append(entry)
            elif overlap and overlap.intersection(pending_heads):
                matches.append(entry)
        return matches[:6]

    def _command_tokens(self, commands: list[str]) -> set[str]:
        tokens: set[str] = set()
        stop = {
            "the", "a", "an", "to", "at", "in", "on", "with", "from",
            "under", "over", "behind", "around", "near", "into", "onto", "off",
            "out", "through", "inside", "outside", "beneath", "below", "above",
            "across", "against", "beside", "between",
            "look", "examine", "search", "take", "get", "open", "close",
            "move", "use", "read", "climb", "shake", "turn", "light",
            "switch", "push", "pull", "lift", "attack", "kill", "hit",
            "enter", "go", "walk", "head", "travel", "up", "down",
        }
        for command in commands or []:
            for token in self._normalize(command).split():
                if token and token not in stop:
                    tokens.add(token)
        return tokens

    def _command_head_tokens(self, commands: list[str]) -> set[str]:
        """Return the last meaningful token for each command as an object anchor."""
        heads: set[str] = set()
        for command in commands or []:
            meaningful = []
            command_tokens = self._command_tokens([command])
            for token in self._normalize(command).split():
                if token in command_tokens:
                    meaningful.append(token)
            if meaningful:
                heads.add(meaningful[-1])
        return heads

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
        key_signature = dict(signature)
        key_signature.pop("observation", None)
        return json.dumps(key_signature, sort_keys=True, ensure_ascii=False)

    def _attempt_state_key(self, location: str, signature: dict[str, Any] | None) -> str:
        state_key = self._state_key(signature)
        location_key = normalize_location_key(location)
        if not state_key and not location_key:
            return ""
        return f"{location_key}|{state_key}"

    def _command_key(self, command: Any) -> str:
        return normalize_command_key(command)

    def _ledger_for_equivalent(self, command: str,
                               attempt_counts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        exact = attempt_counts.get(self._command_key(command), {})
        if exact:
            return exact
        for key, record in (attempt_counts or {}).items():
            recorded_command = record.get("command", key)
            if commands_equivalent(command, recorded_command):
                return record
        return {}

    def _normalize(self, value: Any) -> str:
        return normalize_text(value)
