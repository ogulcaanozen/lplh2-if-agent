"""Module 1: Dynamic Knowledge Graph Map.

Builds and maintains a knowledge graph of the game world,
tracking locations, objects, directions, and relationships.
Updated after every step via LLM-based relation extraction.
"""

import json
import copy
import logging
import re

logger = logging.getLogger(__name__)


class KGMap:
    """Dynamic Knowledge Graph Map for spatial reasoning and memory.
    
    Stores the game world as a graph:
    - Nodes = locations (rooms)
    - Edges = directional connections between rooms  
    - Properties = objects, requirements per location
    """

    def __init__(self):
        self.nodes = {}           # {location_name: {objects, directions, ...}}
        self.location_aliases = {} # {normalized_location_key: display_name}
        self.current_location = None
        self.visited_rooms = []
        self.inventory = []       # items the player is carrying

    def reset(self):
        """Reset the KG-map for a new game run."""
        self.nodes = {}
        self.location_aliases = {}
        self.current_location = None
        self.visited_rooms = []
        self.inventory = []

    def update(self, triples: list, action: str = ""):
        """Update the knowledge graph with extracted triples.
        
        Args:
            triples: List of (subject, relation, object) tuples from LLM extraction
            action: The action that was just taken (for context)
        """
        if not triples:
            return

        new_location = None
        for subj, rel, obj in triples:
            if subj.strip().lower() == "you" and rel.strip().lower() == "in":
                new_location = self._ensure_node(obj.strip())
                break

        for subj, rel, obj in triples:
            rel_lower = rel.strip().lower()
            subj_clean = subj.strip()
            obj_clean = obj.strip()

            # Handle location updates: <You, in, Location>
            if subj_clean.lower() == "you" and rel_lower == "in":
                new_location = self._ensure_node(obj_clean)

            # Handle objects in location: <Location, have, object>
            elif rel_lower == "have":
                loc = self._resolve_location_subject(subj_clean, new_location)
                if loc:
                    loc = self._ensure_node(loc)
                    if (self._should_store_room_object(loc, obj_clean)
                            and obj_clean not in self.nodes[loc]["have"]):
                        self.nodes[loc]["have"].append(obj_clean)
                else:
                    self._add_object_relation(new_location or self.current_location,
                                              subj_clean, rel_lower, obj_clean)

            # Handle directional connections: <Location, direction, Destination>
            elif rel_lower in self._direction_set():
                loc = self._resolve_location_subject(subj_clean, new_location)
                if loc:
                    loc = self._ensure_node(loc)
                    direction = self._canonical_direction(rel_lower)
                    if self._is_placeholder_destination(obj_clean, direction):
                        if direction not in self.nodes[loc]["may_direction"]:
                            self.nodes[loc]["may_direction"].append(direction)
                        continue
                    destination = self._canonicalize_known_location(obj_clean)
                    self.nodes[loc]["direction"][direction] = destination
                    # Direction is now confirmed — remove from may_direction
                    may = self.nodes[loc]["may_direction"]
                    if direction in may:
                        may.remove(direction)

            # Handle requirements: <Location, need/require, action>
            elif rel_lower in ("need", "require"):
                loc = self._resolve_location_subject(subj_clean, new_location)
                if loc:
                    loc = self._ensure_node(loc)
                    if obj_clean not in self.nodes[loc].get("needs", []):
                        self.nodes[loc].setdefault("needs", []).append(obj_clean)
                else:
                    self._add_object_relation(new_location or self.current_location,
                                              subj_clean, rel_lower, obj_clean)

            # Handle object-to-object relations: <obj1, on/in, obj2>
            elif rel_lower in ("on", "in") and subj_clean.lower() != "you":
                self._add_object_relation(new_location or self.current_location,
                                          subj_clean, rel_lower, obj_clean)

            # Handle object state relations: <trap door, state/is, open>
            elif rel_lower in self._state_relation_set() and subj_clean.lower() != "you":
                self._add_object_state_relation(new_location or self.current_location,
                                                subj_clean, rel_lower, obj_clean)

        # Update current location if changed
        if new_location:
            self.current_location = new_location
            if new_location not in self.visited_rooms:
                self.visited_rooms.append(new_location)

        # Inventory is modeled separately as temp_have. If relation extraction
        # re-adds carried items to room object lists, remove those stale copies.
        for carried in self.inventory:
            self._remove_item_from_world(carried.lower())

    def take_item(self, item: str):
        """Record a successfully taken item: add to inventory, remove from room.

        Called only after the action is confirmed valid (agent.py Step 2).
        """
        item_lower = item.strip().lower()
        if not item_lower:
            return
        if item_lower not in [i.lower() for i in self.inventory]:
            self.inventory.append(item_lower)
        self._remove_item_from_world(item_lower)

    def drop_item(self, item: str):
        """Record a successfully dropped item: remove from inventory, add to room.

        Called only after the action is confirmed valid (agent.py Step 2).
        """
        item_lower = item.strip().lower()
        if not item_lower:
            return
        self.inventory = [i for i in self.inventory if i.lower() != item_lower]
        if self.current_location and self.current_location in self.nodes:
            if item_lower not in self.nodes[self.current_location]["have"]:
                self.nodes[self.current_location]["have"].append(item_lower)

    def consume_item(self, item: str):
        """Remove a consumed item from inventory without adding it to the room.

        Used for verbs that permanently destroy or transfer the item
        (eat, drink, give). The item is gone — it does not reappear in `have`.
        """
        item_lower = item.strip().lower()
        if item_lower:
            self.inventory = [i for i in self.inventory if i.lower() != item_lower]
            self._remove_item_from_world(item_lower)

    def set_inventory(self, items: list[str]):
        """Replace inventory from an authoritative inventory listing.

        Preserve existing canonical names where possible. For example, if KG
        already stores "lantern" and the game lists "brass lantern", keep the
        shorter canonical name.
        """
        aligned = []
        used_existing = set()
        existing = list(self.inventory)
        for raw_item in items or []:
            cleaned = self._clean_item_name(raw_item)
            if not cleaned:
                continue
            match = self._match_existing_inventory_name(cleaned, existing, used_existing)
            item = (match or cleaned).lower()
            if item not in [x.lower() for x in aligned]:
                aligned.append(item)
                if match:
                    used_existing.add(match.lower())
        self.inventory = aligned
        for item in aligned:
            self._remove_item_from_world(item)

    def remove_inventory_item(self, item: str, remove_from_world: bool = False):
        """Remove an item from carried inventory."""
        cleaned = self._clean_item_name(item)
        if not cleaned:
            return
        match = self._match_existing_inventory_name(cleaned, self.inventory, set())
        target = (match or cleaned).lower()
        self.inventory = [i for i in self.inventory if i.lower() != target]
        if remove_from_world:
            self._remove_item_from_world(target)

    def add_inventory_item(self, item: str):
        """Add an item to inventory and remove it from room object lists."""
        cleaned = self._clean_item_name(item)
        if not cleaned:
            return
        match = self._match_existing_inventory_name(cleaned, self.inventory, set())
        target = (match or cleaned).lower()
        if target not in [i.lower() for i in self.inventory]:
            self.inventory.append(target)
        self._remove_item_from_world(target)

    def apply_inventory_update(self, update: dict, inventory_before: list[str] = None,
                               action: str = "") -> dict:
        """Apply a structured LLM inventory update and return an audit record."""
        before = list(self.inventory)
        result = {
            "applied": False,
            "status": "noop",
            "before": before,
            "after": before,
            "authoritative": False,
            "items_now_carried": [],
            "items_added": [],
            "items_removed": [],
            "reason": "",
        }
        if not isinstance(update, dict):
            return result

        changed = self._coerce_bool(update.get("changed", False))
        result["authoritative"] = self._coerce_bool(
            update.get("authoritative", update.get("authoritative_inventory", False))
        )
        result["reason"] = str(update.get("reason", "") or "").strip()
        result["items_now_carried"] = [
            self._clean_item_name(item)
            for item in self._as_list(update.get("items_now_carried", []))
            if self._clean_item_name(item)
        ]
        result["items_added"] = [
            self._clean_item_name(item)
            for item in self._as_list(update.get("items_added", []))
            if self._clean_item_name(item)
        ]
        result["items_removed"] = [
            self._clean_item_name(item)
            for item in self._as_list(update.get("items_removed", []))
            if self._clean_item_name(item)
        ]
        if not changed:
            return result

        if result["authoritative"]:
            self.set_inventory(result["items_now_carried"])
            result["status"] = "authoritative_set"
        else:
            before_step = {
                self._clean_item_name(item)
                for item in (inventory_before or [])
            }
            added_this_step = {
                self._clean_item_name(item)
                for item in before
            } - before_step
            for item in result["items_removed"]:
                if item in added_this_step:
                    continue
                if self._is_drop_action(action):
                    self.drop_item(item)
                else:
                    self.remove_inventory_item(item)
            for item in result["items_added"]:
                self.add_inventory_item(item)
            result["status"] = "delta_applied"

        result["applied"] = True
        result["after"] = list(self.inventory)
        return result

    def _is_drop_action(self, action: str) -> bool:
        words = str(action or "").lower().strip().split()
        return bool(words and words[0] in {"drop", "discard"})

    def _resolve_location_subject(self, subject: str, new_location: str = None):
        """Return a room node for a triple subject, or None for object subjects."""
        if subject == "[Location]":
            return new_location or self.current_location
        subject_key = self._location_key(subject)
        if new_location and subject_key == self._location_key(new_location):
            return new_location
        if subject_key in self.location_aliases:
            return self.location_aliases[subject_key]
        return None

    def _add_object_relation(self, loc: str, subject: str, relation: str, obj: str):
        """Store object-object facts under the surrounding room, not as rooms."""
        if not loc:
            return
        self._ensure_node(loc)
        relation_row = {
            "subject": subject,
            "relation": relation,
            "object": obj,
        }
        if relation_row not in self.nodes[loc]["relations"]:
            self.nodes[loc]["relations"].append(relation_row)
        for item in [subject, obj]:
            if (self._should_store_room_object(loc, item)
                    and item not in self.nodes[loc]["have"]):
                self.nodes[loc]["have"].append(item)

    def _add_object_state_relation(self, loc: str, subject: str,
                                   relation: str, obj: str):
        """Store object state without treating state words as room objects."""
        if not loc or not subject:
            return
        self._ensure_node(loc)
        state = obj if relation in {"state", "is", "are", "be", "become", "becomes", "became"} else relation
        state = str(state or "").strip()
        if not state:
            return
        relation_row = {
            "subject": subject,
            "relation": "state",
            "object": state,
        }
        if relation_row not in self.nodes[loc]["relations"]:
            self.nodes[loc]["relations"].append(relation_row)
        if (self._should_store_room_object(loc, subject)
                and subject not in self.nodes[loc]["have"]):
            self.nodes[loc]["have"].append(subject)

    def _state_relation_set(self):
        """Relations that usually describe an object's state, not a new object."""
        return {
            "state", "is", "are", "be", "become", "becomes", "became",
            "open", "opened", "closed", "shut", "locked", "unlocked",
            "revealed", "moved", "lit", "unlit", "on", "off",
        }

    def _should_store_room_object(self, loc: str, item: str) -> bool:
        """Return False for room names, carried items, and empty scenery leaks."""
        item_clean = str(item or "").strip()
        if not item_clean:
            return False
        item_lower = item_clean.lower()
        if item_lower in {"you", "[location]"}:
            return False
        if loc and item_lower == loc.lower():
            return False
        if self._location_key(item_clean) in self.location_aliases:
            return False
        if any(item_lower == carried.lower() for carried in self.inventory):
            return False
        return True

    def _remove_item_from_world(self, item_lower: str):
        """Remove a carried/consumed item from room object lists and relations."""
        for node in self.nodes.values():
            node["have"] = [
                o for o in node["have"] if o.lower() != item_lower
            ]
            node["relations"] = [
                rel for rel in node.get("relations", [])
                if rel.get("subject", "").lower() != item_lower
                and rel.get("object", "").lower() != item_lower
            ]

    def _clean_item_name(self, item: str) -> str:
        text = str(item or "").lower()
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"[^a-z0-9\s-]+", " ", text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _match_existing_inventory_name(self, item: str, existing: list[str],
                                       used_existing: set[str]) -> str | None:
        item_clean = self._clean_item_name(item)
        if not item_clean:
            return None
        item_tokens = set(item_clean.split())
        best = None
        best_score = 0.0
        for candidate in existing or []:
            cand_clean = self._clean_item_name(candidate)
            if not cand_clean or cand_clean in used_existing:
                continue
            if item_clean == cand_clean:
                return candidate
            if item_clean in cand_clean or cand_clean in item_clean:
                score = min(len(item_clean), len(cand_clean)) / max(len(item_clean), len(cand_clean))
            else:
                cand_tokens = set(cand_clean.split())
                overlap = item_tokens & cand_tokens
                score = len(overlap) / max(len(item_tokens | cand_tokens), 1)
            if score > best_score:
                best = candidate
                best_score = score
        return best if best_score >= 0.34 else None

    def _as_list(self, value) -> list:
        if isinstance(value, list):
            return value
        if value in (None, ""):
            return []
        return [value]

    def _coerce_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    def _ensure_node(self, location: str):
        """Create a node if it doesn't exist and return its canonical display name."""
        display = self._clean_location_display(location)
        if not display:
            return ""
        key = self._location_key(display)
        existing = self.location_aliases.get(key)
        if existing:
            return existing
        self.location_aliases[key] = display
        if display not in self.nodes:
            self.nodes[display] = {
                "have": [],           # confirmed objects
                "direction": {},      # confirmed exits {dir: destination}
                "may_have": [],       # uncertain objects
                "may_direction": self._standard_directions(),  # all directions untried on discovery
                "needs": [],          # requirements
                "relations": [],      # object-object relations in this room
            }
        return display

    def _clean_location_display(self, location: str) -> str:
        text = re.sub(r"\s+", " ", str(location or "")).strip()
        return text

    def _location_key(self, location: str) -> str:
        """Lexical room key for casing/article/spacing normalization."""
        text = self._clean_location_display(location).lower()
        text = re.sub(r"[^a-z0-9\s-]+", " ", text)
        text = re.sub(r"\b(the|a|an)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _canonicalize_known_location(self, location: str) -> str:
        """Return canonical display if a destination room is already known."""
        display = self._clean_location_display(location)
        if not display:
            return ""
        return self.location_aliases.get(self._location_key(display), display)

    def _standard_directions(self):
        """Full-word directions populated into may_direction on room discovery."""
        return ["north", "south", "east", "west",
                "northeast", "northwest", "southeast", "southwest",
                "up", "down"]

    def _direction_set(self):
        """All valid direction strings (full words and all abbreviations)."""
        return {
            "north", "south", "east", "west",
            "northeast", "northwest", "southeast", "southwest",
            "up", "down",
            "n", "s", "e", "w",
            "ne", "nw", "se", "sw",
            "u", "d",
        }

    def _canonical_direction(self, direction: str) -> str:
        """Normalize direction abbreviations to full direction names."""
        mapping = {
            "n": "north",
            "s": "south",
            "e": "east",
            "w": "west",
            "ne": "northeast",
            "nw": "northwest",
            "se": "southeast",
            "sw": "southwest",
            "u": "up",
            "d": "down",
        }
        direction_lower = direction.strip().lower()
        return mapping.get(direction_lower, direction_lower)

    def _is_placeholder_destination(self, destination: str, direction: str) -> bool:
        """True when a direction triple has no real destination room."""
        dest = re.sub(r"\s+", " ", str(destination or "").strip().lower())
        direction = self._canonical_direction(direction)
        if not dest:
            return True
        placeholders = {
            direction,
            f"to {direction}",
            f"the {direction}",
            f"toward {direction}",
            f"towards {direction}",
            f"go {direction}",
            f"going {direction}",
            f"{direction} direction",
            f"the {direction} direction",
        }
        return dest in placeholders

    def mark_direction_tried(self, direction: str):
        """Mark a failed direction at the CURRENT location as invalid.

        Removes it from both may_direction (unverified) and direction (confirmed).
        The relation extractor sometimes produces a false direction triple, putting
        a non-existent exit into the confirmed `direction` dict. When the game then
        rejects that direction, we must purge it from both structures so the agent
        never retries it.
        """
        direction_lower = self._canonical_direction(direction)
        if self.current_location and self.current_location in self.nodes:
            node = self.nodes[self.current_location]
            may = node["may_direction"]
            if direction_lower in may:
                may.remove(direction_lower)
            # Also remove from confirmed exits if it was falsely recorded there
            if direction_lower in node["direction"]:
                del node["direction"][direction_lower]

    def mark_direction_tried_at(self, direction: str, location: str):
        """Mark a failed direction at a SPECIFIC location as invalid.

        Used when current_location has already changed (e.g. after a valid move)
        and we need to update the SOURCE room, not the destination.
        Removes from both may_direction and direction (handles false relation-extractor triples).
        """
        direction_lower = self._canonical_direction(direction)
        location = self._canonicalize_known_location(location)
        if location and location in self.nodes:
            node = self.nodes[location]
            may = node["may_direction"]
            if direction_lower in may:
                may.remove(direction_lower)
            if direction_lower in node["direction"]:
                del node["direction"][direction_lower]

    def confirm_direction(self, from_location: str, direction: str, to_location: str):
        """Record a confirmed valid exit in the source room.

        Called as a backup when the relation extractor fails to produce the
        direction triple for a successful movement command.
        """
        direction_lower = self._canonical_direction(direction)
        from_location = self._canonicalize_known_location(from_location)
        to_location = self._canonicalize_known_location(to_location)
        if from_location and from_location in self.nodes:
            node = self.nodes[from_location]
            if to_location and not self._is_placeholder_destination(to_location, direction_lower):
                node["direction"][direction_lower] = to_location
            may = node["may_direction"]
            if direction_lower in may:
                may.remove(direction_lower)

    def get_current_room_info(self) -> dict:
        """Get objects and directions for the current location."""
        if not self.current_location or self.current_location not in self.nodes:
            return {"location": self.current_location, "objects": [], "directions": {}}
        
        node = self.nodes[self.current_location]
        return {
            "location": self.current_location,
            "objects": node["have"],
            "directions": node["direction"],
            "may_direction": node.get("may_direction", []),
            "needs": node.get("needs", []),
            "relations": node.get("relations", []),
        }

    def to_prompt_string(self) -> str:
        """Serialize the KG-map as JSON for inclusion in the LLM prompt.

        The paper explicitly states the KG-map is "JSON-structured" (Limitations
        section). The field names — temp_have, have, may_have, direction,
        may_direction — match exactly those referenced in the action generation
        prompt (Table 9 Priority Usage rules).

        temp_have: player's current inventory (items on hand, highest priority).
                   Only populated for the current room node since inventory
                   items are always available to the player wherever they are.
        have:       confirmed objects present in a room.
        may_have:   objects whose presence is uncertain.
        direction:  confirmed exits {direction: destination_room}.
        may_direction: possible exits not yet verified.
        needs:      requirements to progress (e.g., "machete to go west").
        """
        return json.dumps(self.to_clean_dict(), indent=2, ensure_ascii=False)

    def to_clean_dict(self) -> dict:
        """Return prompt-facing map JSON with local state plus topology."""
        current_node = self.nodes.get(self.current_location or "", {})
        current_room_state = {
            "inventory": list(self.inventory),
            "visible_objects": self._visible_objects_with_state(self.current_location),
            "confirmed_exits": dict(current_node.get("direction", {}) or {}),
            "untried_exits": list(current_node.get("may_direction", []) or []),
            "needs_or_blockers": list(current_node.get("needs", []) or []),
        }
        if current_node.get("relations"):
            current_room_state["relations"] = list(current_node.get("relations", []))

        return {
            "current_location": self.current_location,
            "current_room_state": current_room_state,
            "navigation_graph": {
                loc: dict(data.get("direction", {}) or {})
                for loc, data in self.nodes.items()
                if data.get("direction")
            },
            "frontier": {
                loc: list(data.get("may_direction", []) or [])
                for loc, data in self.nodes.items()
                if data.get("may_direction")
            },
            "known_object_locations": self._known_object_locations(),
            "visited_rooms": list(self.visited_rooms),
        }

    def _visible_objects_with_state(self, location: str) -> list[dict]:
        if not location or location not in self.nodes:
            return []
        node = self.nodes[location]
        states: dict[str, list[str]] = {
            obj: []
            for obj in node.get("have", [])
        }
        for rel in node.get("relations", []) or []:
            subject = str(rel.get("subject", "") or "").strip()
            relation = str(rel.get("relation", "") or "").strip()
            obj = str(rel.get("object", "") or "").strip()
            if not subject:
                continue
            if self._should_store_room_object(location, subject):
                states.setdefault(subject, [])
            if relation == "state" and subject in states and obj:
                if obj not in states[subject]:
                    states[subject].append(obj)
            elif relation in {"on", "in", "under", "inside", "near"} and subject in states and obj:
                fact = f"{relation} {obj}"
                if fact not in states[subject]:
                    states[subject].append(fact)
        return [
            {"name": obj, "state": ", ".join(state_values) if state_values else ""}
            for obj, state_values in states.items()
        ]

    def _known_object_locations(self) -> dict:
        locations: dict[str, str] = {}
        carried = {item.lower() for item in self.inventory}
        for loc, data in self.nodes.items():
            for obj in data.get("have", []) or []:
                obj_clean = str(obj or "").strip()
                if not obj_clean or obj_clean.lower() in carried:
                    continue
                if obj_clean not in locations:
                    locations[obj_clean] = loc
        return locations

    def to_dict(self) -> dict:
        """Export KG-map as a dictionary (for saving)."""
        return {
            "nodes": copy.deepcopy(self.nodes),
            "current_location": self.current_location,
            "visited_rooms": list(self.visited_rooms),
            "inventory": list(self.inventory),
        }

    def num_rooms(self) -> int:
        """Number of discovered rooms."""
        return len(self.visited_rooms)
