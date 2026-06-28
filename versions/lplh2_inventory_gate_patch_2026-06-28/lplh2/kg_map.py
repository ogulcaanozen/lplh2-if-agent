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
        self.current_location = None
        self.visited_rooms = []
        self.inventory = []       # items the player is carrying

    def reset(self):
        """Reset the KG-map for a new game run."""
        self.nodes = {}
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
                new_location = obj.strip()
                self._ensure_node(new_location)
                break

        for subj, rel, obj in triples:
            rel_lower = rel.strip().lower()
            subj_clean = subj.strip()
            obj_clean = obj.strip()

            # Handle location updates: <You, in, Location>
            if subj_clean.lower() == "you" and rel_lower == "in":
                new_location = obj_clean
                self._ensure_node(new_location)

            # Handle objects in location: <Location, have, object>
            elif rel_lower == "have":
                loc = self._resolve_location_subject(subj_clean, new_location)
                if loc:
                    self._ensure_node(loc)
                    if obj_clean not in self.nodes[loc]["have"]:
                        self.nodes[loc]["have"].append(obj_clean)
                else:
                    self._add_object_relation(new_location or self.current_location,
                                              subj_clean, rel_lower, obj_clean)

            # Handle directional connections: <Location, direction, Destination>
            elif rel_lower in self._direction_set():
                loc = self._resolve_location_subject(subj_clean, new_location)
                if loc:
                    self._ensure_node(loc)
                    self.nodes[loc]["direction"][rel_lower] = obj_clean
                    # Direction is now confirmed — remove from may_direction
                    may = self.nodes[loc]["may_direction"]
                    if rel_lower in may:
                        may.remove(rel_lower)

            # Handle requirements: <Location, need/require, action>
            elif rel_lower in ("need", "require"):
                loc = self._resolve_location_subject(subj_clean, new_location)
                if loc:
                    self._ensure_node(loc)
                    if obj_clean not in self.nodes[loc].get("needs", []):
                        self.nodes[loc].setdefault("needs", []).append(obj_clean)
                else:
                    self._add_object_relation(new_location or self.current_location,
                                              subj_clean, rel_lower, obj_clean)

            # Handle object-to-object relations: <obj1, on/in, obj2>
            elif rel_lower in ("on", "in") and subj_clean.lower() != "you":
                self._add_object_relation(new_location or self.current_location,
                                          subj_clean, rel_lower, obj_clean)

        # Update current location if changed
        if new_location:
            self.current_location = new_location
            if new_location not in self.visited_rooms:
                self.visited_rooms.append(new_location)

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

    def apply_inventory_update(self, update: dict, inventory_before: list[str] = None) -> dict:
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
                self.remove_inventory_item(item)
            for item in result["items_added"]:
                self.add_inventory_item(item)
            result["status"] = "delta_applied"

        result["applied"] = True
        result["after"] = list(self.inventory)
        return result

    def _resolve_location_subject(self, subject: str, new_location: str = None):
        """Return a room node for a triple subject, or None for object subjects."""
        if subject == "[Location]":
            return new_location or self.current_location
        if new_location and subject == new_location:
            return new_location
        if subject in self.nodes or subject in self.visited_rooms:
            return subject
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
            if item and item not in self.nodes[loc]["have"]:
                self.nodes[loc]["have"].append(item)

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
        """Create a node if it doesn't exist."""
        if location not in self.nodes:
            self.nodes[location] = {
                "have": [],           # confirmed objects
                "direction": {},      # confirmed exits {dir: destination}
                "may_have": [],       # uncertain objects
                "may_direction": self._standard_directions(),  # all directions untried on discovery
                "needs": [],          # requirements
                "relations": [],      # object-object relations in this room
            }

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

    def mark_direction_tried(self, direction: str):
        """Mark a failed direction at the CURRENT location as invalid.

        Removes it from both may_direction (unverified) and direction (confirmed).
        The relation extractor sometimes produces a false direction triple, putting
        a non-existent exit into the confirmed `direction` dict. When the game then
        rejects that direction, we must purge it from both structures so the agent
        never retries it.
        """
        direction_lower = direction.strip().lower()
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
        direction_lower = direction.strip().lower()
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
        direction_lower = direction.strip().lower()
        if from_location and from_location in self.nodes:
            node = self.nodes[from_location]
            if direction_lower not in node["direction"]:
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
        map_nodes = {}
        for loc, data in self.nodes.items():
            node = {
                "temp_have": list(self.inventory) if loc == self.current_location else [],
                "have": data["have"],
                "may_have": data.get("may_have", []),
                "direction": data["direction"],
                "may_direction": data.get("may_direction", []),
            }
            if data.get("needs"):
                node["needs"] = data["needs"]
            if data.get("relations"):
                node["relations"] = data["relations"]
            map_nodes[loc] = node

        kg_json = {
            "current_location": self.current_location,
            "visited_rooms": list(self.visited_rooms),
            "map": map_nodes,
        }
        return json.dumps(kg_json, indent=2)

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
