"""Module 1: Dynamic Knowledge Graph Map.

Builds and maintains a knowledge graph of the game world,
tracking locations, objects, directions, and relationships.
Updated after every step via LLM-based relation extraction.
"""

import json
import copy
import logging
import re

from .command_keys import normalize_location_key

logger = logging.getLogger(__name__)

INVERSE_DIRECTIONS = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "northeast": "southwest",
    "southwest": "northeast",
    "northwest": "southeast",
    "southeast": "northwest",
    "up": "down",
    "down": "up",
    "in": "out",
    "out": "in",
}


def canonical_room_display(raw_title: str) -> str:
    """Normalize display decoration without erasing a stable ``#N`` suffix."""
    text = re.sub(r"\s+", " ", str(raw_title or "")).strip()
    suffix_match = re.search(r"\s+(#\d+)\s*$", text)
    suffix = f" {suffix_match.group(1)}" if suffix_match else ""
    if suffix_match:
        text = text[:suffix_match.start()].rstrip()
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"[^A-Za-z0-9]+$", "", text)
    return (re.sub(r"\s+", " ", text).strip() + suffix).strip()


class KGMap:
    """Dynamic Knowledge Graph Map for spatial reasoning and memory.

    Stores the game world as a graph:
    - Nodes = locations (rooms)
    - Edges = directional connections between rooms
    - Properties = objects, requirements per location
    """

    def __init__(self, strict_location_authority: bool = False):
        self.strict_location_authority = bool(strict_location_authority)
        self.nodes = {}           # {location_name: {objects, directions, ...}}
        self.location_aliases = {} # {normalized_location_key: display_name}
        self.current_location = None
        self.visited_rooms = []
        self.inventory = []       # items the player is carrying
        self.room_fingerprints = {}  # {room_node: normalized description signature}
        self.room_description_fingerprints = {}  # full description, used only as an identity-risk signal
        self._confirmed_direction_history = {}  # {room_node: {canonical directions}}
        self.room_registry = {}  # persistent {rN: text-derived naming evidence}
        self.node_registry_ids = {}  # epoch-local {room label: rN}
        self._registry_counter = 0
        self._visit_counter = 0
        self._current_visit_id = 0
        self.visit_counts = {}
        self._visit_blocked_writes = {}
        self.location_uncertain = False
        self.location_entered_dark_via = ""
        self.location_uncertain_since_step = None

    def reset(self, full: bool = False):
        """Reset epoch state; preserve text-derived room naming across epochs."""
        self.nodes = {}
        self.location_aliases = {}
        self.current_location = None
        self.visited_rooms = []
        self.inventory = []
        self.room_fingerprints = {}
        self.room_description_fingerprints = {}
        self._confirmed_direction_history = {}
        self.node_registry_ids = {}
        self._visit_counter = 0
        self._current_visit_id = 0
        self.visit_counts = {}
        self._visit_blocked_writes = {}
        self.location_uncertain = False
        self.location_entered_dark_via = ""
        self.location_uncertain_since_step = None
        if full:
            self.room_registry = {}
            self._registry_counter = 0

    def _legacy_update(self, triples: list, action: str = ""):
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
                loc = self._resolve_location_subject_legacy(
                    subj_clean, new_location
                )
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
                loc = self._resolve_location_subject_legacy(
                    subj_clean, new_location
                )
                if loc:
                    loc = self._ensure_node(loc)
                    direction = self._canonical_direction(rel_lower)
                    if self._is_placeholder_destination(obj_clean, direction):
                        if direction not in self.nodes[loc]["may_direction"]:
                            self.nodes[loc]["may_direction"].append(direction)
                        continue
                    destination = self._canonicalize_known_location(obj_clean)
                    self.nodes[loc]["direction"][direction] = destination
                    self._remember_confirmed_direction(loc, direction)
                    # Direction is now confirmed, remove it from may_direction.
                    may = self.nodes[loc]["may_direction"]
                    if direction in may:
                        may.remove(direction)

            # Handle requirements: <Location, need/require, action>
            elif rel_lower in ("need", "require"):
                loc = self._resolve_location_subject_legacy(
                    subj_clean, new_location
                )
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

    def update(self, triples: list, action: str = ""):
        """Apply FM facts only to the room selected by text-based resolution.

        ``You in X`` and free-text destinations are intentionally inert here.
        This keeps room minting and movement under one arrival authority.
        """
        if not self.strict_location_authority:
            return self._legacy_update(triples, action)
        if not triples or not self.current_location:
            return
        for subj, rel, obj in triples:
            rel_lower = str(rel or "").strip().lower()
            subj_clean = str(subj or "").strip()
            obj_clean = str(obj or "").strip()
            if subj_clean.lower() == "you" and rel_lower == "in":
                continue
            loc = self._resolve_location_subject(subj_clean)
            if rel_lower == "have":
                if loc:
                    if (self._should_store_room_object(loc, obj_clean)
                            and obj_clean not in self.nodes[loc]["have"]):
                        self.nodes[loc]["have"].append(obj_clean)
                else:
                    self._add_object_relation(
                        self.current_location, subj_clean, rel_lower, obj_clean
                    )
            elif rel_lower in self._direction_set():
                if loc:
                    direction = self._canonical_direction(rel_lower)
                    node = self.nodes[loc]
                    if (direction not in node["direction"]
                            and direction not in node["blocked_directions"]
                            and direction not in node["may_direction"]):
                        node["may_direction"].append(direction)
            elif rel_lower in ("need", "require"):
                if loc:
                    if obj_clean not in self.nodes[loc].get("needs", []):
                        self.nodes[loc].setdefault("needs", []).append(obj_clean)
                else:
                    self._add_object_relation(
                        self.current_location, subj_clean, rel_lower, obj_clean
                    )
            elif rel_lower in ("on", "in") and subj_clean.lower() != "you":
                self._add_object_relation(
                    self.current_location, subj_clean, rel_lower, obj_clean
                )
            elif (rel_lower in self._state_relation_set()
                  and subj_clean.lower() != "you"):
                self._add_object_state_relation(
                    self.current_location, subj_clean, rel_lower, obj_clean
                )
        for carried in self.inventory:
            self._remove_item_from_world(carried.lower())

    def resolve_arrival_location(self, title: str, observation: str = "",
                                 from_location: str = "",
                                 action: str = "") -> str:
        """Return a stable room node for a movement-confirmed arrival title.

        Some games reuse titles such as "Forest" or "Clearing" for distinct
        rooms. When the title matches an existing room but the description
        fingerprint differs, allocate a suffixed node (e.g. "Clearing #2").
        """
        title = self._clean_location_display(title)
        if not title:
            return ""

        existing = self._destination_for_known_edge(from_location, action, title)
        if existing:
            return existing

        fingerprint = self._room_fingerprint(title, observation)
        description_fingerprint = self._full_room_fingerprint(title, observation)
        base_key = self._location_key(title)
        candidates = [
            loc for loc in self.nodes
            if self._base_location_key(loc) == base_key
        ]
        if not candidates:
            node = self._ensure_node(title)
            if fingerprint:
                self.room_fingerprints[node] = fingerprint
            if description_fingerprint:
                self.room_description_fingerprints[node] = description_fingerprint
            return node

        if fingerprint:
            for loc in candidates:
                if self.room_fingerprints.get(loc) == fingerprint:
                    self.room_description_fingerprints.setdefault(
                        loc,
                        description_fingerprint,
                    )
                    return loc
            for loc in candidates:
                if not self.room_fingerprints.get(loc):
                    self.room_fingerprints[loc] = fingerprint
                    if description_fingerprint:
                        self.room_description_fingerprints[loc] = description_fingerprint
                    return loc
            numbered = self._next_numbered_location(title, candidates)
            node = self._ensure_node(numbered)
            self.room_fingerprints[node] = fingerprint
            if description_fingerprint:
                self.room_description_fingerprints[node] = description_fingerprint
            return node

        return candidates[0]

    def room_candidates(self, title: str) -> list[str]:
        """Return same-base candidates, most recently visited first."""
        base_key = self._base_location_key(canonical_room_display(title))
        candidates = [
            label for label in self.nodes
            if self._base_location_key(label) == base_key
        ]
        return sorted(
            candidates,
            key=lambda label: int(self.nodes[label].get("last_visit_order", 0)),
            reverse=True,
        )

    def candidate_cards(self, title: str, limit: int = 4) -> list[dict]:
        cards = []
        for label in self.room_candidates(title)[:max(1, int(limit or 4))]:
            node = self.nodes[label]
            cards.append({
                "label": label,
                "stored_arrival_description": node.get("arrival_description", ""),
                "known_exits": dict(node.get("direction", {}) or {}),
                "blocked_directions": list(node.get("blocked_directions", []) or []),
                "known_arrival_ways": list(node.get("arrival_ways", []) or [])[-3:],
                "registry_id": self.node_registry_ids.get(label, ""),
            })
        return cards

    @staticmethod
    def entry_signatures(entry: dict) -> list[str]:
        """Read current and legacy registry signature schemas uniformly."""
        signatures = entry.get("description_signatures")
        if isinstance(signatures, list):
            return [str(value) for value in signatures if str(value)]
        legacy = str(entry.get("description_signature", "") or "")
        return [legacy] if legacy else []

    def add_signature_alias(self, registry_id: str, signature: str) -> bool:
        """Remember one resolver-confirmed textual variant for a room."""
        entry = self.room_registry.get(str(registry_id or ""))
        signature = str(signature or "").strip()
        if not entry or not signature:
            return False
        signatures = self.entry_signatures(entry)
        if signature in signatures or len(signatures) >= 6:
            return False
        signatures.append(signature)
        entry["description_signatures"] = signatures
        entry.pop("description_signature", None)
        return True

    def registry_candidate_cards(self, title: str,
                                 limit: int = 4) -> list[dict]:
        """Build resolver evidence from persistent same-title room entries."""
        candidates = self.registry_candidates_for_base(title)
        candidates.sort(key=lambda item: int(item[1].get("first_seen_epoch", 0)))
        cards = []
        for registry_id, entry in candidates[:max(1, int(limit or 4))]:
            cards.append({
                "label": entry.get("label", ""),
                "stored_arrival_description": entry.get(
                    "arrival_description", ""
                ),
                "known_exits": {},
                "blocked_directions": [],
                "known_arrival_ways": [],
                "registry_id": registry_id,
                "first_seen_epoch": int(entry.get("first_seen_epoch", 0)),
            })
        return cards

    def known_edge_evidence(self, from_location: str, action: str, title: str) -> str:
        destination = self._destination_for_known_edge(from_location, action, title)
        if not destination:
            return ""
        return (
            f"Your map previously recorded that {action} from {from_location} "
            f"leads to {destination}."
        )

    def description_signature(self, title: str, observation: str) -> str:
        """Normalize the full first paragraph of text-derived arrival evidence."""
        text = str(observation or "").strip()
        paragraph = re.split(r"\n\s*\n", text, maxsplit=1)[0]
        title_clean = re.escape(canonical_room_display(title))
        paragraph = re.sub(
            rf"^\s*{title_clean}\b", "", paragraph, flags=re.IGNORECASE
        )
        paragraph = paragraph.lower()
        paragraph = re.sub(r"[^a-z0-9\s]+", " ", paragraph)
        paragraph = re.sub(r"\b(the|a|an)\b", " ", paragraph)
        return re.sub(r"\s+", " ", paragraph).strip()[:1000]

    def mint_room(self, title: str, observation: str = "", epoch: int = 1,
                  force_new: bool = False) -> tuple[str, str]:
        """Mint a room through the persistent text-derived naming registry.

        ``force_new=True`` is reserved for contradiction repair, where the map
        has direct evidence that identical text represents distinct rooms.
        Resolver-created rooms must keep registry deduplication enabled.
        """
        base = canonical_room_display(title) or "Starting Location"
        base = re.sub(r"\s+#\d+$", "", base).strip() or "Starting Location"
        signature = self.description_signature(base, observation)
        matches = [
            (rid, entry) for rid, entry in self.room_registry.items()
            if self._base_location_key(entry.get("base", ""))
            == self._base_location_key(base)
            and signature in self.entry_signatures(entry)
        ]
        if matches and not force_new:
            registry_id, entry = matches[0]
        else:
            self._registry_counter += 1
            registry_id = f"r{self._registry_counter}"
            same_base = [
                entry for entry in self.room_registry.values()
                if self._base_location_key(entry.get("base", ""))
                == self._base_location_key(base)
            ]
            entry = {
                "base": base,
                "label": base if not same_base else f"{base} #{len(same_base) + 1}",
                "description_signatures": [signature] if signature else [],
                "arrival_description": str(observation).strip()[:1200],
                "first_seen_epoch": int(epoch or 1),
            }
            self.room_registry[registry_id] = entry
        label = self._ensure_room(entry["label"])
        self.node_registry_ids[label] = registry_id
        if observation:
            arrival_text = str(observation).strip()[:1200]
            node = self.nodes[label]
            if not node.get("arrival_description"):
                node["arrival_description"] = arrival_text
            node["last_arrival_description"] = arrival_text
            self.room_fingerprints[label] = self._room_fingerprint(base, observation)
            self.room_description_fingerprints[label] = self._full_room_fingerprint(
                base, observation
            )
        return label, registry_id

    def adopt_registry_room(self, registry_id: str,
                            observation: str = "") -> str:
        """Attach an epoch-fresh map node to a persistent room entry."""
        entry = self.room_registry.get(str(registry_id or ""))
        if not entry:
            return ""
        label = self._ensure_room(entry.get("label", ""))
        if not label:
            return ""
        self.node_registry_ids[label] = str(registry_id)
        node = self.nodes[label]
        stored_arrival = str(entry.get("arrival_description", "") or "").strip()
        if stored_arrival and not node.get("arrival_description"):
            node["arrival_description"] = stored_arrival[:1200]
        arrival_text = str(observation or "").strip()[:1200]
        if arrival_text:
            node["last_arrival_description"] = arrival_text
            self.room_fingerprints[label] = self._room_fingerprint(
                entry.get("base", label), arrival_text
            )
            self.room_description_fingerprints[label] = (
                self._full_room_fingerprint(entry.get("base", label), arrival_text)
            )
        elif stored_arrival:
            self.room_fingerprints[label] = self._room_fingerprint(
                entry.get("base", label), stored_arrival
            )
            self.room_description_fingerprints[label] = (
                self._full_room_fingerprint(entry.get("base", label), stored_arrival)
            )
        return label

    def confirm_arrival(self, location: str, observation: str = "",
                        from_location: str = "", action: str = "") -> str:
        """Set the live cursor only after grounded text arrival resolution."""
        location = self._lookup_room(location)
        if not location:
            return ""
        self._visit_counter += 1
        self._current_visit_id = self._visit_counter
        self.current_location = location
        self.location_uncertain = False
        self.location_entered_dark_via = ""
        self.location_uncertain_since_step = None
        self.visit_counts[location] = int(self.visit_counts.get(location, 0)) + 1
        if location not in self.visited_rooms:
            self.visited_rooms.append(location)
        node = self.nodes[location]
        node["last_visit_order"] = self._visit_counter
        if observation:
            arrival_text = str(observation).strip()[:1200]
            if not node.get("arrival_description"):
                node["arrival_description"] = arrival_text
            node["last_arrival_description"] = arrival_text
        if from_location and action:
            way = f"{action} from {from_location}"
            if way not in node.setdefault("arrival_ways", []):
                node["arrival_ways"].append(way)
        return location

    def set_location_uncertain(self, value: bool = True,
                               entered_via: str = "",
                               step_index: int = None):
        self.location_uncertain = bool(value)
        if self.location_uncertain:
            if self.location_uncertain_since_step is None and step_index is not None:
                self.location_uncertain_since_step = int(step_index)
            if entered_via:
                self.location_entered_dark_via = re.sub(
                    r"\s+", " ", str(entered_via).strip().lower()
                )
        else:
            self.location_entered_dark_via = ""
            self.location_uncertain_since_step = None

    def likely_way_back_from_uncertain_location(self) -> str:
        action = self.location_entered_dark_via
        if action.startswith("go "):
            action = action[3:].strip()
        direction = self._canonical_direction(action)
        return INVERSE_DIRECTIONS.get(direction, "")

    def registry_id_for(self, location: str) -> str:
        label = self._lookup_room(location)
        return self.node_registry_ids.get(label, "") if label else ""

    def registry_candidates_for_base(self, title: str) -> list[tuple[str, dict]]:
        key = self._base_location_key(title)
        return [
            (rid, copy.deepcopy(entry))
            for rid, entry in self.room_registry.items()
            if self._base_location_key(entry.get("base", "")) == key
        ]

    def seed_room_fingerprint(self, location: str, observation: str) -> str:
        """Assign an observation fingerprint to an already-created room node."""
        location = self._canonicalize_known_location(location)
        if not location or location not in self.nodes:
            return ""
        fingerprint = self._room_fingerprint(location, observation)
        if fingerprint and not self.room_fingerprints.get(location):
            self.room_fingerprints[location] = fingerprint
        description_fingerprint = self._full_room_fingerprint(location, observation)
        if description_fingerprint and not self.room_description_fingerprints.get(location):
            self.room_description_fingerprints[location] = description_fingerprint
        return self.room_fingerprints.get(location, "")

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
        (eat, drink, give). The item is gone, so it does not reappear in `have`.
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

    def apply_world_state_update(self, update: dict, default_location: str = None) -> dict:
        """Apply a structured LLM world/object-state update.

        This is intentionally separate from relation extraction. Relation
        extraction gives broad KG triples; this pass lets an LLM record durable
        state changes such as "window=open" or "rug=moved" without turning state
        words into room objects.
        """
        result = {
            "applied": False,
            "status": "noop",
            "object_state_updates": [],
            "new_objects": [],
            "removed_objects": [],
            "reason": "",
        }
        if not isinstance(update, dict):
            return result
        result["reason"] = str(update.get("reason", "") or "").strip()
        if not self._coerce_bool(update.get("changed", False)):
            result["status"] = "no_world_state_change"
            return result

        default_location = self._known_location_or_default("", default_location)
        if not default_location:
            result["status"] = "no_known_location"
            return result

        state_updates = self._as_list(update.get("object_state_updates", []))
        for item in state_updates:
            if not isinstance(item, dict):
                continue
            obj = self._clean_object_state_name(item.get("object", ""))
            state = self._clean_state_text(item.get("state", ""))
            loc = self._known_location_or_default(item.get("location"), default_location)
            if not obj or not state:
                continue
            self._replace_object_state_relation(loc, obj, state)
            result["object_state_updates"].append({
                "object": obj,
                "location": loc,
                "state": state,
            })

        for raw in self._as_list(update.get("new_objects", [])):
            obj, loc = self._coerce_world_object_entry(raw, default_location)
            if not obj:
                continue
            loc = self._known_location_or_default(loc, default_location)
            if not loc:
                continue
            loc = self._ensure_node(loc)
            if self._should_store_room_object(loc, obj) and obj not in self.nodes[loc]["have"]:
                self.nodes[loc]["have"].append(obj)
            result["new_objects"].append({"object": obj, "location": loc})

        for raw in self._as_list(update.get("removed_objects", [])):
            obj, loc = self._coerce_world_object_entry(raw, default_location)
            if not obj:
                continue
            loc = self._known_location_or_default(loc, default_location)
            if not loc:
                continue
            self._remove_room_object(obj, loc)
            result["removed_objects"].append({"object": obj, "location": loc})

        changed = bool(
            result["object_state_updates"]
            or result["new_objects"]
            or result["removed_objects"]
        )
        result["applied"] = changed
        result["status"] = "applied" if changed else "empty_update"
        return result

    def _is_drop_action(self, action: str) -> bool:
        words = str(action or "").lower().strip().split()
        return bool(words and words[0] in {"drop", "discard"})

    def _resolve_location_subject(self, subject: str, new_location: str = None):
        """Resolve FM room subjects only to the current resolved sibling."""
        current = self.current_location
        if not current or current not in self.nodes:
            return None
        subject_key = self._location_key(subject)
        placeholders = {"location", "current location", "room", "current room"}
        if subject == "[Location]" or subject_key in placeholders:
            return current
        if subject_key == self._base_location_key(current):
            return current
        return None

    def _resolve_location_subject_legacy(self, subject: str,
                                         new_location: str = None):
        """Preserve the baseline subject rules for explicit fallback mode."""
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

    def _replace_object_state_relation(self, loc: str, subject: str, state: str):
        """Replace durable state for one object in one room."""
        if not loc or not subject:
            return
        loc = self._ensure_node(loc)
        state = self._clean_state_text(state)
        if not state:
            return
        subject_key = self._clean_item_name(subject)
        self.nodes[loc]["relations"] = [
            rel for rel in self.nodes[loc].get("relations", [])
            if not (
                rel.get("relation") == "state"
                and self._clean_item_name(rel.get("subject", "")) == subject_key
            )
        ]
        relation_row = {
            "subject": subject,
            "relation": "state",
            "object": state,
        }
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

    def _remove_room_object(self, item: str, location: str):
        """Remove an object from one location only."""
        loc = self._canonicalize_known_location(location)
        if not loc or loc not in self.nodes:
            return
        item_key = self._clean_item_name(item)
        node = self.nodes[loc]
        node["have"] = [
            obj for obj in node.get("have", [])
            if self._clean_item_name(obj) != item_key
        ]
        node["relations"] = [
            rel for rel in node.get("relations", [])
            if self._clean_item_name(rel.get("subject", "")) != item_key
            and self._clean_item_name(rel.get("object", "")) != item_key
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

    def _coerce_world_object_entry(self, value, default_location: str) -> tuple[str, str]:
        if isinstance(value, dict):
            obj = self._clean_object_state_name(value.get("object", value.get("name", "")))
            loc = self._known_location_or_default(value.get("location"), default_location)
            return obj, loc
        return self._clean_object_state_name(value), default_location

    def _known_location_or_default(self, location: str, default_location: str = None) -> str:
        """Return a known room only; never let state extraction create rooms.

        Location discovery belongs to relation extraction / room-title handling.
        The world-state extractor may describe subareas such as "under rug" or
        "dark upstairs", but those should be stored under the current known room
        unless they already exist as real KG rooms.
        """
        default = self._clean_location_display(default_location or self.current_location or "")
        default_key = self._location_key(default)
        if default_key in self.location_aliases:
            default = self.location_aliases[default_key]
        elif self.current_location and self._location_key(self.current_location) in self.location_aliases:
            default = self.location_aliases[self._location_key(self.current_location)]
        else:
            default = ""

        candidate = self._clean_location_display(location)
        candidate_key = self._location_key(candidate)
        if candidate_key in self.location_aliases:
            return self.location_aliases[candidate_key]
        return default

    def _clean_object_state_name(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def _clean_state_text(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        text = text.strip(" .;:")
        return text[:120]

    def _coerce_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    def _ensure_room(self, location: str):
        """Create a room node. Call only from arrival/registry paths."""
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
                "confirmed_actions": {},  # non-cardinal transitions {command: destination}
                "blocked_directions": [],  # confirmed rejected exits in this room
                "may_have": [],       # uncertain objects
                "may_direction": self._standard_directions(),  # all directions untried on discovery
                "needs": [],          # requirements
                "relations": [],      # object-object relations in this room
                "arrival_description": "",
                "last_arrival_description": "",
                "arrival_ways": [],
                "last_visit_order": 0,
            }
            self.room_fingerprints.setdefault(display, "")
        return display

    def _lookup_room(self, location: str):
        """Return a known room label without creating anything."""
        display = self._clean_location_display(location)
        if not display:
            return ""
        if display in self.nodes:
            return display
        return self.location_aliases.get(self._location_key(display), "")

    def _ensure_node(self, location: str):
        """Compatibility alias for the legacy fallback pipeline."""
        return self._ensure_room(location)

    def _destination_for_known_edge(self, from_location: str, action: str,
                                    title: str) -> str:
        from_location = self._canonicalize_known_location(from_location)
        if not from_location or from_location not in self.nodes:
            return ""
        action_key = re.sub(r"\s+", " ", str(action or "").strip().lower())
        if not action_key:
            return ""
        node = self.nodes[from_location]
        destinations = []
        direction = self._canonical_direction(action_key)
        if direction in self._direction_set():
            destinations.append((node.get("direction", {}) or {}).get(direction, ""))
        destinations.append((node.get("confirmed_actions", {}) or {}).get(action_key, ""))
        title_key = self._location_key(title)
        for dest in destinations:
            if dest and self._base_location_key(dest) == title_key:
                return dest
        return ""

    def _room_fingerprint(self, title: str, observation: str) -> str:
        text = re.sub(r"\s+", " ", str(observation or "")).strip()
        title_clean = re.escape(self._clean_location_display(title))
        text = re.sub(rf"^{title_clean}\b", "", text, flags=re.IGNORECASE).strip()
        if not text:
            return ""
        first_sentence = re.split(r"(?<=[.!?])\s+", text)[0]
        first_sentence = first_sentence[:240].lower()
        first_sentence = re.sub(r"[^a-z0-9\s]+", " ", first_sentence)
        first_sentence = re.sub(r"\b(the|a|an)\b", " ", first_sentence)
        return re.sub(r"\s+", " ", first_sentence).strip()

    def _full_room_fingerprint(self, title: str, observation: str) -> str:
        """Normalize a fuller room description for conservative identity checks."""
        text = re.sub(r"\s+", " ", str(observation or "")).strip()
        title_clean = re.escape(self._clean_location_display(title))
        text = re.sub(rf"^{title_clean}\b", "", text, flags=re.IGNORECASE).strip().lower()
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:600]

    def _base_location_key(self, location: str) -> str:
        display = self._clean_location_display(location)
        display = re.sub(r"\s+#\d+$", "", display)
        return self._location_key(display)

    def _next_numbered_location(self, title: str, candidates: list[str]) -> str:
        base = self._clean_location_display(title)
        used = {self._clean_location_display(loc) for loc in candidates}
        idx = 2
        while f"{base} #{idx}" in used or f"{base} #{idx}" in self.nodes:
            idx += 1
        return f"{base} #{idx}"

    def _clean_location_display(self, location: str) -> str:
        return canonical_room_display(location)

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
            if direction_lower and direction_lower not in node.setdefault("blocked_directions", []):
                node["blocked_directions"].append(direction_lower)
                self._visit_blocked_writes.setdefault(
                    (self._current_visit_id, self.current_location), set()
                ).add(direction_lower)
            may = node["may_direction"]
            if direction_lower in may:
                may.remove(direction_lower)
            # Also remove from confirmed exits if it was falsely recorded there
            if direction_lower in node["direction"]:
                self._remember_confirmed_direction(self.current_location, direction_lower)
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
            if direction_lower and direction_lower not in node.setdefault("blocked_directions", []):
                node["blocked_directions"].append(direction_lower)
                self._visit_blocked_writes.setdefault(
                    (self._current_visit_id, location), set()
                ).add(direction_lower)
            may = node["may_direction"]
            if direction_lower in may:
                may.remove(direction_lower)
            if direction_lower in node["direction"]:
                self._remember_confirmed_direction(location, direction_lower)
                del node["direction"][direction_lower]

    def confirm_direction(self, from_location: str, direction: str, to_location: str,
                          epoch: int = 0, step: int = 0,
                          source_visit_id: int = None,
                          allow_split: bool = False,
                          preserve_conflict: bool = False) -> dict:
        """Record a confirmed valid exit in the source room.

        Called as a backup when the relation extractor fails to produce the
        direction triple for a successful movement command.
        """
        direction_lower = self._canonical_direction(direction)
        from_location = self._canonicalize_known_location(from_location)
        to_location = self._canonicalize_known_location(to_location)
        split_event = {}
        if from_location and from_location in self.nodes:
            node = self.nodes[from_location]
            prior_destination = (node.get("direction", {}) or {}).get(
                direction_lower, ""
            )
            edge_conflict = bool(
                prior_destination and to_location and prior_destination != to_location
            )
            blocked_conflict = direction_lower in node.get("blocked_directions", [])
            if allow_split and (edge_conflict or blocked_conflict):
                split_event = self._split_contradictory_source_room(
                    from_location=from_location,
                    direction=direction_lower,
                    to_location=to_location,
                    trigger=("edge_contradiction" if edge_conflict
                             else "blocked_success_contradiction"),
                    epoch=epoch,
                    step=step,
                    source_visit_id=(
                        self._current_visit_id
                        if source_visit_id is None else source_visit_id
                    ),
                )
                return split_event
            if preserve_conflict and (edge_conflict or blocked_conflict):
                return {
                    "epoch": int(epoch or 0),
                    "step": int(step or 0),
                    "from_room": from_location,
                    "direction": direction_lower,
                    "destination": to_location,
                    "trigger": "split_cap_reached",
                    "suppressed": True,
                }
            self._remember_confirmed_direction(from_location, direction_lower)
            if to_location and not self._is_placeholder_destination(to_location, direction_lower):
                node["direction"][direction_lower] = to_location
            may = node["may_direction"]
            if direction_lower in may:
                may.remove(direction_lower)
            if direction_lower in node.setdefault("blocked_directions", []):
                node["blocked_directions"].remove(direction_lower)
        return split_event

    def _split_contradictory_source_room(self, from_location: str,
                                         direction: str, to_location: str,
                                         trigger: str, epoch: int, step: int,
                                         source_visit_id: int) -> dict:
        """Repair one merged source belief while preserving its older facts."""
        original = self.nodes[from_location]
        description = (
            original.get("last_arrival_description")
            or original.get("arrival_description", "")
        )
        sibling, registry_id = self.mint_room(
            re.sub(r"\s+#\d+$", "", from_location),
            observation=description,
            epoch=epoch or 1,
            force_new=True,
        )
        sibling_node = self.nodes[sibling]
        sibling_node["arrival_description"] = description
        sibling_node["last_arrival_description"] = description
        sibling_node["direction"][direction] = to_location
        if direction in sibling_node.get("may_direction", []):
            sibling_node["may_direction"].remove(direction)
        if direction in sibling_node.get("blocked_directions", []):
            sibling_node["blocked_directions"].remove(direction)
        visit_key = (int(source_visit_id or 0), from_location)
        moved_blocked = []
        if direction in self._visit_blocked_writes.get(visit_key, set()):
            if direction in original.get("blocked_directions", []):
                original["blocked_directions"].remove(direction)
            moved_blocked.append(direction)
        self._remember_confirmed_direction(sibling, direction)
        return {
            "epoch": int(epoch or 0),
            "step": int(step or 0),
            "from_room": from_location,
            "new_sibling": sibling,
            "registry_id": registry_id,
            "direction": direction,
            "destination": to_location,
            "trigger": trigger,
            "moved_current_visit_blocked": moved_blocked,
        }

    def is_direction_blocked(self, location: str, direction: str) -> bool:
        """True when the game has already rejected this exit from location."""
        direction_lower = self._canonical_direction(direction)
        location = self._canonicalize_known_location(location)
        if not direction_lower or not location or location not in self.nodes:
            return False
        return direction_lower in self.nodes[location].get("blocked_directions", [])

    def _remember_confirmed_direction(self, location: str, direction: str):
        location = self._canonicalize_known_location(location)
        direction = self._canonical_direction(direction)
        if location and direction in self._standard_directions():
            self._confirmed_direction_history.setdefault(location, set()).add(direction)

    def was_direction_confirmed(self, location: str, direction: str) -> bool:
        """True when an exit is currently or historically confirmed for a node."""
        location = self._canonicalize_known_location(location)
        direction = self._canonical_direction(direction)
        if not location or location not in self.nodes or not direction:
            return False
        if direction in (self.nodes[location].get("direction", {}) or {}):
            return True
        return direction in self._confirmed_direction_history.get(location, set())

    def has_same_title_sibling(self, location: str) -> bool:
        """True when multiple KG nodes share this room's unsuffixed title."""
        location = self._canonicalize_known_location(location)
        if not location:
            return False
        base_key = self._base_location_key(location)
        return sum(
            1 for candidate in self.nodes
            if self._base_location_key(candidate) == base_key
        ) > 1

    def room_fingerprint_conflicts(self, location: str, observation: str) -> bool:
        """Flag a possible merged-room identity without mutating the map."""
        location = self._canonicalize_known_location(location)
        if not location:
            return False
        text = re.sub(r"\s+", " ", str(observation or "")).strip()
        base_title = re.sub(
            r"\s+#\d+$",
            "",
            self._clean_location_display(location),
        )
        if not base_title or not text.lower().startswith(base_title.lower()):
            return False
        stored = self.room_description_fingerprints.get(location, "")
        current = self._full_room_fingerprint(location, observation)
        return bool(stored and current and stored != current)

    def confirm_action_transition(self, from_location: str, action: str, to_location: str):
        """Record a confirmed non-cardinal transition in the source room.

        Examples: "enter window" -> Kitchen, "climb tree" -> Up a Tree.
        """
        action_key = re.sub(r"\s+", " ", str(action or "").strip().lower())
        from_location = self._canonicalize_known_location(from_location)
        to_location = self._canonicalize_known_location(to_location)
        if not action_key or not from_location or from_location not in self.nodes:
            return
        if not to_location or to_location == from_location:
            return
        self.nodes[from_location].setdefault("confirmed_actions", {})[action_key] = to_location

    def get_current_room_info(self) -> dict:
        """Get objects and directions for the current location."""
        if not self.current_location or self.current_location not in self.nodes:
            return {"location": self.current_location, "objects": [], "directions": {}}

        if self.location_uncertain:
            return {
                "location": self.current_location,
                "objects": [],
                "directions": {},
                "confirmed_actions": {},
                "may_direction": [],
                "blocked_directions": [],
                "needs": [],
                "relations": [],
            }

        node = self.nodes[self.current_location]
        return {
            "location": self.current_location,
            "objects": node["have"],
            "directions": node["direction"],
            "confirmed_actions": node.get("confirmed_actions", {}),
            "may_direction": node.get("may_direction", []),
            "blocked_directions": node.get("blocked_directions", []),
            "needs": node.get("needs", []),
            "relations": node.get("relations", []),
        }

    def to_prompt_string(self, active_goal_locations: list[str] = None) -> str:
        """Serialize the KG-map as JSON for inclusion in the LLM prompt.

        The paper explicitly states the KG-map is "JSON-structured" (Limitations
        section). The field names temp_have, have, may_have, direction, and
        may_direction match exactly those referenced in the action generation
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
        return json.dumps(
            self.to_clean_dict(active_goal_locations=active_goal_locations),
            indent=2,
            ensure_ascii=False,
        )

    def to_clean_dict(self, active_goal_locations: list[str] = None) -> dict:
        """Return prompt-facing map JSON with local state plus topology."""
        current_node = self.nodes.get(self.current_location or "", {})
        likely_way_back = self.likely_way_back_from_uncertain_location()
        goal_keys = {
            normalize_location_key(location)
            for location in (active_goal_locations or [])
            if normalize_location_key(location)
        }
        if self.location_uncertain:
            current_room_state = {
                "inventory": list(self.inventory),
                "visible_objects": [],
                "confirmed_exits": {},
                "confirmed_actions": {},
                "blocked_exits": [],
                "untried_exits": [],
                "needs_or_blockers": [],
                "darkness": True,
                "last_known_location": self.current_location,
                "entered_darkness_via": self.location_entered_dark_via,
                "likely_way_back": likely_way_back,
            }
        else:
            exit_advisory = {}
            action_advisory = {}
            for direction, destination in dict(
                    current_node.get("direction", {}) or {}).items():
                exit_advisory[direction] = self._destination_visit_advisory(
                    destination, goal_keys
                )
            for action, destination in dict(
                    current_node.get("confirmed_actions", {}) or {}).items():
                action_advisory[action] = self._destination_visit_advisory(
                    destination, goal_keys
                )
            current_room_state = {
                "inventory": list(self.inventory),
                "visible_objects": self._visible_objects_with_state(
                    self.current_location
                ),
                "confirmed_exits": dict(current_node.get("direction", {}) or {}),
                "confirmed_actions": dict(
                    current_node.get("confirmed_actions", {}) or {}
                ),
                "blocked_exits": list(
                    current_node.get("blocked_directions", []) or []
                ),
                "untried_exits": list(
                    current_node.get("may_direction", []) or []
                ),
                "needs_or_blockers": list(current_node.get("needs", []) or []),
                "visit_advisory": (
                    f"your {self._ordinal(self.visit_counts.get(self.current_location, 0))} "
                    "visit this epoch"
                ),
                "exit_visit_advisory": exit_advisory,
                "action_transition_visit_advisory": action_advisory,
            }
        hidden_location = self.current_location if self.location_uncertain else ""
        entered_via_label = self.location_entered_dark_via or "an unknown route"
        way_back_note = (
            f" The most reliable escape is the way you came: {likely_way_back}."
            if likely_way_back else ""
        )

        return {
            "current_location": self.current_location,
            "location_uncertain": self.location_uncertain,
            "location_note": (
                f"It is pitch black here. You are NOT in {self.current_location} "
                f"anymore; you moved {entered_via_label} into an unseen dark "
                f"room. You cannot see "
                f"objects or exits. A light source is required.{way_back_note}"
                if self.location_uncertain else ""
            ),
            "current_room_state": current_room_state,
            "navigation_graph": {
                loc: dict(data.get("direction", {}) or {})
                for loc, data in self.nodes.items()
                if data.get("direction") and loc != hidden_location
            },
            "action_transitions": {
                loc: dict(data.get("confirmed_actions", {}) or {})
                for loc, data in self.nodes.items()
                if data.get("confirmed_actions") and loc != hidden_location
            },
            "rooms_with_unexplored_exits": [
                loc
                for loc, data in self.nodes.items()
                if (
                    data.get("may_direction")
                    and loc != hidden_location
                    and normalize_location_key(loc)
                    != normalize_location_key(self.current_location)
                )
            ],
            "known_object_locations": self._known_object_locations(
                exclude_location=hidden_location
            ),
            "visited_rooms": list(self.visited_rooms),
        }

    def _destination_visit_advisory(self, destination: str,
                                    goal_keys: set[str]) -> str:
        destination = str(destination or "").strip()
        count = int(self.visit_counts.get(destination, 0))
        status = f"visited {count}x this epoch" if count else "unexplored"
        if normalize_location_key(destination) in goal_keys:
            status += "; ACTIVE GOAL there"
        return f"{destination} ({status})"

    @staticmethod
    def _ordinal(value: int) -> str:
        value = max(1, int(value or 1))
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

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

    def _known_object_locations(self, exclude_location: str = "") -> dict:
        locations: dict[str, str] = {}
        carried = {item.lower() for item in self.inventory}
        for loc, data in self.nodes.items():
            if exclude_location and loc == exclude_location:
                continue
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
            "room_fingerprints": dict(self.room_fingerprints),
            "room_registry": copy.deepcopy(self.room_registry),
            "node_registry_ids": dict(self.node_registry_ids),
            "location_uncertain": self.location_uncertain,
            "location_entered_dark_via": self.location_entered_dark_via,
        }

    def num_rooms(self) -> int:
        """Number of discovered rooms."""
        return len(self.visited_rooms)
