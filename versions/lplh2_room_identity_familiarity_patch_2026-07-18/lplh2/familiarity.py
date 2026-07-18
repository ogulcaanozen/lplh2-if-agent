"""Persistent room familiarity for prompt-facing exploration advice."""

from __future__ import annotations

from typing import Any

from . import config


class RoomFamiliarity:
    """Track visits and grounded novelty by persistent room registry id."""

    def __init__(self):
        self._records: dict[str, dict[str, Any]] = {}

    def visit(self, registry_room_id: str, location: str, epoch: int) -> dict:
        record = self._ensure(registry_room_id, location)
        if not record:
            return {}
        record["visits_epoch"] += 1
        record["visits_total"] += 1
        record["epochs_seen"].add(int(epoch or 1))
        return self._public(record)

    def observe_objects(
        self,
        registry_room_id: str,
        location: str,
        object_nouns: list,
    ) -> list[str]:
        """Record visible nouns and return those that are new for this room."""
        record = self._ensure(registry_room_id, location)
        if not record:
            return []
        new_nouns = []
        for value in object_nouns or []:
            noun = str(
                value.get("name") if isinstance(value, dict) else value
            ).strip().lower()
            if noun and noun not in record["seen_object_nouns"]:
                record["seen_object_nouns"].add(noun)
                new_nouns.append(noun)
        if new_nouns:
            self.record_novelty(
                registry_room_id,
                location,
                "new_visible_object",
            )
        return new_nouns

    def record_novelty(
        self,
        registry_room_id: str,
        location: str,
        kind: str,
    ) -> dict:
        record = self._ensure(registry_room_id, location)
        if not record:
            return {}
        kind = str(kind or "").strip()
        if kind:
            record["novelty_this_epoch"].add(kind)
        record["last_novelty_visit_index"] = int(record["visits_epoch"])
        return self._public(record)

    def update_described_exits(
        self,
        registry_room_id: str,
        location: str,
        described_exits_untried: list,
    ):
        record = self._ensure(registry_room_id, location)
        if record:
            record["described_exits_untried"] = sorted({
                str(value).strip().lower()
                for value in (described_exits_untried or [])
                if str(value).strip()
            })

    def tier(
        self,
        registry_room_id: str,
        described_exits_untried: list | None = None,
    ) -> dict:
        record = self._records.get(str(registry_room_id or "").strip())
        if not record:
            return {
                "tier": "FRESH",
                "reason": "not visited in the current epoch",
                "visits_epoch": 0,
                "visits_total": 0,
                "exhausted_in_prior_epoch": False,
            }
        described = (
            list(described_exits_untried)
            if described_exits_untried is not None
            else list(record.get("described_exits_untried", []))
        )
        visits = int(record.get("visits_epoch", 0))
        novelty = set(record.get("novelty_this_epoch", set()))
        last_novelty = int(record.get("last_novelty_visit_index", 0))
        prior_exhausted = bool(record.get("exhausted_in_prior_epoch"))

        if visits <= 1 or described:
            reason = (
                f"{self._ordinal(max(visits, 1))} visit this epoch"
                if not described
                else (
                    f"{self._ordinal(max(visits, 1))} visit this epoch; "
                    f"untried described exits: {', '.join(described)}"
                )
            )
            tier = "FRESH"
        elif (
            visits >= config.ROOM_EXHAUSTED_VISITS
            and last_novelty <= 2
            and not described
        ) or (
            prior_exhausted
            and visits >= config.ROOM_PRIOR_EXHAUSTED_RECHECK_VISITS
            and not novelty
            and not described
        ):
            tier = "EXHAUSTED"
            if prior_exhausted and not novelty:
                reason = (
                    f"{visits} visits this epoch after earlier exhaustion, "
                    "with nothing new"
                )
            else:
                reason = (
                    f"{visits} visits this epoch, nothing new since visit "
                    f"{max(last_novelty, 2)}"
                )
        else:
            tier = "COVERED"
            if prior_exhausted and visits <= 2:
                reason = (
                    "exhausted in an earlier epoch; a brief re-check is "
                    "acceptable, then move on"
                )
            elif novelty:
                reason = (
                    f"{visits} visits this epoch; recent novelty: "
                    f"{', '.join(sorted(novelty))}"
                )
            else:
                reason = f"{visits} visits this epoch"

        return {
            "tier": tier,
            "reason": reason,
            "visits_epoch": visits,
            "visits_total": int(record.get("visits_total", 0)),
            "exhausted_in_prior_epoch": prior_exhausted,
            "novelty_this_epoch": sorted(novelty),
            "last_novelty_visit_index": last_novelty,
        }

    def reset_epoch(self):
        """Close the old epoch and clear only epoch-local counters."""
        for record in self._records.values():
            old_tier = self.tier(record["registry_room_id"])
            record["exhausted_in_prior_epoch"] = (
                old_tier.get("tier") == "EXHAUSTED"
            )
            record["visits_epoch"] = 0
            record["novelty_this_epoch"] = set()
            record["last_novelty_visit_index"] = 0
            record["described_exits_untried"] = []

    def full_reset(self):
        self._records = {}

    def records(self) -> list[dict]:
        return [self._public(record) for record in self._records.values()]

    def _ensure(self, registry_room_id: str, location: str) -> dict:
        room_id = str(registry_room_id or "").strip()
        if not room_id:
            return {}
        record = self._records.setdefault(
            room_id,
            {
                "registry_room_id": room_id,
                "location": str(location or "").strip(),
                "visits_epoch": 0,
                "visits_total": 0,
                "epochs_seen": set(),
                "novelty_this_epoch": set(),
                "last_novelty_visit_index": 0,
                "exhausted_in_prior_epoch": False,
                "seen_object_nouns": set(),
                "described_exits_untried": [],
            },
        )
        if location:
            record["location"] = str(location).strip()
        return record

    @staticmethod
    def _ordinal(value: int) -> str:
        value = max(1, int(value or 1))
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    @staticmethod
    def _public(record: dict) -> dict:
        return {
            **record,
            "epochs_seen": sorted(record.get("epochs_seen", set())),
            "novelty_this_epoch": sorted(
                record.get("novelty_this_epoch", set())
            ),
            "seen_object_nouns": sorted(record.get("seen_object_nouns", set())),
        }
