"""Regression checks for exact physical-room experience retrieval."""

import unittest

from .agent import LPLHAgent
from .kg_map import KGMap


class RetrievalLocationTests(unittest.TestCase):
    def _agent(self, location="Room", fingerprint="room fingerprint"):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.update([("You", "in", location)], "look")
        if fingerprint:
            agent.kg_map.room_fingerprints[location] = fingerprint
        agent.step_count = 1
        agent.current_epoch = 1
        agent.total_score = 0
        agent.earned_score_event_keys_this_epoch = set()
        agent.earned_score_location_reward_keys_this_epoch = set()
        return agent

    def _fingerprint(self, agent, location=None):
        return agent._location_fingerprint_hash(
            location or agent.kg_map.current_location
        )

    def _record(self, kind, location, fingerprint="", **metadata):
        values = {
            "kind": kind,
            "location": location,
            "location_issued": location,
            "location_fingerprint": fingerprint,
            "event_key": metadata.pop("event_key", f"{kind}:{location}"),
        }
        values.update(metadata)
        return {"text": metadata.get("text", "memory text"), "metadata": values}

    def test_death_warning_never_roams_by_text_overlap(self):
        mayor = self._agent("Mayor's house", "mayor house room")
        outside = self._agent("<< Outside >> #4", "outside crossroads")
        record = self._record(
            "death_warning",
            "<< Outside >> #4",
            self._fingerprint(outside),
            event_key="death outside",
            text="Outside the house, entering east caused a fatal ambush.",
        )
        self.assertFalse(mayor._warning_experience_relevant(record))
        self.assertTrue(outside._warning_experience_relevant(record))

    def test_achievement_matches_source_not_destination_or_neighbor(self):
        agent = self._agent("Room X", "source room")
        agent.kg_map._ensure_node("Room Y")
        agent.kg_map.nodes["Room X"]["direction"]["east"] = "Room Y"
        record = self._record(
            "achievement",
            "Room X",
            self._fingerprint(agent, "Room X"),
            location_after="Room Y",
            score_change=10,
        )
        self.assertTrue(agent._experience_location_relevant(record))

        agent.kg_map.current_location = "Room Y"
        agent.kg_map.room_fingerprints["Room Y"] = "destination room"
        self.assertFalse(agent._experience_location_relevant(record))

        agent.kg_map.current_location = "Room X"
        agent.kg_map._ensure_node("Room Z")
        agent.kg_map.nodes["Room Z"]["direction"]["west"] = "Room X"
        agent.kg_map.current_location = "Room Z"
        agent.kg_map.room_fingerprints["Room Z"] = "neighbor room"
        self.assertFalse(agent._experience_location_relevant(record))

    def test_step_one_has_no_distant_achievement_waiver(self):
        agent = self._agent("Starting Location", "detective opening room")
        local = self._record(
            "achievement",
            "Starting Location",
            self._fingerprint(agent),
            event_key="local achievement",
            score_change=10,
        )
        distant = self._record(
            "achievement",
            "Distant Room",
            "distantfingerprint",
            event_key="distant achievement",
            score_change=10,
        )
        selected = agent._select_diverse_experiences([distant, local], top_k=5)
        self.assertEqual([item["metadata"]["event_key"] for item in selected], [
            "local achievement"
        ])

    def test_fingerprint_disambiguates_label_and_stabilizes_suffix_drift(self):
        room_a = self._agent("<< Outside >> #4", "crossroads room a")
        room_b = self._agent("<< Outside >> #4", "street room b")
        same_room_new_suffix = self._agent("<< Outside >> #2", "crossroads room a")
        fingerprint_a = self._fingerprint(room_a)
        record = self._record(
            "achievement",
            "<< Outside >> #4",
            fingerprint_a,
            score_change=10,
        )

        self.assertFalse(room_b._experience_location_relevant(record))
        self.assertTrue(same_room_new_suffix._experience_location_relevant(record))

        key_a = room_a._score_event_key("gain", "east", "<< Outside >> #4", 10)
        key_b = room_b._score_event_key("gain", "east", "<< Outside >> #4", 10)
        key_same = same_room_new_suffix._score_event_key(
            "gain", "east", "<< Outside >> #2", 10
        )
        death_a = room_a._death_event_key("east", "<< Outside >> #4")
        death_b = room_b._death_event_key("east", "<< Outside >> #4")
        death_same = same_room_new_suffix._death_event_key(
            "east", "<< Outside >> #2"
        )
        reward_a = room_a._score_location_reward_key("<< Outside >> #4", 10)
        reward_b = room_b._score_location_reward_key("<< Outside >> #4", 10)
        reward_same = same_room_new_suffix._score_location_reward_key(
            "<< Outside >> #2", 10
        )
        self.assertNotEqual(key_a, key_b)
        self.assertEqual(key_a, key_same)
        self.assertNotEqual(death_a, death_b)
        self.assertEqual(death_a, death_same)
        self.assertNotEqual(reward_a, reward_b)
        self.assertEqual(reward_a, reward_same)

    def test_missing_fingerprint_falls_back_to_exact_name(self):
        agent = self._agent("Room X", "current fingerprint")
        record = self._record("clue", "Room X", "")
        self.assertTrue(agent._experience_location_relevant(record))
        record["metadata"]["location_issued"] = "Room Y"
        record["metadata"]["location"] = "Room Y"
        self.assertFalse(agent._experience_location_relevant(record))

    def test_route_filler_only_uses_route_starting_here(self):
        agent = self._agent("Room X", "source x")
        local = self._record(
            "route",
            "Room Y",
            self._fingerprint(agent),
            event_key="local route",
            location_issued="Room X",
            prev_location="Room X",
            action="east",
        )
        distant = self._record(
            "route",
            "Room Q",
            "differentfingerprint",
            event_key="distant route",
            location_issued="Room P",
            prev_location="Room P",
            action="north",
        )
        selected = agent._select_diverse_experiences([distant, local], top_k=5)
        self.assertEqual([item["metadata"]["event_key"] for item in selected], [
            "local route"
        ])


if __name__ == "__main__":
    unittest.main()
