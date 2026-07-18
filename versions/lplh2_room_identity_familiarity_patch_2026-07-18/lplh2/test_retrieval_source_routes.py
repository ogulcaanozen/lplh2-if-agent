"""Behavioral tests for source-room retrieval and filtered local fetches."""

import unittest

from .agent import LPLHAgent
from .kg_map import KGMap


class _FakeExperienceLib:
    def __init__(self, local_record):
        self.local_record = local_record
        self.calls = []

    def retrieve_relevant_structured(self, query, **kwargs):
        self.calls.append(kwargs.get("where"))
        where = kwargs.get("where") or {}
        if "$and" in where and {"kind": "enabler"} in where["$and"]:
            return [self.local_record]
        return []


class RetrievalSourceRouteTests(unittest.TestCase):
    def _agent(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        source, source_id = agent.kg_map.mint_room(
            "Source",
            "Source\nA room.",
            epoch=1,
        )
        destination, _ = agent.kg_map.mint_room(
            "Destination",
            "Destination\nAnother room.",
            epoch=1,
        )
        agent.kg_map.current_location = source
        agent.current_epoch = 1
        agent.step_count = 3
        agent.total_score = 0
        agent.earned_score_event_keys_this_epoch = set()
        agent.earned_score_location_reward_keys_this_epoch = set()
        return agent, source, source_id, destination

    def test_route_is_actionable_only_at_its_source(self):
        agent, source, source_id, destination = self._agent()
        record = {
            "text": "east reaches Destination",
            "metadata": {
                "kind": "route",
                "event_key": "route",
                "source_location": source,
                "source_registry_id": source_id,
                "location": destination,
                "action": "east",
            },
        }
        self.assertTrue(agent._route_for_current_room(record))
        agent.kg_map.current_location = destination
        self.assertFalse(agent._route_for_current_room(record))

    def test_filtered_in_room_enabler_reaches_candidate_pool(self):
        agent, source, source_id, _ = self._agent()
        record = {
            "text": "open the window first",
            "metadata": {
                "kind": "enabler",
                "event_key": "enabler",
                "location": source,
                "location_issued": source,
                "location_registry_id": source_id,
                "enables_event_key": "reward",
                "enables_reward": 10,
            },
        }
        agent.experience_lib = _FakeExperienceLib(record)
        text = agent._retrieve_experiences_for_prompt("window")
        self.assertIn("open the window first", text)
        self.assertEqual(
            agent.last_retrieval_debug["local_filtered_fetch_counts"]["enablers"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
