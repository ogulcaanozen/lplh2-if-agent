"""Behavioral tests for same-title contradiction splitting."""

import unittest

from .action_memory import FailedActionMemory, StateScopedActionMemory
from .attempt_ledger import AttemptLedger
from .kg_map import KGMap
from .llm_client import LLMClient


class RoomIdentitySplitTests(unittest.TestCase):
    def _map(self):
        kg = KGMap()
        forest, _ = kg.mint_room(
            "Forest",
            "Forest\nTall trees surround this clearing.",
            epoch=1,
        )
        clearing, _ = kg.mint_room(
            "Clearing",
            "Clearing\nA broad clearing lies here.",
            epoch=1,
        )
        return kg, forest, clearing

    def test_blocked_then_open_structural_conflict_splits_source(self):
        kg, forest, clearing = self._map()
        kg.mark_direction_tried_at(
            "east",
            forest,
            message="You can't go that way.",
        )
        split = kg.confirm_direction(
            forest,
            "east",
            clearing,
            epoch=1,
            step=4,
            allow_split=True,
        )
        sibling = split["new_sibling"]
        self.assertIn("east", kg.nodes[forest]["blocked_directions"])
        self.assertEqual(kg.nodes[sibling]["direction"]["east"], clearing)
        self.assertNotEqual(sibling, forest)

    def test_confirmed_exit_then_structural_failure_splits_block(self):
        kg, forest, clearing = self._map()
        kg.confirm_direction(forest, "east", clearing)
        split = kg.split_structural_block(
            forest,
            "east",
            "You can't go that way.",
            epoch=1,
            step=5,
        )
        sibling = split["new_sibling"]
        self.assertEqual(kg.nodes[forest]["direction"]["east"], clearing)
        self.assertIn("east", kg.nodes[sibling]["blocked_directions"])

    def test_conditional_classifier_does_not_request_structural_split(self):
        client = LLMClient.__new__(LLMClient)
        client._es_client = None
        client._chat_aux_fallback = lambda *_args, **_kwargs: "conditional"
        self.assertEqual(
            client.classify_blocked_message(
                "east",
                "The window is closed.",
            ),
            "conditional",
        )
        client._chat_aux_fallback = lambda *_args, **_kwargs: "unparseable"
        self.assertEqual(
            client.classify_blocked_message("east", "Something happened."),
            "conditional",
        )

    def test_split_label_does_not_inherit_attempt_memories(self):
        kg, forest, clearing = self._map()
        kg.mark_direction_tried_at(
            "east",
            forest,
            message="You can't go that way.",
        )
        ledger = AttemptLedger()
        failed = FailedActionMemory()
        scoped = StateScopedActionMemory()
        snapshot = {
            "location": forest,
            "observation": "Forest",
            "inventory": [],
            "visible_objects": [],
            "score": 0,
        }
        ledger.record_step(
            forest, "east", "You can't go that way.", False, 0, False,
            forest, False, False, "stored_invalid", False, "state", 3, 1,
        )
        failed.record(
            forest,
            "east",
            "You can't go that way.",
            "No east exit exists.",
            snapshot,
        )
        scoped.record(
            snapshot,
            "east",
            "You can't go that way.",
            "No east exit exists.",
            "test",
        )
        sibling = kg.confirm_direction(
            forest,
            "east",
            clearing,
            epoch=1,
            step=4,
            allow_split=True,
        )["new_sibling"]
        sibling_snapshot = {**snapshot, "location": sibling}
        self.assertEqual(ledger.counts_for_location(sibling), {})
        self.assertEqual(failed.records_for_location(sibling), [])
        self.assertEqual(scoped.records_for_state(sibling_snapshot), [])
        self.assertTrue(ledger.counts_for_location(forest))

    def test_arrival_edge_breaks_same_signature_tie(self):
        kg = KGMap()
        source, _ = kg.mint_room("Path", "Path\nA narrow path.", epoch=1)
        first, _ = kg.mint_room(
            "Forest",
            "Forest\nIdentical trees.",
            epoch=1,
        )
        second, second_id = kg.mint_room(
            "Forest",
            "Forest\nIdentical trees.",
            epoch=1,
            force_new=True,
        )
        kg.record_arrival_edge(second, source, "north")
        candidates = kg.registry_candidates_for_base("Forest")
        registry_id, label = kg.arrival_edge_match(
            candidates,
            source,
            "north",
        )
        self.assertEqual(registry_id, second_id)
        self.assertEqual(label, second)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
