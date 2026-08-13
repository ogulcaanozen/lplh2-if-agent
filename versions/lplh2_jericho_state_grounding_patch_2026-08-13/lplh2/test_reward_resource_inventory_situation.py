"""Focused checks for the reward-resource, inventory, and situation patch."""

import unittest

from .opportunity_module import SituationMemory
from .prompts import (
    AFFORDANCE_BRAINSTORMING_PROMPT,
    AUXILIARY_MODULE_GATE_PROMPT,
    INVENTORY_RECONCILIATION_PROMPT,
    LPLH_ACTION_GENERATION_PROMPT,
    STORED_SITUATION_DETECTION_PROMPT,
)


class RewardResourceInventorySituationTests(unittest.TestCase):
    def test_main_and_brainstorm_prompts_check_missing_reward_resources(self):
        self.assertIn("Missing Reward Resources", LPLH_ACTION_GENERATION_PROMPT)
        self.assertIn(
            "setup command names a visible object that is not carried",
            LPLH_ACTION_GENERATION_PROMPT,
        )
        self.assertIn(
            "setup command names a visible object that is not carried",
            AFFORDANCE_BRAINSTORMING_PROMPT,
        )
        self.assertIn(
            "The setup command is the grounding evidence",
            AFFORDANCE_BRAINSTORMING_PROMPT,
        )

    def test_mixed_inventory_and_state_change_examples_are_present(self):
        for prompt in (
            AUXILIARY_MODULE_GATE_PROMPT,
            INVENTORY_RECONCILIATION_PROMPT,
        ):
            self.assertIn("Previous Action: activate lamp", prompt)
            self.assertIn("(Taken) The lamp is now on.", prompt)
        self.assertIn('"inventory": true', AUXILIARY_MODULE_GATE_PROMPT)
        self.assertIn('"world_state": true', AUXILIARY_MODULE_GATE_PROMPT)
        self.assertIn('"items_added": ["lamp"]', INVENTORY_RECONCILIATION_PROMPT)
        self.assertIn('"items_removed": []', INVENTORY_RECONCILIATION_PROMPT)

    def test_situation_provenance_persists_but_prompt_shape_stays_compact(self):
        memory = SituationMemory()
        created, stored = memory.add(
            {
                "location": "Kitchen",
                "situation": "dark staircase may require light",
                "possible_solution": "a light source may help",
            },
            created_epoch=2,
            created_step=93,
        )
        self.assertTrue(created)
        self.assertEqual(stored["created_epoch"], 2)
        self.assertEqual(stored["created_step"], 93)
        self.assertEqual(
            set(memory.active_situations()[0]),
            {"location", "situation", "possible_solution"},
        )

        memory.reset()
        persistent = memory.persistent_situations()[0]
        self.assertEqual(persistent["created_epoch"], 2)
        self.assertEqual(persistent["created_step"], 93)

    def test_exact_cross_epoch_duplicate_keeps_original_provenance(self):
        memory = SituationMemory()
        situation = {
            "location": "Kitchen",
            "situation": "dark staircase may require light",
            "possible_solution": "a light source may help",
        }
        memory.add(situation, created_epoch=1, created_step=20)
        memory.reset()
        created, existing = memory.add(
            situation,
            created_epoch=3,
            created_step=8,
        )
        self.assertFalse(created)
        self.assertEqual(memory.last_add_status, "duplicate")
        self.assertEqual(len(memory.persistent_situations()), 1)
        self.assertEqual(existing["created_epoch"], 1)
        self.assertEqual(existing["created_step"], 20)

    def test_resolved_situations_remain_visible_to_detector_dedup_context(self):
        memory = SituationMemory()
        _, stored = memory.add(
            {
                "location": "Workshop",
                "situation": "sealed hatch blocks descent",
                "possible_solution": "an opening method may be needed",
            },
            created_epoch=1,
            created_step=12,
        )
        self.assertTrue(memory.remove(stored))
        self.assertEqual(memory.active_situations(), [])
        detector_context = memory.persistent_situations_for_prompt()
        self.assertEqual(len(detector_context), 1)
        self.assertEqual(
            set(detector_context[0]),
            {"location", "situation", "possible_solution"},
        )
        self.assertIn(
            "same underlying object, obstacle, hazard",
            STORED_SITUATION_DETECTION_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
