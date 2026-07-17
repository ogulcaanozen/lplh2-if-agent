"""Behavioral tests for cross-epoch object interaction statistics."""

import unittest

from .agent import LPLHAgent
from .interaction_stats import InteractionStats
from .opportunity_module import SituationMemory


class InteractionStatsTests(unittest.TestCase):
    def test_futile_note_after_repeated_no_progress_attempts(self):
        stats = InteractionStats()
        for index in range(8):
            stats.record(
                registry_room_id="r5",
                object_noun="cliff",
                command=f"test{index} cliff",
                outcome_class="unproductive",
                reward_change=0,
                epoch=1 if index < 4 else 2,
            )
        note = stats.notes_for_visible_objects("r5", ["cliff"])
        self.assertIn("8 commands tried across 2 epoch(s)", note)
        self.assertNotIn("never", note.lower())

    def test_scoring_object_does_not_render_futile_note(self):
        stats = InteractionStats()
        for index in range(8):
            stats.record(
                registry_room_id="r2",
                object_noun="paper",
                command=f"inspect{index} paper",
                outcome_class="unproductive",
                reward_change=0,
                epoch=1,
            )
        stats.record(
            registry_room_id="r2",
            object_noun="paper",
            command="take paper",
            outcome_class="scored",
            reward_change=10,
            epoch=2,
        )
        self.assertEqual(
            stats.notes_for_visible_objects("r2", ["paper"]),
            "",
        )

    def test_epoch_reset_preserves_and_full_reset_clears(self):
        stats = InteractionStats()
        stats.record("r1", "rug", "move rug", "info", 0, 1)
        stats.reset_epoch()
        self.assertEqual(len(stats), 1)
        stats.full_reset()
        self.assertEqual(len(stats), 0)

    def test_active_goal_markers_ignore_epoch_local_situations(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.situation_memory = SituationMemory()
        agent.situation_memory.add({
            "location": "Clearing",
            "situation": "locked grating remains unresolved",
            "possible_solution": "find a way to unlock it",
        })
        self.assertEqual(agent._active_goal_locations(), [])

        agent.situation_memory.add_goal_situation(
            hazard_location="Music Store",
            hazard_fingerprint="music store signature",
            fatal_action="north",
            gateway={"room": "Outside", "command": "north"},
            hazard_text="A man blocks progress.",
            requires=["protective item"],
            item_keywords=["gun"],
            advice="prepare before returning",
            created_epoch=1,
        )
        locations = agent._active_goal_locations()
        self.assertIn("Music Store", locations)
        self.assertIn("Outside", locations)


if __name__ == "__main__":
    unittest.main()
