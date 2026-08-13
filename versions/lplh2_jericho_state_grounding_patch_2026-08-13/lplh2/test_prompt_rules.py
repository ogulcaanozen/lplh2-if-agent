"""Regression checks for prompt-only safety and retry rules."""

import unittest

from .prompts import (
    AFFORDANCE_BRAINSTORMING_PROMPT,
    LOSS_EXPERIENCE_SUMMARIZATION_PROMPT,
    LPLH_ACTION_GENERATION_PROMPT,
    STORED_SITUATION_DETECTION_PROMPT,
    STORED_SITUATION_RESOLUTION_PROMPT,
    WORLD_STATE_EXTRACTION_PROMPT,
)


class PromptRuleTests(unittest.TestCase):
    def test_death_warning_precedence_is_present(self):
        self.assertIn(
            "Death Warnings Outrank Familiarity Labels",
            LPLH_ACTION_GENERATION_PROMPT,
        )
        self.assertIn(
            "FRESH means unexplored, not safe",
            LPLH_ACTION_GENERATION_PROMPT,
        )

    def test_death_summary_uses_issuing_room_for_movement(self):
        self.assertIn(
            "room where the fatal command was issued",
            LOSS_EXPERIENCE_SUMMARIZATION_PROMPT,
        )
        self.assertIn(
            "Never attribute the fatal direction to the destination room",
            LOSS_EXPERIENCE_SUMMARIZATION_PROMPT,
        )

    def test_command_tier_rules_are_present(self):
        self.assertIn(
            "EXHAUSTED means it has repeatedly produced no movement",
            LPLH_ACTION_GENERATION_PROMPT,
        )
        self.assertIn(
            "prefer in this order",
            LPLH_ACTION_GENERATION_PROMPT,
        )
        self.assertIn(
            "Two-command loops",
            LPLH_ACTION_GENERATION_PROMPT,
        )
        self.assertIn(
            "is not progress",
            LPLH_ACTION_GENERATION_PROMPT,
        )
        self.assertIn(
            "Do not re-propose EXHAUSTED commands",
            AFFORDANCE_BRAINSTORMING_PROMPT,
        )

    def test_side_reward_advice_is_present(self):
        self.assertIn(
            "side reward one or two hops off your current route",
            LPLH_ACTION_GENERATION_PROMPT,
        )

    def test_arrival_world_state_and_source_grounding_rules_are_present(self):
        self.assertIn(
            "forced:first_confirmed_room_visit",
            WORLD_STATE_EXTRACTION_PROMPT,
        )
        self.assertIn(
            "Do not invent or return a synthetic destination name",
            STORED_SITUATION_DETECTION_PROMPT,
        )
        self.assertIn(
            "last confirmed source room",
            STORED_SITUATION_RESOLUTION_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
