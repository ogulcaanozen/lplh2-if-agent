"""Regression tests for preparation memory, synonym consumption, and loops."""

import json
import unittest

from .affordance_brainstormer import AffordanceBrainstormer
from .agent import LPLHAgent
from .command_keys import commands_equivalent
from .kg_map import KGMap
from .opportunity_module import SituationMemory
from .prompts import AFFORDANCE_BRAINSTORMING_PROMPT, LPLH_ACTION_GENERATION_PROMPT


class _SituationLLM:
    def __init__(self, response):
        self.response = response
        self.last_situation_prompt = "test prompt"
        self.last_situation_raw_response = response
        self.last_situation_finish_reason = "stop"

    def detect_stored_situation(self, **kwargs):
        return self.response


class LoopResistanceTests(unittest.TestCase):
    def test_command_equivalence_and_agenda_consumption(self):
        self.assertTrue(commands_equivalent("climb tree", "climb up tree"))
        self.assertTrue(commands_equivalent(
            "go through opening", "climb through opening"
        ))
        self.assertTrue(commands_equivalent("look at nest", "look in nest"))
        self.assertFalse(commands_equivalent("take egg", "drop egg"))

        brainstormer = AffordanceBrainstormer()
        ideas = [{
            "location": "Path",
            "situation": "low branches",
            "commands_to_try": ["climb up tree"],
        }]
        agenda = brainstormer.build_agenda(
            ideas,
            attempt_counts={
                "climb tree": {
                    "command": "climb tree",
                    "count": 1,
                    "last_outcome": "moved to upper area",
                    "last_step": 4,
                }
            },
        )
        self.assertEqual(agenda, [])

    def test_preparation_idea_ranks_first_and_survives_state_change(self):
        brainstormer = AffordanceBrainstormer()
        state_one = brainstormer.state_signature(
            "Workshop", ["protective coat"], [], 0, "A coat hangs here."
        )
        preparation = {
            "location": "Workshop",
            "situation": "protective coat is visible",
            "reason": "It may address the remembered hazard.",
            "preparation_for": "hazardous passage requires protection",
            "commands_to_try": ["take protective coat"],
        }
        ordinary = {
            "location": "Workshop",
            "situation": "cabinet is visible",
            "commands_to_try": ["examine cabinet"],
        }
        brainstormer.merge_with_carryover(
            "Workshop", [ordinary, preparation], [], state_one,
            active_situation_present=True,
        )
        agenda = brainstormer.build_agenda([ordinary, preparation])
        self.assertEqual(agenda[0]["agenda_type"], "PREPARATION")

        state_two = brainstormer.state_signature(
            "Workshop", [], [], 0, "The room is quiet."
        )
        merge = brainstormer.merge_with_carryover(
            "Workshop", [], [], state_two,
            active_situation_present=True,
        )
        self.assertTrue(any(
            idea.get("preparation_for") for idea in merge["merged_ideas"]
        ))

    def test_attempted_unproductive_command_is_consumed_from_agenda(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [{
                "location": "Room",
                "situation": "cabinet is visible",
                "commands_to_try": ["search cabinet"],
            }],
            attempt_counts={
                "search cabinet": {
                    "command": "search cabinet",
                    "count": 1,
                    "last_outcome": "nothing useful",
                    "outcomes": {"unproductive": 1},
                },
            },
        )
        self.assertEqual(agenda, [])

    def test_generic_condition_carryover_expires_after_room_change(self):
        brainstormer = AffordanceBrainstormer()
        chamber_state = brainstormer.state_signature(
            "Chamber", ["door"], [], 0, "A humming chamber."
        )
        ideas = [{
            "location": "Chamber",
            "kind": "condition",
            "situation": "commands sound distorted",
            "commands_to_try": ["listen", "wait", "open door"],
        }]
        brainstormer.merge_with_carryover(
            "Chamber", ideas, [], chamber_state,
        )
        hall_state = brainstormer.state_signature(
            "Hall", [], [], 0, "A quiet hall."
        )
        brainstormer.merge_with_carryover(
            "Hall", [], [], hall_state,
        )

        cleaned = brainstormer.cached_ideas_for_state(
            "Chamber",
            chamber_state,
            active_condition_present=False,
        )
        self.assertEqual(
            brainstormer.pending_commands(cleaned),
            ["open door"],
        )
        preserved = brainstormer.cached_ideas_for_state(
            "Chamber",
            chamber_state,
            active_condition_present=True,
        )
        self.assertEqual(
            brainstormer.pending_commands(preserved),
            ["listen", "wait", "open door"],
        )

    def test_uncertain_situation_is_grounded_to_entry_gateway(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap(strict_location_authority=True)
        room, _ = agent.kg_map.mint_room(
            "Landing", "Landing\nA plain landing.", epoch=1
        )
        agent.kg_map.confirm_arrival(room, "Landing\nA plain landing.")
        agent.kg_map.set_location_uncertain(
            True, entered_via="down", step_index=3
        )
        agent.situation_memory = SituationMemory()
        agent.llm = _SituationLLM(json.dumps({
            "location": "invented dark place",
            "situation": "the unseen area is dangerous without protection",
            "possible_solution": "find suitable protection",
        }))
        result = agent._detect_and_store_situation(
            "look", "It is too dark to see."
        )
        stored = result["new_stored_situation"]
        self.assertEqual(stored["location"], "Landing")
        self.assertIn("entered via 'down' from Landing", stored["situation"])

        hazard = agent._goal_hazard_context(
            action="north",
            action_valid=True,
            location_issued="Landing",
            location_after="Landing",
            was_location_uncertain=True,
            uncertain_entered_via="down",
            uncertain_last_known="Landing",
        )
        self.assertEqual(hazard["gateway"]["command"], "down")
        self.assertEqual(hazard["gateway"]["room"], "Landing")

    def test_death_history_includes_repeated_uncertain_observation(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        repeated = "The surroundings cannot be perceived."
        agent.history = [
            ("down", repeated),
            ("look", repeated),
            ("north", repeated),
            ("wait", repeated),
            ("east", "Something brushes past."),
            ("west", "You stumble."),
            ("jump", "You have died."),
        ]
        agent.step_count = 7
        rendered = agent._format_death_history(uncertainty_since_step=1)
        self.assertGreaterEqual(rendered.count(repeated), 3)
        self.assertLessEqual(rendered.count("Turn "), 10)
        context = agent._death_uncertainty_context(
            True, "down", "Landing", 1
        )
        self.assertIn("via 'down' from Landing", context)

    def test_visit_advisory_and_darkness_non_leak(self):
        kg = KGMap(strict_location_authority=True)
        first, _ = kg.mint_room("Hall", "Hall\nA hall.", epoch=1)
        second, _ = kg.mint_room("Study", "Study\nA study.", epoch=1)
        kg.confirm_arrival(first, "Hall\nA hall.")
        kg.confirm_direction(first, "east", second)
        kg.confirm_arrival(second, "Study\nA study.")
        kg.confirm_arrival(first, "Hall\nA hall.")
        context = kg.to_clean_dict(active_goal_locations=[second])
        room_state = context["current_room_state"]
        self.assertIn("your 2nd visit this epoch", room_state["visit_advisory"])
        self.assertIn("visited 1x", room_state["exit_visit_advisory"]["east"])
        self.assertIn("ACTIVE GOAL there", room_state["exit_visit_advisory"]["east"])

        kg.set_location_uncertain(True, entered_via="down", step_index=5)
        dark = json.dumps(kg.to_clean_dict(), ensure_ascii=False)
        self.assertNotIn("visit_advisory", dark)
        self.assertNotIn("visited 1x", dark)

    def test_prompts_include_preparation_visits_and_oscillation_rules(self):
        self.assertIn("PREPARATION item", LPLH_ACTION_GENERATION_PROMPT)
        self.assertIn("RECENT PATH", LPLH_ACTION_GENERATION_PROMPT)
        self.assertIn("bouncing between the same two rooms", LPLH_ACTION_GENERATION_PROMPT)
        self.assertIn("VISITED-PLACES ADVISORY", LPLH_ACTION_GENERATION_PROMPT)
        self.assertIn("PREPARATION MATCHING", AFFORDANCE_BRAINSTORMING_PROMPT)
        self.assertIn("preparation_for", AFFORDANCE_BRAINSTORMING_PROMPT)


if __name__ == "__main__":
    unittest.main()
