"""Behavioral checks for the consolidated prompt contexts."""

import json
import unittest

from .affordance_brainstormer import AffordanceBrainstormer
from .agent import LPLHAgent
from .attempt_ledger import AttemptLedger
from .kg_map import KGMap
from .prompts import (
    AFFORDANCE_BRAINSTORMING_PROMPT,
    LPLH_ACTION_GENERATION_PROMPT,
)


class PromptSlimTests(unittest.TestCase):
    def test_main_prompt_has_one_tried_here_and_new_sections(self):
        prompt = LPLH_ACTION_GENERATION_PROMPT.format(
            kg_map="{}",
            action_pairs="none",
            experiences="none",
            known_rewards="none known yet",
            object_history_notes="none",
            stored_situations="[]",
            score=0,
            brainstormed_command_ideas="[]",
            tried_here="open door | locked | x2 *",
            recent_path="none",
            history_length=6,
            history="none",
            observation="A closed door is here.",
        )
        self.assertEqual(
            prompt.count(
                "=== TRIED HERE (COMMANDS ALREADY ATTEMPTED AT THIS LOCATION) ==="
            ),
            1,
        )
        self.assertNotIn("=== SAME-STATE TRIED COMMANDS ===", prompt)
        self.assertNotIn("=== PROBLEMATIC ATTEMPTS FROM LEDGER ===", prompt)
        self.assertIn("=== KNOWN SCORING OPPORTUNITIES", prompt)
        self.assertIn("=== OBJECT HISTORY NOTES ===", prompt)

    def test_real_tried_context_merges_ledger_and_exact_state(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.attempt_ledger = AttemptLedger()
        from .action_memory import FailedActionMemory, StateScopedActionMemory
        agent.failed_action_memory = FailedActionMemory()
        agent.state_action_memory = StateScopedActionMemory()
        snapshot = {
            "location": "Room",
            "observation": "A locked door is here.",
            "inventory": [],
            "visible_objects": ["door"],
            "score": 0,
        }
        state_key = agent._snapshot_key(snapshot)
        agent.attempt_ledger.record_step(
            location="Room",
            command="open door",
            observation="It is locked.",
            action_valid=False,
            reward_change=0,
            location_changed=False,
            destination="Room",
            inventory_changed=False,
            environment_changed=False,
            repetition_status="stored_invalid",
            terminal_defeat=False,
            state_key=state_key,
            step=2,
            epoch=1,
        )
        agent.failed_action_memory.record(
            location="Room",
            command="open door",
            observation="It is locked.",
            failure_reason="The door is locked.",
            world_signature=snapshot,
        )
        agent.state_action_memory.record(
            state_snapshot=snapshot,
            command="open door",
            result_observation="It is locked.",
            reason="The door is locked.",
            source="test",
        )
        context = agent._tried_here_context("Room", snapshot)
        self.assertEqual(context.count("open door |"), 1)
        self.assertIn("x1 *", context)

    def test_agenda_omits_old_duplicate_attempt_fields(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [{
                "location": "Room",
                "situation": "door is locked",
                "commands_to_try": ["open door", "unlock door"],
                "reason": "The door blocks progress.",
            }],
            attempt_counts={
                "open door": {
                    "command": "open door",
                    "count": 2,
                    "last_outcome": "INVALID: It is locked.",
                    "outcomes": {"invalid": 2},
                },
            },
        )
        rendered = brainstormer.format_agenda_for_prompt(agenda)
        self.assertNotIn("already_tried_here", rendered)
        self.assertIn("open door: tried x2", rendered)
        self.assertNotIn('"tried_count": 0', rendered)

    def test_kg_context_has_compact_frontier_and_no_local_relations(self):
        kg = KGMap()
        kg._ensure_node("Room A")
        kg._ensure_node("Room B")
        kg.current_location = "Room A"
        kg.nodes["Room A"]["may_direction"] = ["north"]
        kg.nodes["Room A"]["relations"] = [{
            "subject": "door", "relation": "state", "object": "open"
        }]
        kg.nodes["Room B"]["may_direction"] = ["east", "west"]
        context = kg.to_clean_dict()
        self.assertNotIn("frontier", context)
        self.assertEqual(context["rooms_with_unexplored_exits"], ["Room B"])
        self.assertNotIn("relations", context["current_room_state"])
        self.assertEqual(
            context["current_room_state"]["untried_exits"],
            ["north"],
        )

    def test_brainstorm_template_formats_with_consolidated_inputs(self):
        prompt = AFFORDANCE_BRAINSTORMING_PROMPT.format(
            location="Room",
            observation="A door is here.",
            score=0,
            visible_objects=json.dumps(["door"]),
            inventory="[]",
            recent_failed_commands="[]",
            tried_here="none",
            recent_command_outcomes="[]",
            failed_command_verbs="[]",
            pending_carryover_commands="[]",
            stored_situations="[]",
            visit_advisory="{}",
            relevant_lessons="none",
            known_rewards="none known yet",
            object_history_notes="none",
        )
        self.assertIn("Tried Here: none", prompt)
        self.assertNotIn("Known Failed Commands Here:", prompt)
        self.assertIn("Relevant Lessons: none", prompt)


if __name__ == "__main__":
    unittest.main()
