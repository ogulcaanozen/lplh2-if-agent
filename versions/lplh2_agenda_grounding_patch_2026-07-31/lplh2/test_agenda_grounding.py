"""Behavioral coverage for agenda grounding and object futility."""

import unittest

from .agent import LPLHAgent
from .affordance_brainstormer import AffordanceBrainstormer
from .interaction_stats import InteractionStats


class AgendaGroundingTests(unittest.TestCase):
    def test_invalid_syntax_falls_back_to_grounded_visible_object(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        result = agent._interaction_target_for_command(
            command="use knife on wooden door",
            source_generation={
                "state_snapshot_at_generation": {
                    "visible_objects": ["wooden door", "rug"],
                },
                "affordance_brainstorming": {
                    "visible_object_names": ["wooden door", "rug"],
                },
            },
            action_split=None,
            inventory_before={"knife"},
        )
        self.assertEqual(result["object_noun"], "wooden door")
        self.assertEqual(
            result["source"],
            "grounded_visible_or_inventory_fallback",
        )

    def test_five_distinct_door_failures_exhaust_existing_tracker(self):
        stats = InteractionStats()
        for command in (
            "open wooden door",
            "pry wooden door",
            "cut wooden door",
            "break wooden door",
            "search wooden door",
        ):
            stats.record(
                "r1", "wooden door", command, "invalid", 0, 1,
                observation="That does not work.",
            )
        tier = stats.tier("r1", "wooden door")
        self.assertEqual(tier["tier"], "EXHAUSTED")
        self.assertEqual(tier["distinct_no_progress_commands"], 5)

    def test_new_information_resets_but_does_not_permanently_block_exhaustion(self):
        stats = InteractionStats()
        stats.record(
            "r1", "rug", "look under rug", "info", 0, 1,
            observation="A trap door is beneath the rug.",
        )
        for command in ("lift rug", "touch rug", "inspect rug", "pull rug"):
            stats.record(
                "r1", "rug", command, "unproductive", 0, 1,
                observation="Nothing else happens.",
            )
        tier = stats.tier("r1", "rug")
        self.assertEqual(tier["tier"], "COVERED")
        self.assertEqual(tier["information_gains"], 1)
        stats.record(
            "r1", "rug", "take rug", "unproductive", 0, 1,
            observation="The rug cannot be taken.",
        )
        self.assertEqual(stats.tier("r1", "rug")["tier"], "EXHAUSTED")

    def test_repeating_one_no_progress_command_does_not_exhaust_object(self):
        stats = InteractionStats()
        for _ in range(8):
            stats.record(
                "r1", "grating", "search grating", "unproductive", 0, 1,
                observation="You find nothing unusual.",
            )
        tier = stats.tier("r1", "grating")
        self.assertEqual(tier["tier"], "COVERED")
        self.assertEqual(tier["distinct_no_progress_commands"], 1)

    def test_rejected_object_command_is_recorded_through_agent_integration(self):
        class FakeMap:
            @staticmethod
            def registry_id_for(location):
                return "room-living" if location == "Living Room" else ""

        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = FakeMap()
        agent.interaction_stats = InteractionStats()
        agent.current_epoch = 1
        detail = agent._record_interaction_outcome(
            completed_action="use knife on wooden door",
            source_generation={
                "state_snapshot_at_generation": {
                    "visible_objects": ["wooden door", "rug"],
                },
            },
            action_split=None,
            inventory_before={"knife"},
            source_memory_location="Living Room",
            attempt_ledger_detail={
                "status": "recorded",
                "outcome_class": "invalid",
            },
            reward_change=0,
            observation='I do not know the word "use".',
        )
        self.assertEqual(detail["status"], "recorded")
        self.assertEqual(detail["object_noun"], "wooden door")
        self.assertEqual(
            agent.interaction_stats.tier("room-living", "wooden door")[
                "attempts"
            ],
            1,
        )

    def test_movement_outcome_does_not_record_interaction_target(self):
        class FakeMap:
            @staticmethod
            def registry_id_for(_location):
                return "room-behind"

        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = FakeMap()
        agent.interaction_stats = InteractionStats()
        agent.current_epoch = 1
        detail = agent._record_interaction_outcome(
            completed_action="enter window",
            source_generation={
                "state_snapshot_at_generation": {
                    "visible_objects": ["window"],
                },
            },
            action_split={"objects": ["window"]},
            inventory_before=set(),
            source_memory_location="Behind House",
            attempt_ledger_detail={
                "status": "recorded",
                "outcome_class": "moved",
            },
            reward_change=10,
            observation="Kitchen",
        )
        self.assertEqual(detail["status"], "target_resolved_not_recorded")
        self.assertEqual(len(agent.interaction_stats), 0)

    def test_preparation_requires_grounded_resource(self):
        brainstormer = AffordanceBrainstormer()
        ideas = [
            {
                "location": "Living Room",
                "situation": "nailed wooden door blocks access",
                "target_object": "wooden door",
                "preparation_for": "a tool may be needed",
                "commands_to_try": ["pry wooden door", "search wooden door"],
            },
            {
                "location": "Living Room",
                "situation": "lantern may help with a dark area",
                "target_object": "lantern",
                "preparation_for": "dark area requires light",
                "preparation_resource": "lantern",
                "commands_to_try": ["take lantern", "turn on lantern"],
            },
        ]
        agenda = brainstormer.build_agenda(
            ideas,
            visible_objects=["wooden door", "lantern"],
        )
        self.assertEqual(agenda[0]["agenda_type"], "PREPARATION")
        self.assertEqual(agenda[0]["preparation_resource"], "lantern")
        door = next(item for item in agenda if item["target_object"] == "wooden door")
        self.assertNotIn("agenda_type", door)
        self.assertTrue(any(
            item["status"] == "demoted"
            and item["reason"] == "missing_preparation_resource"
            for item in brainstormer.last_preparation_validations
        ))

    def test_resource_named_in_situation_is_not_rejected(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [{
                "location": "Hall",
                "situation": "door requires a key",
                "target_object": "key",
                "preparation_for": "door requires a key",
                "preparation_resource": "key",
                "commands_to_try": ["take key"],
            }],
            visible_objects=["key", "door"],
        )
        self.assertEqual(agenda[0]["agenda_type"], "PREPARATION")

    def test_preparation_resource_can_be_inferred_from_grounded_acquisition(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [{
                "location": "Living Room",
                "situation": "a dark area may require light",
                "target_object": "lantern",
                "preparation_for": "a dark area may require light",
                "commands_to_try": ["take lantern", "turn on lantern"],
            }],
            visible_objects=["lantern"],
        )
        self.assertEqual(agenda[0]["agenda_type"], "PREPARATION")
        self.assertEqual(agenda[0]["preparation_resource"], "lantern")
        self.assertEqual(
            brainstormer.last_preparation_validations[0]["reason"],
            "inferred_grounded_resource_with_establishing_command",
        )

    def test_duplicate_command_sets_merge_before_cap(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [
                {
                    "location": "Living Room",
                    "situation": situation,
                    "target_object": "rug",
                    "commands_to_try": ["move rug", "lift rug"],
                }
                for situation in (
                    "trap door is concealed under rug",
                    "rug may hide something",
                    "rug is a movable obstruction",
                )
            ],
            visible_objects=["rug"],
        )
        self.assertEqual(len(agenda), 1)
        self.assertEqual(len(agenda[0]["supporting_situations"]), 3)
        self.assertEqual(len(brainstormer.last_agenda_deduplication), 2)

    def test_exhausted_object_sorts_after_fresh_object(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [
                {
                    "location": "Living Room",
                    "situation": "door blocks access",
                    "target_object": "wooden door",
                    "commands_to_try": ["search wooden door"],
                },
                {
                    "location": "Living Room",
                    "situation": "rug is untried",
                    "target_object": "rug",
                    "commands_to_try": ["move rug"],
                },
            ],
            visible_objects=["wooden door", "rug"],
            object_tiers={
                "wooden door": {"tier": "EXHAUSTED", "attempts": 5},
                "rug": {"tier": "FRESH", "attempts": 0},
            },
        )
        self.assertEqual(agenda[0]["target_object"], "rug")

    def test_ungrounded_idea_sorts_after_grounded_covered_object(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [
                {
                    "location": "Living Room",
                    "situation": "unclear ambient condition",
                    "commands_to_try": ["wait", "listen"],
                },
                {
                    "location": "Living Room",
                    "situation": "rug may conceal something",
                    "target_object": "rug",
                    "commands_to_try": ["move rug"],
                },
            ],
            visible_objects=["rug"],
            object_tiers={
                "rug": {"tier": "COVERED", "attempts": 1},
            },
        )
        self.assertEqual(agenda[0]["target_object"], "rug")
        self.assertNotIn("target_object", agenda[1])

    def test_preparation_ranks_by_resource_not_exhausted_blocker(self):
        brainstormer = AffordanceBrainstormer()
        agenda = brainstormer.build_agenda(
            [
                {
                    "location": "Living Room",
                    "situation": "door may require a portable light source",
                    "target_object": "wooden door",
                    "preparation_for": "door may require a portable light source",
                    "preparation_resource": "lantern",
                    "commands_to_try": ["take lantern", "open wooden door"],
                },
                {
                    "location": "Living Room",
                    "situation": "rug is untried",
                    "target_object": "rug",
                    "commands_to_try": ["move rug"],
                },
            ],
            visible_objects=["wooden door", "rug", "lantern"],
            object_tiers={
                "wooden door": {"tier": "EXHAUSTED", "attempts": 5},
                "rug": {"tier": "FRESH", "attempts": 0},
                "lantern": {"tier": "FRESH", "attempts": 0},
            },
        )
        self.assertEqual(agenda[0]["agenda_type"], "PREPARATION")
        self.assertEqual(agenda[0]["preparation_resource"], "lantern")
        self.assertEqual(agenda[0]["target_object"], "lantern")
        self.assertEqual(agenda[0]["object_tier"], "FRESH")

    def test_selection_audit_records_skipped_first_command_in_entry(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        audit = agent._agenda_selection_audit(
            "lift rug",
            [{
                "target_object": "rug",
                "target_grounded": True,
                "pending_commands": ["move rug", "lift rug"],
            }],
        )
        self.assertTrue(audit["selected_from_agenda"])
        self.assertEqual(audit["selected_command_position"], 2)
        self.assertEqual(
            audit["earlier_commands_in_selected_entry"],
            ["move rug"],
        )


if __name__ == "__main__":
    unittest.main()
