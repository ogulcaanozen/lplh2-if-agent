"""Focused checks for inventory grounding and one-shot action revision."""

import json
import unittest

from .agent import LPLHAgent
from .attempt_ledger import AttemptLedger
from .familiarity import RoomFamiliarity
from .kg_map import KGMap
from .llm_client import LLMClient
from .opportunity_module import SituationMemory
from .reward_directory import RewardDirectory


class _InventoryLLM:
    def __init__(self, update):
        self.update = update
        self.last_inventory_reconciliation_prompt = "inventory prompt"
        self.last_inventory_reconciliation_raw_response = ""
        self.last_inventory_reconciliation_finish_reason = "stop"

    def reconcile_inventory(self, **kwargs):
        response = f"|start|{json.dumps(self.update)}|end|"
        self.last_inventory_reconciliation_raw_response = response
        return response


class _RevisionLLM:
    def __init__(self, command):
        self.command = command
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "|start|\n"
            f"<com>{self.command}</com>\n"
            '<repeat>{"is_repeat": false, "reason": "grounded revision"}</repeat>\n'
            "<rea>use current evidence</rea>\n"
            "|end|"
        )


class _SequenceRevisionLLM:
    def __init__(self, commands):
        self.commands = list(commands)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.commands:
            raise AssertionError("Unexpected additional revision call")
        command = self.commands.pop(0)
        return (
            "|start|\n"
            f"<com>{command}</com>\n"
            '<repeat>{"is_repeat": false, "reason": "grounded alternative"}</repeat>\n'
            "<rea>Use grounded current evidence.</rea>\n"
            "|end|"
        )


class _WorldStateLLM:
    def __init__(self):
        self.calls = []
        self.last_world_state_extraction_prompt = "world-state prompt"
        self.last_world_state_extraction_raw_response = ""
        self.last_world_state_extraction_finish_reason = "stop"

    def extract_world_state_update(self, **kwargs):
        self.calls.append(kwargs)
        response = json.dumps({
            "changed": True,
            "object_state_updates": [],
            "new_objects": [
                {"object": "rug", "location": "Living Room"},
            ],
            "removed_objects": [],
            "reason": "The arrival observation explicitly shows a rug.",
        })
        self.last_world_state_extraction_raw_response = response
        return response


class _SituationLLM:
    def __init__(self):
        self.last_situation_prompt = "situation prompt"
        self.last_situation_raw_response = ""
        self.last_situation_finish_reason = "stop"

    def detect_stored_situation(self, **kwargs):
        response = json.dumps({
            "location": "Unseen area via up from Kitchen",
            "situation": "dark unseen area is dangerous",
            "possible_solution": "a light source may help",
        })
        self.last_situation_raw_response = response
        return response


class GroundingConsistencyTests(unittest.TestCase):
    def _inventory_agent(self, inventory, update):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.inventory = list(inventory)
        agent.llm = _InventoryLLM(update)
        return agent

    @staticmethod
    def _gate(observation):
        return {
            "status": "routed",
            "action": "inventory",
            "action_valid": True,
            "observation": observation,
            "decision": {
                "command_outcome": {"status": "accepted"},
                "inventory_reconciliation": {
                    "run": True,
                    "reason": "inventory evidence",
                    "focus": [],
                },
            },
        }

    def test_false_changed_with_delta_is_repaired_and_applied(self):
        agent = self._inventory_agent(
            ["egg", "leaflet"],
            {
                "changed": False,
                "authoritative": False,
                "items_now_carried": [],
                "items_added": [],
                "items_removed": ["egg"],
                "evidence_quote": "The egg vanishes.",
                "reason": "The egg is gone.",
            },
        )
        result = agent._apply_gate_inventory_update(
            self._gate("The egg vanishes."),
            inventory_before={"egg", "leaflet"},
        )
        self.assertTrue(result["applied"])
        self.assertTrue(result["raw_update"]["schema_repaired"])
        self.assertEqual(agent.kg_map.inventory, ["leaflet"])

    def test_false_changed_authoritative_inventory_is_applied(self):
        agent = self._inventory_agent(
            ["egg", "leaflet"],
            {
                "changed": False,
                "authoritative": True,
                "items_now_carried": ["leaflet"],
                "items_added": [],
                "items_removed": [],
                "evidence_quote": "You are carrying a leaflet.",
                "reason": "The game listed current inventory.",
            },
        )
        result = agent._apply_gate_inventory_update(
            self._gate("You are carrying a leaflet."),
            inventory_before={"egg", "leaflet"},
        )
        self.assertEqual(result["status"], "authoritative_set")
        self.assertEqual(agent.kg_map.inventory, ["leaflet"])

    def test_kg_layer_repairs_false_changed_delta_independently(self):
        kg = KGMap()
        kg.inventory = ["egg", "leaflet"]
        result = kg.apply_inventory_update({
            "changed": False,
            "authoritative": False,
            "items_removed": ["egg"],
        })
        self.assertTrue(result["schema_repaired"])
        self.assertEqual(kg.inventory, ["leaflet"])

    def test_uncorroborated_authoritative_empty_inventory_is_rejected(self):
        agent = self._inventory_agent(
            ["lantern"],
            {
                "changed": False,
                "authoritative": True,
                "items_now_carried": [],
                "items_added": [],
                "items_removed": [],
                "evidence_quote": "",
                "reason": "No list was generated.",
            },
        )
        result = agent._apply_gate_inventory_update(
            self._gate("A lantern is visible."),
            inventory_before={"lantern"},
        )
        self.assertEqual(
            result["status"],
            "skipped_uncorroborated_empty_authoritative",
        )
        self.assertEqual(agent.kg_map.inventory, ["lantern"])

    def test_grounded_authoritative_empty_inventory_clears_inventory(self):
        agent = self._inventory_agent(
            ["lantern"],
            {
                "changed": False,
                "authoritative": True,
                "items_now_carried": [],
                "items_added": [],
                "items_removed": [],
                "evidence_quote": "You are carrying nothing.",
                "reason": "The game explicitly lists an empty inventory.",
            },
        )
        result = agent._apply_gate_inventory_update(
            self._gate("You are carrying nothing."),
            inventory_before={"lantern"},
        )
        self.assertEqual(result["status"], "authoritative_set")
        self.assertEqual(agent.kg_map.inventory, [])

    def test_open_goal_keeps_relevant_enabler_after_linked_reward(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.inventory = []
        agent.situation_memory = SituationMemory()
        agent.situation_memory.add_goal_situation(
            hazard_location="Cellar",
            fatal_action="down",
            gateway={"room": "Living Room", "command": "down"},
            requires=["a light source"],
            item_keywords=["lantern"],
        )
        agent.earned_score_event_keys_this_epoch = {"reward-window"}
        agent.earned_score_location_reward_keys_this_epoch = set()
        record = {
            "metadata": {
                "kind": "enabler",
                "enables_event_key": "reward-window",
                "enabler_action": "take lantern",
            },
        }
        self.assertFalse(agent._enabler_completed_this_epoch(record))
        agent.kg_map.inventory = ["lantern"]
        self.assertTrue(agent._enabler_completed_this_epoch(record))

    def test_grounding_revision_calls_main_llm_once_and_never_substitutes(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.update([("You", "in", "Forest Path")], "look")
        agent.situation_memory = SituationMemory()
        agent.llm = _RevisionLLM("south")
        agent.step_count = 20
        agent._visit_direction_failures = {}
        command, _, _, debug = agent._apply_navigation_enforcement(
            command="climb tree",
            raw_llm_response="original",
            repeat_check={},
            prompt="action prompt",
            observation="A large tree is here.",
            affordance_agenda=[],
            tried_here=(
                "climb tree [EXHAUSTED] x12, \"no lasting change\"\n"
                "Any command not listed here has never been tried in this room."
            ),
        )
        self.assertEqual(command, "south")
        self.assertEqual(len(agent.llm.calls), 1)
        self.assertEqual(debug["revision_trigger"], "grounding_consistency")
        self.assertEqual(debug["substituted_command"], "")

    def test_unprepared_goal_gateway_gets_one_advisory_revision(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.update([("You", "in", "Living Room")], "look")
        agent.situation_memory = SituationMemory()
        agent.situation_memory.add_goal_situation(
            hazard_location="Cellar",
            fatal_action="down",
            gateway={"room": "Living Room", "command": "down"},
            requires=["a light source"],
            item_keywords=["lantern"],
        )
        agent.llm = _RevisionLLM("take lantern")
        agent.step_count = 20
        agent._visit_direction_failures = {}
        command, _, _, debug = agent._apply_navigation_enforcement(
            command="down",
            raw_llm_response="original",
            repeat_check={},
            prompt="action prompt",
            observation="A trap door leads down.",
            affordance_agenda=[],
            tried_here="none",
        )
        self.assertEqual(command, "take lantern")
        self.assertIn("preparation", debug["revision_reason"])
        self.assertEqual(len(agent.llm.calls), 1)

    def test_revision_allows_main_llm_to_insist_once(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.update([("You", "in", "Forest Path")], "look")
        agent.situation_memory = SituationMemory()
        agent.llm = _RevisionLLM("climb tree")
        agent.step_count = 20
        agent._visit_direction_failures = {}
        command, _, _, debug = agent._apply_navigation_enforcement(
            command="climb tree",
            raw_llm_response="original",
            repeat_check={},
            prompt="action prompt",
            observation="A large tree is here.",
            affordance_agenda=[],
            tried_here="climb tree [EXHAUSTED] x12, \"no lasting change\"",
        )
        self.assertEqual(command, "climb tree")
        self.assertTrue(debug["revision_insisted"])
        self.assertEqual(len(agent.llm.calls), 1)

    def test_grounding_revision_rechecks_revised_blocked_direction(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.update([("You", "in", "Forest Path")], "look")
        agent.situation_memory = SituationMemory()
        agent.llm = _SequenceRevisionLLM(["north", "south"])
        agent.step_count = 20
        agent._visit_direction_failures = {}
        agent._record_visit_direction_failure(
            "Forest Path",
            "north",
            "You can't go that way.",
            step=19,
        )

        command, _, _, debug = agent._apply_navigation_enforcement(
            command="climb tree",
            raw_llm_response="original",
            repeat_check={},
            prompt="action prompt",
            observation="A large tree is here.",
            affordance_agenda=[],
            tried_here=(
                "climb tree [EXHAUSTED] x12, \"no lasting change\"\n"
                "Any command not listed here has never been tried in this room."
            ),
        )

        self.assertEqual(command, "south")
        self.assertEqual(debug["revision_trigger"], "grounding_consistency")
        self.assertEqual(debug["revision_final_command"], "north")
        self.assertEqual(debug["direction"], "north")
        self.assertEqual(debug["layer"], 2)
        self.assertEqual(len(agent.llm.calls), 2)

    def test_reward_rider_exempts_exhausted_destination_from_revision(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        path, _ = agent.kg_map.mint_room(
            "Forest Path",
            "Forest Path\nA tree is here.",
            epoch=1,
        )
        tree, tree_id = agent.kg_map.mint_room(
            "Up a Tree",
            "Up a Tree\nA branch.",
            epoch=1,
        )
        agent.kg_map.confirm_arrival(path, "Forest Path\nA tree is here.")
        agent.kg_map.confirm_action_transition(path, "climb tree", tree)
        agent.room_familiarity = RoomFamiliarity()
        for _ in range(6):
            agent.room_familiarity.visit(tree_id, tree, 1)
        agent.reward_directory = RewardDirectory()
        agent.reward_directory.add_or_update({
            "event_key": "egg-reward",
            "points": 5,
            "location": tree,
            "scoring_command": "take egg",
        })
        agent.earned_score_event_keys_this_epoch = set()
        agent.situation_memory = SituationMemory()
        agent.llm = _RevisionLLM("south")
        agent.step_count = 20
        agent._visit_direction_failures = {}
        command, _, _, debug = agent._apply_navigation_enforcement(
            command="climb tree",
            raw_llm_response="original",
            repeat_check={},
            prompt="action prompt",
            observation="A tree is here.",
            affordance_agenda=[],
            tried_here="climb tree [EXHAUSTED] x12, \"no lasting change\"",
        )
        self.assertEqual(command, "climb tree")
        self.assertFalse(debug["triggered"])
        self.assertEqual(agent.llm.calls, [])

    def test_confirmed_transit_to_exhausted_room_is_advisory_only(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        tree, _ = agent.kg_map.mint_room(
            "Up a Tree",
            "Up a Tree\nA branch is here.",
            epoch=1,
        )
        path, path_id = agent.kg_map.mint_room(
            "Forest Path",
            "Forest Path\nA path winds north and south.",
            epoch=1,
        )
        agent.kg_map.confirm_arrival(tree, "Up a Tree\nA branch is here.")
        agent.kg_map.confirm_direction(tree, "down", path)
        agent.room_familiarity = RoomFamiliarity()
        for _ in range(6):
            agent.room_familiarity.visit(path_id, path, 1)
        agent.reward_directory = RewardDirectory()
        agent.earned_score_event_keys_this_epoch = set()
        agent.situation_memory = SituationMemory()
        agent.attempt_ledger = AttemptLedger()
        agent.llm = _RevisionLLM("look")
        agent.step_count = 20
        agent._visit_direction_failures = {}

        command, _, _, debug = agent._apply_navigation_enforcement(
            command="down",
            raw_llm_response="original",
            repeat_check={},
            prompt="action prompt",
            observation="The only way out is down.",
            affordance_agenda=[],
            tried_here="down [EXHAUSTED] x8, \"moved to Forest Path\"",
        )

        self.assertEqual(command, "down")
        self.assertFalse(debug["triggered"])
        self.assertEqual(debug["grounding_scope"], "movement_advisory_only")
        self.assertEqual(debug["known_movement_destination"], path)
        self.assertEqual(agent.llm.calls, [])

    def test_arrival_force_detects_first_visit_and_new_signature(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        kitchen, _ = agent.kg_map.mint_room(
            "Kitchen", "Kitchen\nA table is here.", epoch=1
        )
        living, _ = agent.kg_map.mint_room(
            "Living Room", "Living Room\nA rug is here.", epoch=1
        )
        agent.kg_map.confirm_arrival(
            living, "Living Room\nA rug is here.", kitchen, "west"
        )
        self.assertEqual(
            agent._arrival_world_state_force_reason(
                kitchen, {kitchen}, {}
            ),
            "first_confirmed_room_visit",
        )
        self.assertEqual(
            agent._arrival_world_state_force_reason(
                kitchen,
                {kitchen, living},
                {"signature_alias_added": True},
            ),
            "meaningfully_new_arrival_description",
        )
        self.assertEqual(
            agent._arrival_world_state_force_reason(
                kitchen, {kitchen, living}, {}
            ),
            "",
        )

    def test_forced_arrival_world_state_runs_when_gate_skips(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        living, _ = agent.kg_map.mint_room(
            "Living Room", "Living Room\nA rug is here.", epoch=1
        )
        agent.kg_map.confirm_arrival(living, "Living Room\nA rug is here.")
        agent.llm = _WorldStateLLM()
        gate = {
            "status": "routed",
            "action": "west",
            "action_valid": True,
            "observation": "Living Room\nA large oriental rug is here.",
            "decision": {
                "command_outcome": {"status": "accepted"},
                "world_state_extraction": {
                    "run": False,
                    "reason": "ordinary room description",
                    "focus": [],
                },
            },
        }

        result = agent._apply_gate_world_state_update(
            gate,
            force_reason="first_confirmed_room_visit",
        )

        self.assertTrue(result["forced"])
        self.assertTrue(result["applied"])
        self.assertIn(
            "forced:first_confirmed_room_visit",
            agent.llm.calls[0]["gate_reason"],
        )
        self.assertIn(
            "rug",
            [
                item.lower()
                for item in agent.kg_map.nodes[living].get("have", [])
            ],
        )

    def test_uncertain_situation_is_anchored_to_confirmed_source(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        kitchen, _ = agent.kg_map.mint_room(
            "Kitchen", "Kitchen\nA staircase leads up.", epoch=1
        )
        agent.kg_map.confirm_arrival(
            kitchen, "Kitchen\nA staircase leads up."
        )
        agent.kg_map.set_location_uncertain(
            True,
            entered_via="up",
            step_index=12,
        )
        agent.situation_memory = SituationMemory()
        agent.llm = _SituationLLM()

        result = agent._detect_and_store_situation(
            action="up",
            observation="It is pitch black.",
        )

        stored = result["new_stored_situation"]
        self.assertEqual(result["status"], "stored")
        self.assertEqual(stored["location"], kitchen)
        self.assertIn("after 'up' from Kitchen", stored["situation"])
        self.assertFalse(
            agent._is_provisional_location(stored["location"])
        )
        self.assertFalse(
            result["uncertain_location_grounding"][
                "synthetic_location_persisted"
            ]
        )

    def test_inventory_and_situation_resolution_prompt_call_paths_format(self):
        client = LLMClient.__new__(LLMClient)
        client._es_client = None
        client._chat_aux_fallback = (
            lambda prompt, max_new_tokens: "|start| [] |end|"
        )

        inventory_response = client.reconcile_inventory(
            location="Kitchen",
            action="take lantern",
            action_valid=True,
            command_outcome={"status": "accepted"},
            observation="Taken.",
            inventory_before=[],
            inventory=["lantern"],
            visible_objects=["table"],
        )
        self.assertEqual(inventory_response, "[]")
        self.assertIn(
            "Current Location: Kitchen",
            client.last_inventory_reconciliation_prompt,
        )

        resolution_response = client.resolve_stored_situations(
            location="Attic",
            previous_location="Kitchen",
            action="up",
            observation="Attic\nThe room is now safely lit.",
            inventory=["lantern"],
            score=10,
            reward_change=0,
            active_situations=[{
                "location": "Kitchen",
                "situation": (
                    "dark area encountered after 'up' from Kitchen; "
                    "destination not yet identified"
                ),
            }],
        )
        self.assertEqual(resolution_response, "[]")
        self.assertIn(
            "Previous Location: Kitchen",
            client.last_situation_resolution_prompt,
        )


if __name__ == "__main__":
    unittest.main()
