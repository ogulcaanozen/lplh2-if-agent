"""Focused regression checks for visit-scoped navigation enforcement."""

import json
import unittest

from .agent import LPLHAgent
from .attempt_ledger import AttemptLedger
from .kg_map import KGMap


class _FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected navigation adjudication call")
        return self.responses.pop(0)


def _response(command):
    return (
        "|start|\n"
        f"<com>{command}</com>\n"
        '<repeat>{"is_repeat": true, "reason": "test"}</repeat>\n'
        "<rea>test</rea>\n"
        "|end|"
    )


class NavigationEnforcementTests(unittest.TestCase):
    def _agent(self, room="Room", observation="Room You are here.", responses=None):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        agent.kg_map.update([("You", "in", room)], "look")
        agent.kg_map.seed_room_fingerprint(room, observation)
        agent.llm = _FakeLLM(responses)
        agent.step_count = 10
        agent._visit_direction_failures = {}
        agent.attempt_ledger = AttemptLedger()
        return agent

    def _enforce(self, agent, command, observation, agenda=None):
        return agent._apply_navigation_enforcement(
            command=command,
            raw_llm_response=_response(command),
            repeat_check={},
            prompt="TEST ACTION PROMPT",
            observation=observation,
            affordance_agenda=agenda or [],
        )

    def test_closet_observation_and_first_probe_safety_switches(self):
        observation = "Closet You are in a closet. There is no reason to be in here. Go south."
        agent = self._agent(room="Closet", observation=observation)
        for direction in agent.kg_map._standard_directions():
            agent.kg_map.mark_direction_tried_at(direction, "Closet")

        command, _, _, debug = self._enforce(agent, "south", observation)
        self.assertEqual(command, "south")
        self.assertEqual(debug["safety_switch"], "observation_mention")
        self.assertEqual(debug["layer"], 0)

        same_description_without_direction = "Closet You are in a closet."
        fresh = self._agent(room="Closet", observation=same_description_without_direction)
        for direction in fresh.kg_map._standard_directions():
            fresh.kg_map.mark_direction_tried_at(direction, "Closet")
        command, _, _, debug = self._enforce(
            fresh,
            "south",
            same_description_without_direction,
        )
        self.assertEqual(command, "south")
        self.assertEqual(debug["safety_switch"], "first_probe")

    def test_layer_two_then_layer_three_with_nonlook_substitution(self):
        observation = "Room You are here."
        agent = self._agent(
            observation=observation,
            responses=[_response("north"), _response("north")],
        )
        agent.kg_map.mark_direction_tried_at("north", "Room")
        agent._record_visit_direction_failure(
            "Room", "north", "You can't go that way.", step=9
        )

        command, _, _, debug = self._enforce(agent, "north", observation)
        self.assertEqual(command, "north")
        self.assertEqual(debug["layer"], 2)
        self.assertTrue(debug["adjudicated_insisted"])

        agent._record_visit_direction_failure(
            "Room", "north", "You can't go that way.", step=10
        )
        command, _, _, debug = self._enforce(agent, "north", observation)
        self.assertEqual(debug["layer"], 3)
        self.assertTrue(debug["adjudicated_insisted"])
        self.assertTrue(debug["menu_offered"])
        self.assertEqual(command, debug["menu_offered"][0])
        self.assertNotEqual(command, "look")
        self.assertNotEqual(command, "north")
        self.assertEqual(len(agent.llm.calls), 2)

        command, _, _, debug = self._enforce(agent, "north", observation)
        self.assertEqual(command, "north")
        self.assertEqual(debug["layer"], 0)
        self.assertEqual(len(agent.llm.calls), 2)

    def test_identity_risk_disables_gating_for_same_title_siblings(self):
        observation = "Room You are here."
        agent = self._agent(observation=observation)
        agent.kg_map._ensure_node("Room #2")
        agent._record_visit_direction_failure(
            "Room", "north", "You can't go that way.", step=9
        )
        command, _, _, debug = self._enforce(agent, "north", observation)
        self.assertEqual(command, "north")
        self.assertEqual(debug["safety_switch"], "identity_risk")
        self.assertFalse(agent.llm.calls)

        agent.kg_map.current_location = "Room #2"
        agent._record_visit_direction_failure(
            "Room #2", "north", "You can't go that way.", step=9
        )
        command, _, _, debug = self._enforce(agent, "north", "Room #2 You are here.")
        self.assertEqual(command, "north")
        self.assertEqual(debug["safety_switch"], "identity_risk")

    def test_full_description_conflict_disables_gating_for_merged_room(self):
        original = "Closet You are in a closet. A broom is leaning in one corner."
        changed = "Closet You are in a closet. There is no reason to be in here."
        agent = self._agent(room="Closet", observation=original)
        agent._record_visit_direction_failure(
            "Closet", "south", "You can't go that way.", step=9
        )
        command, _, _, debug = self._enforce(agent, "south", changed)
        self.assertEqual(command, "south")
        self.assertEqual(debug["safety_switch"], "identity_risk")
        self.assertFalse(agent.llm.calls)

    def test_rejection_message_is_not_mistaken_for_identity_conflict(self):
        room_description = "Room You are here."
        agent = self._agent(
            observation=room_description,
            responses=[_response("north")],
        )
        agent._record_visit_direction_failure(
            "Room", "north", "You can't go that way.", step=9
        )
        command, _, _, debug = self._enforce(
            agent,
            "north",
            "You can't go that way.",
        )
        self.assertEqual(command, "north")
        self.assertEqual(debug["layer"], 2)
        self.assertEqual(debug["safety_switch"], "")

    def test_leaving_and_reentering_resets_first_probe_budget(self):
        observation = "Room You are here."
        agent = self._agent(observation=observation)
        agent._record_visit_direction_failure(
            "Room", "north", "You can't go that way.", step=9
        )
        agent._reset_direction_visit_budget("Room", "Hall")
        agent._reset_direction_visit_budget("Hall", "Room")
        agent.kg_map.current_location = "Room"
        command, _, _, debug = self._enforce(agent, "north", observation)
        self.assertEqual(command, "north")
        self.assertEqual(debug["safety_switch"], "first_probe")
        self.assertEqual(debug["failed_this_visit"], 0)

    def test_confirmed_history_disables_gating_after_rejection(self):
        observation = "Room You are here."
        agent = self._agent(observation=observation)
        agent.kg_map._ensure_node("Hall")
        agent.kg_map.confirm_direction("Room", "east", "Hall")
        agent.kg_map.mark_direction_tried_at("east", "Room")
        agent._record_visit_direction_failure(
            "Room", "east", "You can't go that way.", step=9
        )
        command, _, _, debug = self._enforce(agent, "east", observation)
        self.assertEqual(command, "east")
        self.assertEqual(debug["safety_switch"], "confirmed_history")

    def test_prompt_map_contains_enriched_blocked_exit_evidence(self):
        observation = "Room You are here."
        agent = self._agent(observation=observation)
        agent.kg_map.mark_direction_tried_at("west", "Room")
        agent._record_visit_direction_failure(
            "Room", "west", "You can't go that way.", step=8
        )
        payload = json.loads(agent._prompt_kg_map_context())
        self.assertEqual(
            payload["current_room_state"]["blocked_exits"],
            [{
                "direction": "west",
                "failed_this_visit": 1,
                "last_message": "You can't go that way.",
                "last_failed_steps_ago": 2,
            }],
        )


if __name__ == "__main__":
    unittest.main()
