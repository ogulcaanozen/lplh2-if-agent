"""Focused checks for persistent repeated-death preparation goals."""

import json
import unittest

from .agent import LPLHAgent
from .kg_map import KGMap
from .opportunity_module import SituationMemory


class _FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.last_precondition_prompt = ""
        self.last_precondition_raw_response = ""

    def hypothesize_precondition(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected precondition hypothesis call")
        response = self.responses.pop(0)
        self.last_precondition_prompt = json.dumps(kwargs, ensure_ascii=False)
        self.last_precondition_raw_response = f"|start|{response}|end|"
        return response


class _FakeExperienceLib:
    def retrieve_relevant_structured(self, **kwargs):
        return [{"text": "Stored death warning: preparation may be required."}]


def _hypothesis(preparable=True, requires=None, advice="prepare before entry"):
    return json.dumps({
        "preparable": preparable,
        "requires": list(requires or []),
        "reason": "The death text names a missing preparation.",
        "advice": advice if preparable else "",
    })


class PreconditionGoalTests(unittest.TestCase):
    def _agent(self, responses=None):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.situation_memory = SituationMemory()
        agent.llm = _FakeLLM(responses)
        agent.experience_lib = _FakeExperienceLib()
        agent.kg_map = KGMap()
        agent.kg_map._ensure_node("Back Room")
        agent.kg_map._ensure_node("Hazard Room")
        agent.kg_map.room_fingerprints["Back Room"] = "back room description"
        agent.kg_map.room_fingerprints["Hazard Room"] = "hazard description"
        agent.current_epoch = 2
        agent.step_count = 12
        agent._goal_visit_inventory = {}
        agent._recent_outcomes = [{
            "epoch": 2,
            "step": 11,
            "location": "Back Room",
            "destination": "Hazard Room",
            "command": "west",
            "observation": "Hazard Room",
            "outcome_class": "moved",
            "reward_change": 0,
        }]
        return agent

    def _trigger(self, agent, duplicate=True, inventory=None):
        return agent._maybe_handle_repeated_death_goal(
            duplicate_death=duplicate,
            event_key="death:v1:hazard:fp:west",
            hazard_location="Hazard Room",
            hazard_fingerprint=agent._location_fingerprint_hash("Hazard Room"),
            fatal_action="west",
            death_observation="You need protection before entering. You have died.",
            inventory_at_death=list(inventory or []),
        )

    def test_first_death_does_not_call_second_death_creates_goal_and_gateway(self):
        agent = self._agent([_hypothesis(True, ["protective item"])])
        transition, log = self._trigger(agent, duplicate=False)
        self.assertEqual((transition, log), ({}, {}))
        self.assertEqual(len(agent.llm.calls), 0)

        transition, log = self._trigger(agent, duplicate=True)
        self.assertEqual(len(agent.llm.calls), 1)
        self.assertEqual(transition["status"], "created")
        goal = agent.situation_memory.goal_situations()[0]
        self.assertEqual(goal["requires"], ["protective item"])
        self.assertEqual(goal["deaths"], 2)
        self.assertEqual(goal["gateways"][0]["room"], "Back Room")
        self.assertEqual(goal["gateways"][0]["command"], "west")
        self.assertEqual(log["state_type"], "precondition_hypothesis")

    def test_non_preparable_is_declined_and_not_reasked(self):
        agent = self._agent([_hypothesis(False)])
        transition, _ = self._trigger(agent)
        self.assertEqual(transition["status"], "declined")
        self.assertTrue(agent.situation_memory.goal_situations()[0]["declined"])
        self._trigger(agent)
        self.assertEqual(len(agent.llm.calls), 1)

    def test_reset_preserves_goals_but_full_reset_clears_them(self):
        memory = SituationMemory()
        memory.add({
            "location": "Room",
            "situation": "locked route",
            "possible_solution": "key",
        })
        memory.add_goal_situation(
            "Hazard", fatal_action="north", requires=["shield"]
        )
        memory.reset()
        self.assertEqual(len(memory.goal_situations()), 1)
        self.assertEqual(len(memory.active_situations()), 1)
        memory.reset(full=True)
        self.assertEqual(memory.goal_situations(), [])
        self.assertEqual(memory.active_situations(), [])

    def test_same_hazard_merges_fatal_actions_and_gateways(self):
        memory = SituationMemory()
        memory.add_goal_situation(
            "Hazard #2", "fp", "east",
            {"room": "East Hall", "fingerprint": "e", "command": "east"},
            deaths=1,
        )
        created, goal, status = memory.add_goal_situation(
            "Hazard #7", "fp", "west",
            {"room": "West Hall", "fingerprint": "w", "command": "west"},
            deaths=2,
        )
        self.assertFalse(created)
        self.assertEqual(status, "merged")
        self.assertEqual(len(memory.goal_situations()), 1)
        self.assertEqual(goal["fatal_actions"], ["east", "west"])
        self.assertEqual(len(goal["gateways"]), 2)
        self.assertEqual(goal["deaths"], 2)

    def test_observation_suppressed_while_goal_open_then_allowed(self):
        memory = SituationMemory()
        _, goal, _ = memory.add_goal_situation(
            "Hazard Room", "fp", "west", requires=["tool"]
        )
        stored, _ = memory.add({
            "location": "Hazard Room",
            "situation": "dangerous room",
            "possible_solution": "tool",
        })
        self.assertFalse(stored)
        self.assertEqual(memory.last_add_status, "suppressed_by_goal")
        memory.confirm_goal(goal["goal_id"], ["tool"])
        stored, _ = memory.add({
            "location": "Hazard Room",
            "situation": "dangerous room",
            "possible_solution": "tool",
        })
        self.assertTrue(stored)

    def test_feed_open_confirmed_and_avoid_lifecycle(self):
        memory = SituationMemory()
        _, goal, _ = memory.add_goal_situation(
            "Hazard", fatal_action="north", requires=["shield"], deaths=2
        )
        self.assertIn("PREPARATION REQUIRED", memory.active_situations()[0]["situation"])
        memory.confirm_goal(goal["goal_id"], ["shield"])
        self.assertEqual(memory.active_situations(), [])

        _, second, _ = memory.add_goal_situation(
            "Other Hazard", fatal_action="east", requires=["rope"], deaths=2
        )
        for step in range(3):
            memory.refute_goal(second["goal_id"], [f"rope {step}"], step=step)
        active = memory.active_situations()
        self.assertIn("AVOID", active[0]["situation"])

    def test_requirement_in_new_inventory_refutes_and_reasks(self):
        agent = self._agent([_hypothesis(True, ["gun"], "try another defense")])
        _, goal, _ = agent.situation_memory.add_goal_situation(
            "Hazard Room",
            agent._location_fingerprint_hash("Hazard Room"),
            "west",
            requires=["gun"],
            last_death_inventory=[],
            deaths=2,
        )
        transition, _ = self._trigger(agent, inventory=["gun"])
        self.assertEqual(len(agent.llm.calls), 1)
        self.assertEqual(transition["refutation"]["status"], "refuted")
        self.assertEqual(
            agent.llm.calls[0]["previous_hypothesis"]["goal_id"],
            goal["goal_id"],
        )
        self.assertEqual(len(agent.llm.calls[0]["previous_refutations"]), 1)

    def test_leaving_goal_room_alive_confirms_entry_inventory(self):
        agent = self._agent()
        _, goal, _ = agent.situation_memory.add_goal_situation(
            "Hazard Room",
            agent._location_fingerprint_hash("Hazard Room"),
            "west",
            requires=["shield"],
        )
        agent._update_goal_visit_lifecycle(
            "Back Room", "Hazard Room", False, ["shield", "lamp"]
        )
        transitions = agent._update_goal_visit_lifecycle(
            "Hazard Room", "Back Room", False, ["shield", "lamp"]
        )
        self.assertEqual(transitions[0]["status"], "confirmed")
        stored = agent.situation_memory.goal_situations()[0]
        self.assertEqual(stored["status"], "confirmed")
        self.assertEqual(stored["confirmed_inventory"], ["shield", "lamp"])
        self.assertEqual(stored["goal_id"], goal["goal_id"])

    def test_sixth_open_goal_is_refused(self):
        memory = SituationMemory()
        for index in range(5):
            created, _, status = memory.add_goal_situation(
                f"Hazard {index}", fatal_action="north"
            )
            self.assertTrue(created)
            self.assertEqual(status, "created")
        created, goal, status = memory.add_goal_situation(
            "Hazard 6", fatal_action="south"
        )
        self.assertFalse(created)
        self.assertIsNone(goal)
        self.assertEqual(status, "refused_cap")


if __name__ == "__main__":
    unittest.main()
