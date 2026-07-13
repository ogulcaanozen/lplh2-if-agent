"""Focused tests for text-centered room identity and KG bookkeeping."""

import json
import unittest

from . import config
from .agent import LPLHAgent
from .kg_map import KGMap, canonical_room_display


class _FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0
        self.last_location_resolver_prompt = ""
        self.last_location_resolver_raw_response = ""

    def resolve_location_identity(self, **kwargs):
        self.calls += 1
        self.last_location_resolver_prompt = json.dumps(kwargs)
        if not self.responses:
            raise AssertionError("unexpected resolver call")
        self.last_location_resolver_raw_response = self.responses.pop(0)
        return self.last_location_resolver_raw_response


def _resolver_response(decision, label="", confidence="high"):
    return json.dumps({
        "decision": decision,
        "match_label": label,
        "confidence": confidence,
        "reason": "test",
    })


class LLMTextLocationTests(unittest.TestCase):
    def _agent(self, responses=None):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap(strict_location_authority=True)
        agent.llm = _FakeLLM(responses)
        agent.current_epoch = 1
        agent._location_resolver_cache = {}
        agent._contradiction_splits_this_epoch = set()
        return agent

    def _seed(self, agent, title, description):
        label, _ = agent.kg_map.mint_room(title, description, epoch=1)
        agent.kg_map.confirm_arrival(label, description)
        return label

    def _gate(self, moved, title=""):
        return {
            "status": "routed",
            "decision": {
                "location_verdict": {"moved": moved, "room_title": title}
            },
            "action_transition_candidate": {},
        }

    def test_gate_no_move_keeps_room_and_fm_location_is_inert(self):
        agent = self._agent()
        start = self._seed(agent, "Outside", "Outside\nYou are behind a house.")
        result = agent._resolve_step_location(
            self._gate("no", "Kitchen"), "look through window",
            "Kitchen is visible through the window.",
            "Outside\nYou are behind a house.", start, False,
        )
        agent.kg_map.update([("You", "in", "kitchen")], "look through window")
        self.assertEqual(agent.kg_map.current_location, start)
        self.assertEqual(agent.kg_map.visited_rooms, [start])
        self.assertNotIn("kitchen", {key.lower() for key in agent.kg_map.nodes})
        self.assertEqual(result["action_transition_status"], "gate_no_move")

    def test_ungrounded_gate_title_uses_probe_fallback(self):
        agent = self._agent()
        start = self._seed(agent, "Outside", "Outside\nA street.")
        gate = self._gate("yes", "Invented Room")
        result = agent._resolve_step_location(
            gate, "north",
            "You walk onward.", "<< Hallway >>\nA narrow hall.", start, False,
        )
        self.assertEqual(agent.kg_map.current_location, "Hallway")
        self.assertFalse(result["verdict_validated"])
        self.assertEqual(result["resolution_mode"], "text_fallback")
        self.assertEqual(gate["action_transition_candidate"], {})

    def test_probe_finds_title_after_rejection_prefix(self):
        agent = self._agent()
        start = self._seed(agent, "Outside", "Outside\nA street.")
        result = agent._resolve_step_location(
            self._gate("unclear", ""), "south",
            "You can't go south from here!", "<< Hallway >>\nA hall.",
            start, False,
        )
        self.assertEqual(result["chosen_location_hint"], "Hallway")

    def test_resolver_selects_existing_candidate(self):
        agent = self._agent([_resolver_response("existing", "Hallway")])
        self._seed(agent, "Hallway", "Hallway\nA red corridor.")
        label, log = agent._resolve_arrival_identity(
            "Hallway", "Hallway\nA red corridor.", "north", "Lobby"
        )
        self.assertEqual(label, "Hallway")
        self.assertEqual(log["resolver_decision"], "existing")

    def test_new_low_and_invalid_candidate_create_stable_sibling(self):
        agent = self._agent([
            _resolver_response("existing", "Not Offered"),
            _resolver_response("new", "", "low"),
        ])
        self._seed(agent, "Hallway", "Hallway\nA red corridor.")
        label, log = agent._resolve_arrival_identity(
            "Hallway", "Hallway\nA blue corridor.", "east", "Lobby"
        )
        self.assertEqual(label, "Hallway #2")
        self.assertEqual(log["resolver_decision"], "new")

    def test_resolver_cache_avoids_second_call(self):
        agent = self._agent([_resolver_response("existing", "Hallway")])
        self._seed(agent, "Hallway", "Hallway\nA red corridor.")
        first = agent._resolve_arrival_identity(
            "Hallway", "Hallway\nA red corridor.", "north", "Lobby"
        )[0]
        second = agent._resolve_arrival_identity(
            "Hallway", "Hallway\nA red corridor.", "north", "Lobby"
        )[0]
        self.assertEqual(first, second)
        self.assertEqual(agent.llm.calls, 1)

    def test_registry_survives_epoch_reset_and_stabilizes_event_key(self):
        agent = self._agent()
        label1 = self._seed(agent, "Clearing", "Clearing\nA circular clearing.")
        rid1 = agent.kg_map.registry_id_for(label1)
        key1 = agent._score_event_key("gain", "take egg", label1, 5)
        agent.kg_map.reset(full=False)
        label2, rid2 = agent.kg_map.mint_room(
            "Clearing", "Clearing\nA circular clearing.", epoch=2
        )
        agent.kg_map.confirm_arrival(label2, "Clearing\nA circular clearing.")
        key2 = agent._score_event_key("gain", "take egg", label2, 5)
        self.assertEqual((label1, rid1, key1), (label2, rid2, key2))

    def test_edge_and_blocked_contradictions_split_source(self):
        kg = KGMap(strict_location_authority=True)
        source, _ = kg.mint_room("Hallway", "Hallway\nA junction.", epoch=1)
        x, _ = kg.mint_room("Room X", "Room X\nX.", epoch=1)
        y, _ = kg.mint_room("Room Y", "Room Y\nY.", epoch=1)
        kg.confirm_arrival(source, "Hallway\nA junction.")
        visit = kg._current_visit_id
        kg.confirm_direction(source, "north", x)
        event = kg.confirm_direction(
            source, "north", y, epoch=1, step=8,
            source_visit_id=visit, allow_split=True,
        )
        self.assertEqual(kg.nodes[source]["direction"]["north"], x)
        self.assertEqual(kg.nodes[event["new_sibling"]]["direction"]["north"], y)

        blocked, _ = kg.mint_room("Passage", "Passage\nStone.", epoch=1)
        kg.confirm_arrival(blocked, "Passage\nStone.")
        visit = kg._current_visit_id
        kg.mark_direction_tried_at("west", blocked)
        event2 = kg.confirm_direction(
            blocked, "west", y, epoch=1, step=9,
            source_visit_id=visit, allow_split=True,
        )
        self.assertNotIn("west", kg.nodes[blocked]["blocked_directions"])
        self.assertEqual(kg.nodes[event2["new_sibling"]]["direction"]["west"], y)

    def test_triple_hygiene_targets_current_sibling_only(self):
        kg = KGMap(strict_location_authority=True)
        first, _ = kg.mint_room("Hallway", "Hallway\nRed.", epoch=1)
        second, _ = kg.mint_room(
            "Hallway", "Hallway\nBlue.", epoch=1, force_new=True
        )
        kg.confirm_arrival(second, "Hallway\nBlue.")
        kg.nodes[first]["may_direction"].remove("north")
        kg.nodes[second]["may_direction"].remove("north")
        kg.update([("Hallway", "north", "restraunt")], "look")
        self.assertNotIn("north", kg.nodes[first]["may_direction"])
        self.assertIn("north", kg.nodes[second]["may_direction"])
        self.assertFalse(kg.nodes[second]["direction"])
        self.assertNotIn("restraunt", {key.lower() for key in kg.nodes})

    def test_death_room_is_grounded_without_moving_cursor(self):
        agent = self._agent()
        outside = self._seed(agent, "Outside", "Outside\nA street.")
        sauna, _ = agent.kg_map.mint_room("Sauna", "Sauna\nHot room.", epoch=1)
        summary = json.dumps({"death_room_title": "<< Sauna >>"})
        grounded, detail = agent._ground_death_room_from_summary(
            summary, "You enter.\n<< Sauna >>\nThe heat kills you."
        )
        context = agent._goal_hazard_context(
            "east", True, outside, outside, grounded_death_room=grounded
        )
        self.assertEqual(detail["status"], "grounded")
        self.assertEqual(context["hazard_location"], sauna)
        self.assertEqual(agent.kg_map.current_location, outside)

    def test_death_warning_keeps_issuing_room_retrieval_identity(self):
        agent = self._agent()
        outside = self._seed(agent, "Outside", "Outside\nA street.")
        outside_id = agent.kg_map.registry_id_for(outside)
        pool, pool_id = agent.kg_map.mint_room(
            "Pool", "Pool\nDeep water.", epoch=1
        )
        metadata = agent._attach_death_hazard_metadata(
            {
                "kind": "death_warning",
                "location": outside,
                "location_issued": outside,
                "location_fingerprint": agent._location_fingerprint_hash(outside),
                "location_registry_id": outside_id,
            },
            goal_hazard={
                "hazard_location": pool,
                "hazard_fingerprint": pool_id,
                "hazard_registry_id": pool_id,
            },
            grounded_death_room=pool,
            death_room_grounding={"status": "grounded"},
        )
        self.assertEqual(metadata["location"], outside)
        self.assertEqual(metadata["location_registry_id"], outside_id)
        self.assertEqual(metadata["location_after"], pool)
        self.assertEqual(metadata["hazard_registry_id"], pool_id)
        self.assertTrue(agent._warning_experience_relevant({"metadata": metadata}))

    def test_uncertain_source_cannot_ground_navigation_writes(self):
        self.assertFalse(LPLHAgent._source_navigation_writes_allowed(True))
        self.assertTrue(LPLHAgent._source_navigation_writes_allowed(False))

    def test_decoration_normalization_has_one_display(self):
        self.assertEqual(canonical_room_display("<< Outside >>"), "Outside")
        self.assertEqual(canonical_room_display("-- Outside --"), "Outside")
        self.assertEqual(canonical_room_display("Outside"), "Outside")
        self.assertEqual(canonical_room_display("<< Outside >> #2"), "Outside #2")

    def test_flags_off_leave_legacy_pipeline_available(self):
        old_gate = config.AUX_GATE_LOCATION_VERDICT
        old_resolver = config.LLM_LOCATION_RESOLVER
        try:
            config.AUX_GATE_LOCATION_VERDICT = False
            config.LLM_LOCATION_RESOLVER = False
            agent = self._agent()
            outside = self._seed(agent, "Outside", "Outside\nA street.")
            result = agent._resolve_step_location(
                self._gate("yes", "Kitchen"), "enter window",
                "Kitchen\nA room.", "", outside, False,
            )
            transition = agent._apply_gate_action_transition(
                {
                    "decision": {
                        "kg_action_transition": {
                            "record": True,
                            "reason": "test",
                        }
                    },
                    "action_transition_candidate": {
                        "from": outside,
                        "command": "enter window",
                        "to": "Kitchen",
                    },
                },
                use_legacy_location_pipeline=True,
            )
            self.assertEqual(result["resolution_mode"], "text_fallback")
            self.assertEqual(agent.kg_map.current_location, "Kitchen")
            self.assertTrue(transition["applied"])

            agent.kg_map._legacy_update([
                ("[Location]", "have", "table"),
                ("Kitchen", "need", "key"),
            ], "look")
            self.assertIn("table", agent.kg_map.nodes["Kitchen"]["have"])
            self.assertIn("key", agent.kg_map.nodes["Kitchen"]["needs"])
        finally:
            config.AUX_GATE_LOCATION_VERDICT = old_gate
            config.LLM_LOCATION_RESOLVER = old_resolver


if __name__ == "__main__":
    unittest.main()
