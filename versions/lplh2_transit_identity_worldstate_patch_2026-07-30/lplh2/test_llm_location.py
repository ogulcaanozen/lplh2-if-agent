"""Focused tests for text-centered room identity and KG bookkeeping."""

import json
import unittest

from . import config
from .agent import LPLHAgent
from .kg_map import KGMap, canonical_room_display
from .prompts import AFFORDANCE_BRAINSTORMING_PROMPT, LPLH_ACTION_GENERATION_PROMPT


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
        agent.step_count = 0
        agent._location_resolver_cache = {}
        agent._contradiction_splits_this_epoch = set()
        agent._same_title_chain_mints_this_epoch = {}
        return agent

    def _seed(self, agent, title, description):
        label, _ = agent.kg_map.mint_room(title, description, epoch=1)
        agent.kg_map.confirm_arrival(label, description)
        return label

    def _gate(self, moved, title=""):
        return {
            "status": "routed",
            "decision": {
                "location_verdict": {"moved": moved, "room_title": title},
                "command_outcome": {"status": "accepted"},
            },
            "action_transition_candidate": {},
        }

    def test_direction_signature_change_overrides_gate_no_move(self):
        agent = self._agent([
            _resolver_response("existing", "Hallway"),
        ])
        start = self._seed(
            agent, "Hallway", "Hallway\nA red-carpeted corridor."
        )
        result = agent._resolve_step_location(
            self._gate("no", "Hallway"),
            "north",
            "Hallway\nA tiled corridor bends west.",
            "",
            start,
            False,
        )
        self.assertEqual(result["movement_override"]["trigger"], "signature_change")
        self.assertEqual(result["gate_location_verdict"]["moved"], "yes")
        self.assertEqual(agent.kg_map.current_location, start)
        self.assertEqual(
            result["same_title_chain_skipped"],
            "movement_override",
        )
        self.assertEqual(len(agent.kg_map.room_candidates("Hallway")), 1)

    def test_score_gain_override_still_allows_same_title_chain_split(self):
        agent = self._agent([
            _resolver_response("existing", "Hallway"),
        ])
        start = self._seed(
            agent, "Hallway", "Hallway\nA red-carpeted corridor."
        )
        result = agent._resolve_step_location(
            self._gate("no", "Hallway"),
            "west",
            "Hallway\nA tiled corridor bends south.",
            "",
            start,
            False,
            reward_change=10,
        )
        self.assertEqual(
            result["movement_override"]["trigger"],
            "score_gain",
        )
        self.assertEqual(
            result["resolver_decision"],
            "new_same_title_chain",
        )
        self.assertEqual(agent.kg_map.current_location, "Hallway #2")
        self.assertNotIn("same_title_chain_skipped", result)
        self.assertEqual(len(agent.kg_map.room_candidates("Hallway")), 2)

    def test_titleless_signature_change_does_not_override_gate_no_move(self):
        agent = self._agent()
        start = self._seed(
            agent, "Hallway", "Hallway\nA red-carpeted corridor."
        )
        result = agent._resolve_step_location(
            self._gate("no", ""),
            "north",
            "A tiled corridor bends west.",
            "",
            start,
            False,
        )
        self.assertFalse(result["movement_override"])
        self.assertEqual(result["gate_location_verdict"]["moved"], "no")
        self.assertEqual(agent.kg_map.current_location, start)
        self.assertFalse(agent.kg_map.location_uncertain)

    def test_titleless_score_gain_still_marks_destination_uncertain(self):
        agent = self._agent()
        start = self._seed(
            agent, "Hallway", "Hallway\nA red-carpeted corridor."
        )
        result = agent._resolve_step_location(
            self._gate("no", ""),
            "north",
            "You gain ten points. It is too dark to see.",
            "",
            start,
            False,
            reward_change=10,
        )
        self.assertEqual(
            result["movement_override"]["trigger"],
            "score_gain",
        )
        self.assertEqual(
            result["action_transition_status"],
            "moved_destination_unseen",
        )
        self.assertTrue(agent.kg_map.location_uncertain)
        self.assertEqual(agent.kg_map.current_location, start)

    def test_titleless_unclear_nonmovement_keeps_current_room(self):
        agent = self._agent()
        attic = self._seed(
            agent,
            "Attic",
            "Attic\nA narrow attic with a table.",
        )
        result = agent._resolve_step_location(
            self._gate("unclear", ""),
            "take egg",
            "Taken.",
            "Attic\nA narrow attic with a table.",
            attic,
            False,
        )
        self.assertEqual(agent.kg_map.current_location, attic)
        self.assertEqual(
            result["action_transition_status"],
            "titleless_unclear_nonmovement_retained",
        )
        self.assertFalse(result["resolver_invoked"])
        self.assertEqual(agent.llm.calls, 0)

    def test_observation_title_candidate_cannot_mint_phantom_after_take(self):
        agent = self._agent()
        self._seed(
            agent,
            "Attic",
            "Attic\nA narrow attic containing an egg.",
        )
        sibling, _ = agent.kg_map.mint_room(
            "Attic",
            "Attic\nA narrow attic after the egg was removed.",
            epoch=1,
        )
        agent.kg_map.confirm_arrival(
            sibling,
            "Attic\nA narrow attic after the egg was removed.",
        )
        gate = self._gate("unclear", "")
        gate["action_valid"] = True
        gate["action_transition_candidate"] = {
            "from": sibling,
            "command": "take egg",
            "to": "Attic",
            "source": "observation_room_title",
        }

        result = agent._resolve_step_location(
            gate,
            "take egg",
            "Taken.",
            "Attic\nA narrow attic. The egg is gone.",
            sibling,
            False,
        )

        self.assertEqual(agent.kg_map.current_location, sibling)
        self.assertEqual(
            result["action_transition_status"],
            "titleless_unclear_nonmovement_retained",
        )
        self.assertEqual(
            len(agent.kg_map.room_candidates("Attic")),
            2,
        )
        self.assertEqual(agent.llm.calls, 0)

    def test_titleless_unclear_noncardinal_movement_is_preserved(self):
        agent = self._agent()
        behind = self._seed(
            agent,
            "Behind House",
            "Behind House\nAn open window leads inside.",
        )
        gate = self._gate("unclear", "")
        gate["action_valid"] = True
        gate["action_transition_candidate"] = {
            "from": behind,
            "command": "enter window",
            "to": "Kitchen",
            "source": "observation_room_title",
        }

        result = agent._resolve_step_location(
            gate,
            "enter window",
            "Kitchen\nA table stands in the room.",
            "",
            behind,
            False,
        )

        self.assertEqual(agent.kg_map.current_location, "Kitchen")
        self.assertNotEqual(
            result["action_transition_status"],
            "titleless_unclear_nonmovement_retained",
        )
        self.assertEqual(agent.llm.calls, 0)

    def test_rejected_or_no_effect_movement_shape_cannot_bypass_guard(self):
        for outcome_status in ("rejected", "no_effect"):
            with self.subTest(outcome_status=outcome_status):
                agent = self._agent()
                self._seed(
                    agent,
                    "Attic",
                    "Attic\nA narrow attic containing an egg.",
                )
                sibling, _ = agent.kg_map.mint_room(
                    "Attic",
                    "Attic\nA narrow attic after the egg was removed.",
                    epoch=1,
                )
                agent.kg_map.confirm_arrival(
                    sibling,
                    "Attic\nA narrow attic after the egg was removed.",
                )
                gate = self._gate("unclear", "")
                gate["action_valid"] = True
                gate["decision"]["command_outcome"]["status"] = outcome_status
                gate["action_transition_candidate"] = {
                    "from": sibling,
                    "command": "climb tree",
                    "to": "Attic",
                    "source": "observation_room_title",
                }

                result = agent._resolve_step_location(
                    gate,
                    "climb tree",
                    "You cannot climb any higher.",
                    "Attic\nA narrow attic. The egg is gone.",
                    sibling,
                    False,
                )

                self.assertEqual(agent.kg_map.current_location, sibling)
                self.assertEqual(
                    result["action_transition_status"],
                    "titleless_unclear_nonmovement_retained",
                )
                self.assertEqual(
                    len(agent.kg_map.room_candidates("Attic")),
                    2,
                )
                self.assertEqual(agent.llm.calls, 0)

    def test_movement_override_respects_rejection_and_command_shape(self):
        agent = self._agent()
        start = self._seed(agent, "Hallway", "Hallway\nA red corridor.")
        rejected = self._gate("no", "Hallway")
        rejected["decision"]["command_outcome"]["status"] = "rejected"
        rejected_result = agent._resolve_step_location(
            rejected,
            "north",
            "Hallway\nA blue corridor.",
            "",
            start,
            False,
            reward_change=10,
        )
        self.assertFalse(rejected_result["movement_override"])
        self.assertEqual(agent.kg_map.current_location, start)

        non_movement_result = agent._resolve_step_location(
            self._gate("no", "Hallway"),
            "look",
            "Hallway\nA blue corridor.",
            "",
            start,
            False,
            reward_change=10,
        )
        self.assertFalse(non_movement_result["movement_override"])
        self.assertEqual(agent.kg_map.current_location, start)

    def test_same_title_chain_alias_and_mint_cap(self):
        agent = self._agent([
            _resolver_response("existing", "Hallway"),
        ])
        first_text = "Hallway\nA red corridor."
        first = self._seed(agent, "Hallway", first_text)
        alias, alias_log = agent._resolve_arrival_identity(
            "Hallway", first_text, "north", first
        )
        self.assertEqual(alias, first)
        self.assertEqual(alias_log["resolver_decision"], "signature_match")

        sibling, sibling_log = agent._resolve_arrival_identity(
            "Hallway", "Hallway\nA blue corridor.", "north", first
        )
        self.assertEqual(sibling, "Hallway #2")
        self.assertEqual(
            sibling_log["resolver_decision"], "new_same_title_chain"
        )
        self.assertEqual(
            sibling_log["resolver_original_decision"], "existing"
        )
        self.assertEqual(agent.llm.calls, 1)

        old_cap = config.SAME_TITLE_CHAIN_MAX_MINTS_PER_EPOCH
        try:
            config.SAME_TITLE_CHAIN_MAX_MINTS_PER_EPOCH = 1
            agent.llm.responses.append(
                _resolver_response("existing", "Hallway")
            )
            _, cap_log = agent._resolve_arrival_identity(
                "Hallway", "Hallway\nA green corridor.", "south", first
            )
            self.assertIn("same_title_chain_mint_cap_hit", cap_log)
        finally:
            config.SAME_TITLE_CHAIN_MAX_MINTS_PER_EPOCH = old_cap

    def test_same_title_portable_object_drift_uses_resolver_alias(self):
        agent = self._agent([
            _resolver_response("existing", "Forest"),
        ])
        forest = self._seed(
            agent,
            "Forest",
            "Forest\nTall trees surround a narrow path.",
        )
        path, _ = agent.kg_map.mint_room(
            "Path",
            "Path\nA path through the trees.",
            epoch=1,
        )
        label, log = agent._resolve_arrival_identity(
            "Forest",
            "Forest\nTall trees surround a narrow path. A coin lies here.",
            "north",
            path,
        )
        self.assertEqual(label, forest)
        self.assertEqual(log["resolver_decision"], "existing")
        self.assertTrue(log["signature_alias_added"])
        self.assertEqual(len(agent.kg_map.room_candidates("Forest")), 1)

    def test_confirmed_same_title_self_loop_does_not_mint_sibling(self):
        agent = self._agent([
            _resolver_response("existing", "Hallway"),
        ])
        hallway = self._seed(
            agent,
            "Hallway",
            "Hallway\nA long corridor.",
        )
        agent.kg_map.confirm_direction(hallway, "north", hallway)
        label, log = agent._resolve_arrival_identity(
            "Hallway",
            "Hallway\nA long corridor with a discarded note.",
            "north",
            hallway,
        )
        self.assertEqual(label, hallway)
        self.assertEqual(log["resolver_decision"], "existing")
        self.assertNotIn("same_title_chain_mints_this_epoch", log)
        self.assertEqual(len(agent.kg_map.room_candidates("Hallway")), 1)

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
            "Hallway", "Hallway\nA red corridor with a lamp.", "north", "Lobby"
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
            "Hallway", "Hallway\nA red corridor with a lamp.", "north", "Lobby"
        )[0]
        second = agent._resolve_arrival_identity(
            "Hallway", "Hallway\nA red corridor with a lamp.", "north", "Lobby"
        )[0]
        self.assertEqual(first, second)
        self.assertEqual(agent.llm.calls, 1)

    def test_resolver_adds_signature_alias_then_variant_fast_paths(self):
        base = "Gallery\nA stone gallery contains a portable coin."
        variant = base + " A distant bell sounds."
        agent = self._agent([_resolver_response("existing", "Gallery")])
        label = self._seed(agent, "Gallery", base)
        registry_id = agent.kg_map.registry_id_for(label)

        first, log = agent._resolve_arrival_identity(
            "Gallery", variant, "north", "Landing"
        )
        self.assertEqual(first, label)
        self.assertTrue(log["signature_alias_added"])
        self.assertEqual(
            len(agent.kg_map.entry_signatures(
                agent.kg_map.room_registry[registry_id]
            )),
            2,
        )

        agent._location_resolver_cache.clear()
        second, second_log = agent._resolve_arrival_identity(
            "Gallery", variant, "east", "Landing"
        )
        self.assertEqual(second, label)
        self.assertEqual(second_log["resolver_decision"], "signature_match")
        self.assertEqual(agent.llm.calls, 1)

    def test_low_confidence_structural_variant_does_not_become_alias(self):
        base = "Passage\nA stone corridor ends at a north door."
        variant = "Passage\nA wooden bridge crosses a ravine to the east."
        agent = self._agent([_resolver_response("existing", "Passage", "low")])
        label = self._seed(agent, "Passage", base)
        registry_id = agent.kg_map.registry_id_for(label)
        sibling, log = agent._resolve_arrival_identity(
            "Passage", variant, "east", "Landing"
        )
        self.assertEqual(sibling, "Passage #2")
        self.assertEqual(log["resolver_decision"], "new")
        self.assertFalse(log["signature_alias_added"])
        self.assertEqual(
            len(agent.kg_map.entry_signatures(
                agent.kg_map.room_registry[registry_id]
            )),
            1,
        )

    def test_epoch_fresh_variant_uses_registry_candidate_resolver(self):
        base = "Gallery\nA stone gallery contains a portable coin."
        variant = "Gallery\nThe same stone gallery is now empty."
        agent = self._agent([_resolver_response("existing", "Gallery")])
        label = self._seed(agent, "Gallery", base)
        registry_id = agent.kg_map.registry_id_for(label)
        agent.kg_map.reset(full=False)
        agent.current_epoch = 2

        resolved, log = agent._resolve_arrival_identity(
            "Gallery", variant, "south", "Landing"
        )
        self.assertEqual(resolved, label)
        self.assertEqual(log["candidate_source"], "registry")
        self.assertTrue(log["resolver_invoked"])
        self.assertEqual(agent.kg_map.registry_id_for(resolved), registry_id)
        self.assertEqual(len(agent.kg_map.room_registry), 1)

    def test_signature_fast_path_ignores_new_arrival_direction(self):
        agent = self._agent()
        self._seed(agent, "Clearing", "Clearing\nA grating lies under leaves.")
        small, small_id = agent.kg_map.mint_room(
            "Clearing", "Clearing\nA small circular clearing.", epoch=1
        )
        self.assertEqual(small, "Clearing #2")
        label, log = agent._resolve_arrival_identity(
            "Clearing", "Clearing\nA small circular clearing.",
            "northwest", "Forest",
        )
        self.assertEqual(label, small)
        self.assertEqual(agent.kg_map.registry_id_for(label), small_id)
        self.assertEqual(log["resolver_decision"], "signature_match")
        self.assertFalse(log["resolver_invoked"])
        self.assertEqual(agent.llm.calls, 0)

    def test_signature_fast_path_restores_registry_room_across_epochs(self):
        agent = self._agent()
        grating = "Clearing\nA grating lies under leaves."
        small = "Clearing\nA small circular clearing."
        self._seed(agent, "Clearing", grating)
        small_label, small_id = agent.kg_map.mint_room(
            "Clearing", small, epoch=1
        )
        agent.kg_map.reset(full=False)
        agent.current_epoch = 2
        agent.kg_map.mint_room("Clearing", grating, epoch=2)
        label, log = agent._resolve_arrival_identity(
            "Clearing", small, "east", "Forest"
        )
        self.assertEqual(label, small_label)
        self.assertEqual(agent.kg_map.registry_id_for(label), small_id)
        self.assertEqual(log["resolver_decision"], "signature_match")
        self.assertEqual(agent.llm.calls, 0)

    def test_ambiguous_signature_prefers_known_edge_then_recency(self):
        agent = self._agent()
        text = "Hallway\nAn undecorated corridor."
        first, _ = agent.kg_map.mint_room("Hallway", text, epoch=1)
        second, _ = agent.kg_map.mint_room(
            "Hallway", text, epoch=1, force_new=True
        )
        source, _ = agent.kg_map.mint_room(
            "Lobby", "Lobby\nA central lobby.", epoch=1
        )
        agent.kg_map.confirm_arrival(source, "Lobby\nA central lobby.")
        agent.kg_map.confirm_direction(source, "north", second)
        label, log = agent._resolve_arrival_identity(
            "Hallway", text, "north", source
        )
        self.assertEqual(label, second)
        self.assertEqual(log["resolver_ambiguity_preference"], "arrival_edge")

        agent.kg_map.nodes[source]["direction"].clear()
        agent.kg_map.confirm_arrival(first, text)
        agent.kg_map.confirm_arrival(second, text)
        label2, log2 = agent._resolve_arrival_identity(
            "Hallway", text, "east", source
        )
        self.assertEqual(label2, second)
        self.assertEqual(log2["resolver_ambiguity_preference"], "most_recent_visit")
        self.assertEqual(agent.llm.calls, 0)

    def test_registry_dedup_and_splitter_force_new_contract(self):
        kg = KGMap(strict_location_authority=True)
        text = "Passage\nA narrow stone passage."
        first, first_id = kg.mint_room("Passage", text, epoch=1)
        reused, reused_id = kg.mint_room(
            "Passage", text, epoch=1, force_new=False
        )
        split, split_id = kg.mint_room(
            "Passage", text, epoch=1, force_new=True
        )
        self.assertEqual((reused, reused_id), (first, first_id))
        self.assertNotEqual(split, first)
        self.assertNotEqual(split_id, first_id)

    def test_arrival_description_is_immutable_and_split_uses_latest(self):
        kg = KGMap(strict_location_authority=True)
        first_text = "Forest\nSunlight filters through the trees."
        latest_text = "Forest\nLarge trees stand all around."
        source, _ = kg.mint_room("Forest", first_text, epoch=1)
        x, _ = kg.mint_room("Room X", "Room X\nX.", epoch=1)
        y, _ = kg.mint_room("Room Y", "Room Y\nY.", epoch=1)
        kg.confirm_arrival(source, first_text)
        kg.confirm_arrival(source, latest_text)
        self.assertEqual(kg.nodes[source]["arrival_description"], first_text)
        self.assertEqual(
            kg.nodes[source]["last_arrival_description"], latest_text
        )
        visit = kg._current_visit_id
        kg.confirm_direction(source, "north", x)
        event = kg.confirm_direction(
            source, "north", y, epoch=1, step=3,
            source_visit_id=visit, allow_split=True,
        )
        sibling = event["new_sibling"]
        self.assertEqual(kg.nodes[sibling]["arrival_description"], latest_text)

    def test_darkness_view_hides_frozen_room_and_clears_on_arrival(self):
        agent = self._agent()
        living = self._seed(
            agent, "Living Room", "Living Room\nA comfortable room."
        )
        agent.kg_map.nodes[living]["have"].append("lantern")
        agent.kg_map.nodes[living]["direction"]["east"] = "Kitchen"
        agent.kg_map.inventory = ["leaflet"]
        agent.kg_map.set_location_uncertain(True, entered_via="down")
        context = agent.kg_map.to_clean_dict()
        room_state = context["current_room_state"]
        self.assertEqual(room_state["inventory"], ["leaflet"])
        self.assertEqual(room_state["visible_objects"], [])
        self.assertEqual(room_state["confirmed_exits"], {})
        self.assertEqual(room_state["untried_exits"], [])
        self.assertEqual(room_state["last_known_location"], living)
        self.assertEqual(room_state["likely_way_back"], "up")
        self.assertNotIn("lantern", agent.kg_map.to_prompt_string().lower())
        self.assertEqual(agent._visible_objects_for_location(living), [])

        cellar, _ = agent.kg_map.mint_room(
            "Cellar", "Cellar\nA stone cellar.", epoch=1
        )
        agent.kg_map.confirm_arrival(cellar, "Cellar\nA stone cellar.")
        self.assertFalse(agent.kg_map.location_uncertain)
        self.assertNotIn("darkness", agent.kg_map.to_clean_dict()["current_room_state"])

    def test_dark_probe_with_hallucinated_title_keeps_uncertainty(self):
        agent = self._agent()
        living = self._seed(
            agent,
            "Living Room",
            "Living Room\nA comfortable room.",
        )
        agent.kg_map.set_location_uncertain(
            True,
            entered_via="down",
            step_index=4,
        )
        result = agent._resolve_step_location(
            self._gate("no", "Living Room"),
            "look",
            "It is pitch black. You are likely to be eaten.",
            "It is pitch black. You cannot see a thing.",
            living,
            False,
        )
        self.assertTrue(agent.kg_map.location_uncertain)
        self.assertEqual(result["uncertainty_event"], "uncertainty_retained")
        self.assertEqual(
            agent._effective_location(),
            "Unseen area via down from Living Room",
        )
        self.assertNotIn(
            "Unseen area via down from Living Room",
            agent.kg_map.nodes,
        )

    def test_verbatim_title_resolves_uncertainty_without_movement(self):
        agent = self._agent()
        living = self._seed(
            agent,
            "Living Room",
            "Living Room\nA comfortable room.",
        )
        agent.kg_map.set_location_uncertain(True, entered_via="down")
        result = agent._resolve_step_location(
            self._gate("no", "Cellar"),
            "turn on lantern",
            "Cellar\nA dark and damp cellar.",
            "",
            living,
            False,
        )
        self.assertFalse(agent.kg_map.location_uncertain)
        self.assertEqual(agent.kg_map.current_location, "Cellar")
        self.assertEqual(result["uncertainty_event"], "uncertainty_resolved")

    def test_verbatim_title_resolves_uncertainty_after_movement(self):
        agent = self._agent()
        living = self._seed(
            agent,
            "Living Room",
            "Living Room\nA comfortable room.",
        )
        agent.kg_map.set_location_uncertain(True, entered_via="down")
        agent._resolve_step_location(
            self._gate("yes", "Cellar"),
            "north",
            "Cellar\nA dark and damp cellar.",
            "",
            living,
            False,
        )
        self.assertFalse(agent.kg_map.location_uncertain)
        self.assertEqual(agent.kg_map.current_location, "Cellar")

    def test_death_title_after_marker_is_rejected(self):
        agent = self._agent()
        after_summary = json.dumps({"death_room_title": "Forest"})
        grounded, detail = agent._ground_death_room_from_summary(
            after_summary,
            "You enter the room. *** You have died *** Forest This is a forest.",
        )
        self.assertEqual(grounded, "")
        self.assertEqual(detail["status"], "after_death_marker")

        before_summary = json.dumps({"death_room_title": "<< Sauna >>"})
        grounded2, detail2 = agent._ground_death_room_from_summary(
            before_summary,
            "<< Sauna >>\nThe heat rises. *** You have died ***",
        )
        self.assertEqual(grounded2, "Sauna")
        self.assertEqual(detail2["status"], "grounded")

    def test_possession_prompts_make_inventory_authoritative(self):
        self.assertIn("Inventory Is Authoritative", LPLH_ACTION_GENERATION_PROMPT)
        self.assertIn("opening a container", LPLH_ACTION_GENERATION_PROMPT.lower())
        self.assertIn("Inventory is authoritative", AFFORDANCE_BRAINSTORMING_PROMPT)

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
        self.assertIn("west", kg.nodes[blocked]["blocked_directions"])
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
