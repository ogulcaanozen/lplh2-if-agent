"""Focused regression tests for engine-grounded KG room identity."""

import unittest

from .agent import LPLHAgent
from .game_runner import _engine_location_info, _look_probe
from .kg_map import KGMap, canonical_room_display


class EngineLocationTests(unittest.TestCase):
    def test_runner_ascends_enclosure_and_restores_probe_state(self):
        class Obj:
            def __init__(self, num, name, parent=0):
                self.num = num
                self.name = name
                self.parent = parent

        class Env:
            def __init__(self):
                self.objects = {
                    7: Obj(7, "boat", 20),
                    20: Obj(20, "river bank", 0),
                }
                self.state = "before"

            def get_player_location(self):
                return self.objects[7]

            def get_object(self, num):
                return self.objects.get(num)

            def get_state(self):
                return self.state

            def set_state(self, state):
                self.state = state

            def step(self, action):
                self.state = "after-look"
                return "River Bank\nYou are beside a river.", 0, False, {}

        env = Env()
        facts = _engine_location_info(env)
        self.assertEqual(facts["engine_location_num"], 20)
        self.assertEqual(facts["engine_enclosure_name"], "boat")
        self.assertTrue(_look_probe(env).startswith("River Bank"))
        self.assertEqual(env.state, "before")

    def test_identical_titles_get_stable_num_labels_across_reset(self):
        kg = KGMap()
        first = kg.resolve_arrival_location("<< Hallway >>", engine_num=41, epoch=1)
        second = kg.resolve_arrival_location("Hallway", engine_num=52, epoch=1)
        self.assertEqual((first, second), ("Hallway", "Hallway #2"))
        kg.reset()
        self.assertEqual(
            kg.resolve_arrival_location("Hallway", engine_num=52, epoch=2),
            "Hallway #2",
        )
        self.assertEqual(
            kg.resolve_arrival_location("Hallway", engine_num=41, epoch=2),
            "Hallway",
        )

    def test_full_reset_clears_registry(self):
        kg = KGMap()
        kg.resolve_arrival_location("Room", engine_num=8)
        kg.reset(full=True)
        self.assertEqual(kg.room_registry, {})
        self.assertEqual(kg.num_labels, {})

    def test_engine_triples_cannot_move_or_mint_rooms(self):
        kg = KGMap()
        room = kg.resolve_arrival_location("Outside", engine_num=10)
        kg.confirm_arrival(room, engine_num=10)
        kg.update([("You", "in", "kitchen")], "look through window", engine_grounded=True)
        self.assertEqual(kg.current_location, "Outside")
        self.assertNotIn("kitchen", kg.nodes)
        self.assertEqual(kg.visited_rooms, ["Outside"])

    def test_fm_direction_is_hint_on_current_sibling_only(self):
        kg = KGMap()
        first = kg.resolve_arrival_location("Outside", engine_num=10)
        second = kg.resolve_arrival_location("Outside", engine_num=11)
        kg.confirm_arrival(second, engine_num=11)
        kg.nodes[second]["may_direction"].remove("north")
        kg.update([("Outside", "north", "restraunt")], engine_grounded=True)
        self.assertNotIn("restraunt", kg.nodes)
        self.assertNotIn("north", kg.nodes[first]["direction"])
        self.assertNotIn("north", kg.nodes[second]["direction"])
        self.assertIn("north", kg.nodes[second]["may_direction"])

    def test_confirmed_engine_movement_writes_one_edge(self):
        kg = KGMap()
        source = kg.resolve_arrival_location("Outside", engine_num=10)
        destination = kg.resolve_arrival_location("Lobby", engine_num=20)
        kg.confirm_arrival(source, engine_num=10)
        kg.confirm_arrival(destination, engine_num=20)
        kg.confirm_direction(source, "north", destination)
        self.assertEqual(kg.nodes[source]["direction"], {"north": "Lobby"})
        self.assertNotIn("north", kg.nodes[source]["may_direction"])

    def test_death_mid_text_title_is_hazard_not_current_room(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        source = agent.kg_map.resolve_arrival_location("Outside", engine_num=10)
        agent.kg_map.confirm_arrival(source, engine_num=10)
        text = "You can't go south from here!\n\n<< Sauna >>\n*** You have died ***"
        title = agent._extract_observation_room_title(text, scan_all_lines=True)
        self.assertEqual(title, "Sauna")
        context = agent._goal_hazard_context(
            "south", False, source, source, hazard_destination=title
        )
        self.assertEqual(context["hazard_location"], "Sauna")
        self.assertEqual(agent.kg_map.current_location, "Outside")
        self.assertFalse(agent.kg_map.is_direction_blocked("Outside", "south"))

    def test_decoration_normalization(self):
        for raw in ("<< Outside >>", "-- Outside --", "*Outside*", "Outside"):
            self.assertEqual(canonical_room_display(raw), "Outside")

    def test_text_fallback_still_reuses_fingerprint(self):
        kg = KGMap()
        first = kg.resolve_arrival_location("Clearing", "Clearing\nA quiet clearing.")
        second = kg.resolve_arrival_location("Clearing", "Clearing\nA quiet clearing.")
        self.assertEqual(first, second)
        self.assertIsNone(kg.engine_num_for(first))

    def test_unseen_dark_room_generic_then_lit_label_stays_stable(self):
        kg = KGMap()
        dark = kg.resolve_arrival_location("Dark place", engine_num=70)
        kg.reset()
        lit = kg.resolve_arrival_location("Crystal Cave", engine_num=70)
        self.assertEqual(dark, "Dark place")
        self.assertEqual(lit, "Crystal Cave")
        self.assertIn("Crystal Cave", kg.room_registry[70]["titles_seen"])

    def test_event_identity_prefers_engine_num(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        room = agent.kg_map.resolve_arrival_location("Hallway", engine_num=52)
        agent.kg_map.confirm_arrival(room, engine_num=52)
        self.assertEqual(
            agent._hazard_room_identity_key(room),
            "hallway|obj52",
        )


if __name__ == "__main__":
    unittest.main()
