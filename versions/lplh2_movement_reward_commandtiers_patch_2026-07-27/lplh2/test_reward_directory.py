"""Behavioral tests for the persistent reward directory."""

import unittest

from .agent import LPLHAgent
from .kg_map import KGMap
from .reward_directory import (
    RewardDirectory,
    compress_epoch_path,
    render_route_hint,
)


class RewardDirectoryTests(unittest.TestCase):
    def test_score_summary_setup_commands_are_grounded_and_exclude_scoring(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        summary = (
            "A grounded reward summary.\n"
            "<setup_commands>move rug; open trap door; down</setup_commands>"
        )
        commands = agent._parse_setup_commands(summary, scoring_action="down")
        self.assertEqual(commands, ["move rug", "open trap door"])

        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "reward:trapdoor",
            "points": 25,
            "location": "Living Room",
            "scoring_command": "down",
            "setup_commands": commands,
            "first_seen_epoch": 1,
        })
        self.assertEqual(
            directory.entries()[0]["setup_commands"],
            ["move rug", "open trap door"],
        )

    def test_cycle_compression_keeps_surviving_commands(self):
        path = [
            ("A", "east", "B"),
            ("B", "west", "A"),
            ("A", "north", "C"),
            ("C", "east", "D"),
            ("D", "west", "C"),
            ("C", "up", "E"),
        ]
        self.assertEqual(
            compress_epoch_path(path),
            [("A", "north", "C"), ("C", "up", "E")],
        )
        self.assertEqual(
            render_route_hint(compress_epoch_path(path)),
            "Start: A: north -> C: up -> E",
        )

    def test_earned_render_flips_and_epoch_clear_reverts(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "small",
            "points": 10,
            "location": "Room A",
            "scoring_command": "take paper",
            "setup_commands": [],
            "first_seen_epoch": 1,
        })
        directory.add_or_update({
            "event_key": "large",
            "points": 25,
            "location": "Room B",
            "scoring_command": "down",
            "setup_commands": ["open door"],
            "first_seen_epoch": 1,
        })

        unearned = directory.render(set())
        self.assertLess(unearned.index("[+25]"), unearned.index("[+10]"))
        earned = directory.render({"large"})
        self.assertIn("[+25] already earned this epoch", earned)
        self.assertIn("[+10] NOT EARNED this epoch", earned)
        reverted = directory.render(set())
        self.assertIn("[+25] NOT EARNED this epoch", reverted)

    def test_epoch_reset_preserves_and_full_reset_clears(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "reward",
            "points": 10,
            "location": "Office",
            "scoring_command": "take paper",
            "setup_commands": [],
            "first_seen_epoch": 1,
        })
        directory.reset_epoch_flags()
        self.assertEqual(len(directory), 1)
        directory.full_reset()
        self.assertEqual(len(directory), 0)

    def test_route_cross_reference_inserts_unearned_setup(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "window",
            "points": 10,
            "location": "Behind House",
            "scoring_command": "enter window",
            "setup_commands": ["open window"],
            "first_seen_epoch": 1,
        })
        directory.add_or_update({
            "event_key": "cellar",
            "points": 25,
            "location": "Living Room",
            "scoring_command": "down",
            "setup_commands": ["move rug", "open trap door"],
            "route_hint": (
                "Start: West of House: south -> South of House: east -> "
                "Behind House: enter window -> Kitchen: west -> Living Room"
            ),
            "first_seen_epoch": 1,
        })
        rendered = directory.render(set())
        self.assertIn("Behind House (setup: open window)", rendered)

    def test_reearning_refreshes_only_to_a_shorter_route(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.reward_directory = RewardDirectory()
        agent.current_epoch = 1
        agent._epoch_path = [
            ("Start", "north", "Forest"),
            ("Forest", "east", "Kitchen"),
            ("Kitchen", "west", "Living Room"),
        ]
        agent._record_reward_directory_event(
            event_key="reward:cellar",
            points=25,
            location="Living Room",
            scoring_command="down",
            setup_commands=["move rug", "open trap door"],
        )
        first = agent.reward_directory.entries()[0]
        self.assertTrue(first["route_hint"].endswith("Living Room"))
        self.assertNotIn("Cellar", first["route_hint"])

        agent.current_epoch = 2
        agent._epoch_path = [
            ("Start", "west", "Living Room"),
        ]
        agent._record_reward_directory_event(
            event_key="reward:cellar",
            points=25,
            location="Living Room",
            scoring_command="down",
            setup_commands=[],
        )
        refreshed = agent.reward_directory.entries()[0]
        self.assertEqual(
            refreshed["route_hint"],
            "Start: Start: west -> Living Room",
        )
        self.assertEqual(
            refreshed["setup_commands"],
            ["move rug", "open trap door"],
        )

    def test_distance_ranking_and_nearby_side_rewards(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "far",
            "points": 50,
            "location": "Far Room",
            "scoring_command": "take crown",
            "route_hops": [
                ["Start", "north", "A"],
                ["A", "north", "B"],
                ["B", "north", "Far Room"],
            ],
        })
        directory.add_or_update({
            "event_key": "near",
            "points": 10,
            "location": "Near Room",
            "scoring_command": "take coin",
            "route_hops": [["Start", "east", "Near Room"]],
        })
        directory.add_or_update({
            "event_key": "side",
            "points": 5,
            "location": "Side Room",
            "scoring_command": "take note",
            "route_hops": [
                ["Start", "west", "C"],
                ["C", "west", "Side Room"],
            ],
        })
        ranked = directory.ranked_unearned("Start", set())
        self.assertEqual([item["event_key"] for item in ranked], [
            "near", "side", "far",
        ])
        rendered = directory.render(set(), current_location="Start")
        self.assertLess(rendered.index("take coin"), rendered.index("take crown"))
        side = directory.nearby_side_rewards("Start", set(), max_hops=2)
        self.assertEqual([item["event_key"] for item in side], ["side"])
        self.assertEqual(side[0]["first_command"], "west")

    def test_registry_scoped_dedupe_merges_only_same_physical_room(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "copy-a",
            "points": 10,
            "location": "Hallway",
            "location_registry_id": "r7",
            "scoring_command": "north",
            "setup_commands": ["open door"],
            "route_hops": [
                ["Start", "east", "Middle"],
                ["Middle", "north", "Hallway"],
            ],
            "first_seen_epoch": 2,
        })
        directory.add_or_update({
            "event_key": "copy-b",
            "points": 10,
            "location": "Hallway #3",
            "location_registry_id": "r7",
            "scoring_command": "north",
            "setup_commands": ["unlock door"],
            "route_hops": [["Start", "north", "Hallway #3"]],
            "first_seen_epoch": 1,
        })
        self.assertEqual(len(directory), 1)
        merged = directory.entries()[0]
        self.assertEqual(len(merged["route_hops"]), 1)
        self.assertEqual(
            set(merged["setup_commands"]),
            {"open door", "unlock door"},
        )
        self.assertEqual(merged["first_seen_epoch"], 1)
        self.assertIn("copy-a", merged["merged_event_keys"])

        directory.add_or_update({
            "event_key": "other-room",
            "points": 10,
            "location": "Hallway #4",
            "location_registry_id": "r8",
            "scoring_command": "north",
        })
        self.assertEqual(len(directory), 2)

    def test_unearned_entries_honors_merged_event_key_aliases(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "copy-a",
            "points": 10,
            "location": "Office",
            "location_registry_id": "r3",
            "scoring_command": "take paper",
        })
        directory.add_or_update({
            "event_key": "copy-b",
            "points": 10,
            "location": "Office",
            "location_registry_id": "r3",
            "scoring_command": "take paper",
        })
        self.assertEqual(len(directory), 1)
        self.assertEqual(
            [entry["event_key"] for entry in directory.unearned_entries(set())],
            [directory.entries()[0]["event_key"]],
        )
        self.assertEqual(directory.unearned_entries({"copy-b"}), [])

    def test_familiarity_rider_honors_merged_event_key_aliases(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        office, registry_id = agent.kg_map.mint_room(
            "Office",
            "Office\nA quiet office.",
            epoch=1,
        )
        agent.kg_map.confirm_arrival(office, "Office\nA quiet office.")
        agent.reward_directory = RewardDirectory()
        for event_key in ("copy-a", "copy-b"):
            agent.reward_directory.add_or_update({
                "event_key": event_key,
                "points": 10,
                "location": office,
                "location_registry_id": registry_id,
                "scoring_command": "take paper",
            })
        agent.earned_score_event_keys_this_epoch = {"copy-b"}

        familiarity = agent._room_familiarity_by_location()

        self.assertNotIn("rider", familiarity[office])

    def test_adaptive_cutoff_keeps_all_nearby_then_three_far(self):
        directory = RewardDirectory()
        edges = {"Start": {}}
        for index in range(1, 9):
            room = f"Near {index}"
            command = f"go-{index}"
            edges["Start"][command] = room
            directory.add_or_update({
                "event_key": f"near-{index}",
                "points": index,
                "location": room,
                "scoring_command": f"take item {index}",
            })
        for index in range(1, 6):
            directory.add_or_update({
                "event_key": f"far-{index}",
                "points": 100 + index,
                "location": f"Unknown {index}",
                "scoring_command": f"claim prize {index}",
            })
        rendered = directory.render(
            set(),
            current_location="Start",
            extra_edges=edges,
        )
        self.assertEqual(rendered.count("NOT EARNED this epoch"), 10)
        for index in range(1, 9):
            self.assertIn(f"take item {index}", rendered)


if __name__ == "__main__":
    unittest.main()
