"""Behavioral tests for the persistent reward directory."""

import unittest

from .agent import LPLHAgent
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


if __name__ == "__main__":
    unittest.main()
