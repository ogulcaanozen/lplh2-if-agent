"""Behavioral tests for causal reward-setup extraction."""

import unittest

from .agent import LPLHAgent


class SetupFilterTests(unittest.TestCase):
    def setUp(self):
        self.agent = LPLHAgent.__new__(LPLHAgent)

    def test_movement_and_observation_are_rejected(self):
        for command in (
            "north",
            "go east",
            "enter window",
            "look at window",
            "examine rug",
            "read leaflet",
            "listen",
        ):
            self.assertFalse(
                self.agent._is_causal_setup_command(command),
                command,
            )

    def test_state_changing_commands_pass_and_cap_is_three(self):
        summary = (
            "<setup_commands>open window; move rug; unlock door; "
            "take lantern; east; look at window</setup_commands>"
        )
        self.assertEqual(
            self.agent._parse_setup_commands(summary, "down"),
            ["open window", "move rug", "unlock door"],
        )


if __name__ == "__main__":
    unittest.main()
