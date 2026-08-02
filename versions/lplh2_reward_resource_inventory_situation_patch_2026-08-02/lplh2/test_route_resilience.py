"""Behavioral tests for persistent structured reward routes."""

import unittest

from .reward_directory import RewardDirectory


class RouteResilienceTests(unittest.TestCase):
    def test_next_hop_and_failure_lifecycle(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "reward",
            "points": 25,
            "location": "Living Room",
            "scoring_command": "down",
            "route_hint": "Start: Forest: east -> House: west -> Living Room",
            "route_hops": [
                ["Forest", "east", "House"],
                ["House", "west", "Living Room"],
            ],
        })
        self.assertEqual(
            directory.next_hop_from("Forest", set())["command"],
            "east",
        )
        directory.flag_hop_failure("reward", "Forest", "east")
        self.assertIn("this hop failed once", directory.render(set()))
        directory.clear_hop_failure("reward", "Forest", "east")
        self.assertNotIn("this hop failed once", directory.render(set()))

    def test_shorter_route_replaces_but_longer_does_not(self):
        directory = RewardDirectory()
        directory.add_or_update({
            "event_key": "reward",
            "points": 10,
            "location": "Goal",
            "scoring_command": "take prize",
            "route_hint": "Start: A: east -> B: east -> C: east -> Goal",
            "route_hops": [
                ["A", "east", "B"],
                ["B", "east", "C"],
                ["C", "east", "Goal"],
            ],
        })
        directory.add_or_update({
            "event_key": "reward",
            "route_hint": "Start: A: north -> Goal",
            "route_hops": [["A", "north", "Goal"]],
        })
        self.assertEqual(
            directory.entries()[0]["route_hops"],
            [["A", "north", "Goal"]],
        )
        directory.add_or_update({
            "event_key": "reward",
            "route_hint": "Start: A: west -> X: north -> Goal",
            "route_hops": [
                ["A", "west", "X"],
                ["X", "north", "Goal"],
            ],
        })
        self.assertEqual(
            directory.entries()[0]["route_hops"],
            [["A", "north", "Goal"]],
        )


if __name__ == "__main__":
    unittest.main()
