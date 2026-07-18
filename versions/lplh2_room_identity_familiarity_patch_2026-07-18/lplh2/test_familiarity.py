"""Behavioral tests for room, object, and command familiarity labels."""

import json
import unittest

from .action_memory import FailedActionMemory, StateScopedActionMemory
from .agent import LPLHAgent
from .attempt_ledger import AttemptLedger
from .familiarity import RoomFamiliarity
from .interaction_stats import InteractionStats
from .kg_map import KGMap
from .reward_directory import RewardDirectory


class FamiliarityTests(unittest.TestCase):
    def test_room_exhaustion_and_prior_epoch_fast_reclose(self):
        rooms = RoomFamiliarity()
        for _ in range(5):
            rooms.visit("r1", "Forest", 1)
        self.assertEqual(rooms.tier("r1")["tier"], "EXHAUSTED")
        rooms.reset_epoch()
        for _ in range(2):
            rooms.visit("r1", "Forest", 2)
        self.assertEqual(rooms.tier("r1")["tier"], "COVERED")
        rooms.visit("r1", "Forest", 2)
        self.assertEqual(rooms.tier("r1")["tier"], "EXHAUSTED")

    def test_untried_described_exit_keeps_room_fresh(self):
        rooms = RoomFamiliarity()
        for _ in range(8):
            rooms.visit("r1", "Kitchen", 1)
        rooms.update_described_exits("r1", "Kitchen", ["up"])
        self.assertEqual(rooms.tier("r1")["tier"], "FRESH")

    def test_productive_object_is_not_exhausted(self):
        stats = InteractionStats()
        for index in range(7):
            stats.record(
                "r1",
                "rug",
                f"touch rug {index}",
                "state_change" if index < 2 else "unproductive",
                0,
                1,
            )
        self.assertEqual(stats.tier("r1", "rug")["tier"], "COVERED")

    def test_command_exhausted_tag(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.attempt_ledger = AttemptLedger()
        agent.failed_action_memory = FailedActionMemory()
        agent.state_action_memory = StateScopedActionMemory()
        snapshot = {
            "location": "Room",
            "observation": "A wall.",
            "inventory": [],
            "visible_objects": [],
            "score": 0,
        }
        for step in range(1, 4):
            agent.attempt_ledger.record_step(
                "Room", "search wall", "Nothing unusual.", True, 0, False,
                "Room", False, False, "stored_unproductive", False,
                "same", step, 1,
            )
        text = agent._tried_here_context("Room", snapshot)
        self.assertIn("search wall [EXHAUSTED]", text)

    def test_reward_waypoint_exemption_reopens_exhausted_room(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.kg_map = KGMap()
        room, room_id = agent.kg_map.mint_room(
            "Forest",
            "Forest\nTrees.",
            epoch=1,
        )
        agent.kg_map.current_location = room
        agent.room_familiarity = RoomFamiliarity()
        for _ in range(5):
            agent.room_familiarity.visit(room_id, room, 1)
        agent.reward_directory = RewardDirectory()
        agent.reward_directory.add_or_update({
            "event_key": "reward",
            "points": 10,
            "location": "House",
            "scoring_command": "enter window",
            "route_hops": [[room, "east", "House"]],
            "route_hint": f"Start: {room}: east -> House",
        })
        agent.earned_score_event_keys_this_epoch = set()
        agent._active_goal_locations = lambda: []
        labels = agent._room_familiarity_by_location()
        self.assertEqual(labels[room]["tier"], "COVERED")
        self.assertIn("waypoint", labels[room]["rider"])
        self.assertNotIn("never", json.dumps(labels).lower())


if __name__ == "__main__":
    unittest.main()
