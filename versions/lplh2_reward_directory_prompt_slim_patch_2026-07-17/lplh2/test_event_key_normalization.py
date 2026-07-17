"""Regression checks for Detective-style room names in event keys."""

import unittest

from .agent import LPLHAgent


class EventKeyNormalizationTests(unittest.TestCase):
    def _agent(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.earned_score_event_keys_this_epoch = set()
        agent.earned_score_location_reward_keys_this_epoch = set()
        return agent

    def test_named_markup_is_removed_but_room_brackets_survive(self):
        agent = self._agent()
        self.assertEqual(
            agent._normalize_event_piece("<< Outside >> #2"),
            "outside 2",
        )
        self.assertEqual(
            agent._normalize_event_piece("<< Outside >>"),
            "outside",
        )
        self.assertEqual(
            agent._normalize_event_piece("<loc>Kitchen</loc>"),
            "kitchen",
        )
        self.assertEqual(
            agent._normalize_event_piece("<step>open door</step>"),
            "open door",
        )

    def test_score_death_and_reward_keys_do_not_collapse_numbered_rooms(self):
        agent = self._agent()
        outside_score = agent._score_event_key(
            "gain", "east", "<< Outside >> #2", 10
        )
        hallway_score = agent._score_event_key(
            "gain", "east", "<< Hallway >> #2", 10
        )
        outside_death = agent._death_event_key("east", "<< Outside >> #2")
        hallway_death = agent._death_event_key("east", "<< Hallway >> #2")
        outside_reward = agent._score_location_reward_key("<< Outside >> #2", 10)
        hallway_reward = agent._score_location_reward_key("<< Hallway >> #2", 10)

        self.assertNotEqual(outside_score, hallway_score)
        self.assertNotEqual(outside_death, hallway_death)
        self.assertNotEqual(outside_reward, hallway_reward)
        self.assertIn("outside 2", outside_score)
        self.assertIn("hallway 2", hallway_score)
        self.assertIn("outside 2", outside_death)
        self.assertIn("hallway 2", hallway_death)
        self.assertIn("outside 2", outside_reward)
        self.assertIn("hallway 2", hallway_reward)

    def test_first_known_location_accepts_bracketed_detective_title(self):
        agent = self._agent()
        self.assertEqual(
            agent._first_known_location("<< Outside >>", "fallback"),
            "<< Outside >>",
        )
        self.assertEqual(
            agent._first_known_location("", "unknown", "Kitchen"),
            "Kitchen",
        )
        self.assertEqual(
            agent._first_known_location("", "unknown"),
            "Starting Location",
        )

    def test_earned_reward_fallback_distinguishes_sibling_rooms(self):
        agent = self._agent()
        agent.earned_score_location_reward_keys_this_epoch.add(
            agent._score_location_reward_key("<< Outside >>", 10)
        )
        plain_room_record = {
            "metadata": {
                "location_issued": "<< Outside >>",
                "score_change": 10,
            }
        }
        sibling_room_record = {
            "metadata": {
                "location_issued": "<< Outside >> #2",
                "score_change": 10,
            }
        }
        different_sibling_record = {
            "metadata": {
                "location_issued": "<< Hallway >> #2",
                "score_change": 10,
            }
        }

        self.assertTrue(agent._achievement_earned_this_epoch(plain_room_record))
        self.assertFalse(agent._achievement_earned_this_epoch(sibling_room_record))
        self.assertFalse(agent._achievement_earned_this_epoch(different_sibling_record))

    def test_neutral_event_keys_also_keep_location_identity(self):
        agent = self._agent()
        outside = agent._neutral_event_key(
            "navigation",
            "east",
            "",
            "<< Outside >> #2",
            prev_location="<< Outside >>",
        )
        hallway = agent._neutral_event_key(
            "navigation",
            "east",
            "",
            "<< Hallway >> #2",
            prev_location="<< Hallway >>",
        )
        self.assertNotEqual(outside, hallway)
        self.assertIn("outside 2", outside)
        self.assertIn("hallway 2", hallway)

    def test_zork_keys_and_empty_location_fallback_remain_stable(self):
        agent = self._agent()
        self.assertEqual(
            agent._score_event_key("gain", "open door", "Kitchen", 10),
            "score:v1:gain:kitchen::open door:10",
        )
        self.assertEqual(
            agent._death_event_key("down", "Cellar"),
            "death:v1:cellar::down",
        )
        self.assertEqual(
            agent._score_location_reward_key("", 10),
            "score_location_reward:v1:unknown::10",
        )


if __name__ == "__main__":
    unittest.main()
