"""Focused checks for action-generation audit context."""

import io
import unittest

from .game_runner import GameRunner


class ActionGenerationLoggingTests(unittest.TestCase):
    def test_reward_and_route_context_are_written(self):
        runner = GameRunner.__new__(GameRunner)
        runner._action_generation_log_file = io.StringIO()
        runner._write_retrieved_summaries_record = lambda **_kwargs: None

        runner._write_action_generation_record(
            epoch=2,
            step=7,
            label="next_action_generation",
            command="north",
            generation={
                "parsed_command": "north",
                "known_rewards_context": (
                    "[+10] NOT EARNED this epoch | room: Office"
                ),
                "route_guidance": (
                    "From here, 'north' continues the recorded route."
                ),
            },
            observation="A corridor leads north.",
            score=0,
        )

        output = runner._action_generation_log_file.getvalue()
        self.assertIn("known scoring opportunities shown:", output)
        self.assertIn("[+10] NOT EARNED this epoch", output)
        self.assertIn("route guidance shown:", output)
        self.assertIn("'north' continues the recorded route", output)


if __name__ == "__main__":
    unittest.main()
