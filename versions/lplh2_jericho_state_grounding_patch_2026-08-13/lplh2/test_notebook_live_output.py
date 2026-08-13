"""Tests for bounded live output in notebook experiments."""

import unittest

from .game_runner import GameRunner


class NotebookLiveOutputTest(unittest.TestCase):
    def test_window_rolls_after_configured_number_of_steps(self):
        runner = GameRunner.__new__(GameRunner)
        runner.notebook_live_output_steps = 25

        self.assertFalse(runner._should_roll_notebook_output(1))
        self.assertFalse(runner._should_roll_notebook_output(25))
        self.assertTrue(runner._should_roll_notebook_output(26))
        self.assertFalse(runner._should_roll_notebook_output(50))
        self.assertTrue(runner._should_roll_notebook_output(51))

    def test_zero_disables_output_rolling(self):
        runner = GameRunner.__new__(GameRunner)
        runner.notebook_live_output_steps = 0

        self.assertFalse(runner._should_roll_notebook_output(26))


if __name__ == "__main__":
    unittest.main()
