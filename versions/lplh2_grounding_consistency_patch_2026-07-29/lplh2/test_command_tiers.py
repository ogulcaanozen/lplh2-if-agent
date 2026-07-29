"""Behavioral tests for the consolidated state-aware command tiers."""

import unittest

from .action_memory import FailedActionMemory, StateScopedActionMemory
from .agent import LPLHAgent
from .attempt_ledger import AttemptLedger


class CommandTierTests(unittest.TestCase):
    def _agent(self):
        agent = LPLHAgent.__new__(LPLHAgent)
        agent.attempt_ledger = AttemptLedger()
        agent.failed_action_memory = FailedActionMemory()
        agent.state_action_memory = StateScopedActionMemory()
        return agent

    @staticmethod
    def _fingerprint(inventory=None, states=None):
        return {
            "location": "room",
            "inventory": list(inventory or []),
            "object_states": dict(states or {}),
        }

    def _record(self, agent, command, fingerprint, count=3,
                outcome="stored_unproductive", environment_changed=False):
        for step in range(1, count + 1):
            agent.attempt_ledger.record_step(
                location="Room",
                command=command,
                observation="Nothing useful happens.",
                action_valid=True,
                reward_change=0,
                location_changed=False,
                destination="Room",
                inventory_changed=False,
                environment_changed=environment_changed,
                repetition_status=outcome,
                terminal_defeat=False,
                state_key="legacy-state",
                step=step,
                epoch=1,
                world_fingerprint=fingerprint,
            )

    def test_covered_and_exhausted_boundaries(self):
        agent = self._agent()
        fingerprint = self._fingerprint(states={"wall": ""})
        self._record(agent, "search wall", fingerprint, count=2)
        covered = agent._tried_here_context("Room", fingerprint)
        self.assertIn("search wall [COVERED] x2", covered)

        self._record(agent, "search wall", fingerprint, count=1)
        exhausted = agent._tried_here_context("Room", fingerprint)
        self.assertIn("search wall [EXHAUSTED] x3", exhausted)

    def test_productive_history_stays_covered(self):
        agent = self._agent()
        fingerprint = self._fingerprint(states={"rug": "moved"})
        self._record(
            agent,
            "move rug",
            fingerprint,
            count=3,
            outcome="useful_progress",
            environment_changed=True,
        )
        context = agent._tried_here_context("Room", fingerprint)
        self.assertIn("move rug [COVERED] x3", context)

    def test_unrelated_inventory_gain_does_not_reopen_exhausted_command(self):
        agent = self._agent()
        old = self._fingerprint(states={"grating": "locked"})
        current = self._fingerprint(
            inventory=["brass key"],
            states={"grating": "locked"},
        )
        self._record(agent, "open grating", old)
        context = agent._tried_here_context("Room", current)
        self.assertIn("open grating [EXHAUSTED] x3", context)
        self.assertNotIn("[REOPENED]", context)
        self.assertEqual(
            agent._last_reopen_audit[-1]["decision"],
            "not_reopened",
        )

    def test_item_named_in_command_reopens_exhausted_command(self):
        agent = self._agent()
        old = self._fingerprint(states={"grating": "locked"})
        current = self._fingerprint(
            inventory=["brass key"],
            states={"grating": "locked"},
        )
        self._record(agent, "unlock grating with brass key", old)
        context = agent._tried_here_context("Room", current)
        self.assertIn(
            "unlock grating with brass key [REOPENED] x3",
            context,
        )
        self.assertIn("you now carry the brass key", context)

    def test_named_object_state_change_reopens(self):
        agent = self._agent()
        old = self._fingerprint(states={"rug": "covering floor"})
        current = self._fingerprint(states={"rug": "moved"})
        self._record(agent, "look under rug", old)
        context = agent._tried_here_context("Room", current)
        self.assertIn("look under rug [REOPENED] x3", context)
        self.assertIn("the rug is now moved", context)

    def test_unrelated_object_change_does_not_reopen(self):
        agent = self._agent()
        old = self._fingerprint(states={
            "rug": "covering floor",
            "lamp": "off",
        })
        current = self._fingerprint(states={
            "rug": "covering floor",
            "lamp": "on",
        })
        self._record(agent, "look under rug", old)
        context = agent._tried_here_context("Room", current)
        self.assertIn("look under rug [EXHAUSTED] x3", context)
        self.assertNotIn("[REOPENED]", context)

    def test_world_fingerprint_excludes_observation_and_score(self):
        agent = self._agent()
        first = agent._world_fingerprint(
            "Room",
            ["key"],
            [{"name": "door", "state": "locked"}],
        )
        second = agent._world_fingerprint(
            "Room",
            ["key"],
            [{"name": "door", "state": "locked"}],
        )
        self.assertEqual(first, second)
        self.assertNotIn("observation", first)
        self.assertNotIn("score", first)


if __name__ == "__main__":
    unittest.main()
