"""Tests for separate local main and auxiliary LLM routing."""

import unittest
from unittest.mock import Mock

from .llm_client import LLMClient


def _uninitialized_client():
    client = object.__new__(LLMClient)
    client._aux_fallback_client = None
    client.chat = Mock(return_value="main")
    return client


class AuxiliaryRoutingTests(unittest.TestCase):
    def test_auxiliary_fallback_uses_separate_client_when_configured(self):
        main = _uninitialized_client()
        auxiliary = _uninitialized_client()
        auxiliary.chat.return_value = "auxiliary"

        main.set_auxiliary_client(auxiliary)

        result = main._chat_aux_fallback("summarize this", max_new_tokens=321)

        self.assertEqual(result, "auxiliary")
        main.chat.assert_not_called()
        auxiliary.chat.assert_called_once_with(
            "",
            "summarize this",
            temperature=0.0,
            max_new_tokens=321,
        )

    def test_auxiliary_fallback_preserves_single_model_default(self):
        main = _uninitialized_client()

        result = main._chat_aux_fallback("route modules", max_new_tokens=64)

        self.assertEqual(result, "main")
        main.chat.assert_called_once_with(
            "",
            "route modules",
            temperature=0.0,
            max_new_tokens=64,
        )

    def test_auxiliary_client_cannot_reference_main_instance(self):
        main = _uninitialized_client()

        with self.assertRaisesRegex(ValueError, "separate"):
            main.set_auxiliary_client(main)


if __name__ == "__main__":
    unittest.main()
