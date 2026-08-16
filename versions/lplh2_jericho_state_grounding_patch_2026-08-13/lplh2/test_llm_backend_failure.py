"""Checks that unrecoverable model backends cannot produce fallback loops."""

import unittest

from .agent import _is_fatal_llm_backend_error


class LLMBackendFailureTests(unittest.TestCase):
    def test_engine_core_failure_is_fatal(self):
        self.assertTrue(
            _is_fatal_llm_backend_error(
                RuntimeError("EngineCore encountered an issue")
            )
        )

    def test_cuda_oom_is_fatal(self):
        self.assertTrue(
            _is_fatal_llm_backend_error(
                RuntimeError("CUDA out of memory")
            )
        )

    def test_parse_failure_is_not_backend_fatal(self):
        self.assertFalse(
            _is_fatal_llm_backend_error(
                ValueError("response did not contain a command tag")
            )
        )


if __name__ == "__main__":
    unittest.main()
