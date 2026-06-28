# LPLH2 Auxiliary Gate Patch

Date: 2026-06-26

This folder is a separate snapshot of the LPLH2 code with the auxiliary gate
optimization patch applied. It is intentionally stored under `versions/` so the
current root project files remain unchanged.

## Purpose

The patch reduces unnecessary auxiliary LLM calls while keeping the framework
LLM-centered. It introduces one gate LLM call that routes only these selected
mechanisms:

- environmental change detection,
- stored situation detection,
- affordance brainstorming.

The gate does not choose the next game action. It only decides which helper
modules should run after a completed step.

## Main Changes

- Added `AUXILIARY_MODULE_GATE_PROMPT` in `lplh2/prompts.py`.
- Added `LLMClient.gate_auxiliary_modules(...)` in `lplh2/llm_client.py`.
- Added agent-side parsing, fallback, and routing in `lplh2/agent.py`.
- Navigation, environmental, and narrative neutral-summary triggers are now
  supplied by the gate when the gate succeeds.
- Environmental change status is also supplied by the gate when the gate succeeds.
- Stored situation detection is skipped when the gate says there is no likely
  new future-return situation.
- Affordance brainstorming runs fresh only when the gate says it is useful.
- When affordance brainstorming is skipped, exact-state cached ideas are reused
  if available.
- Added `AffordanceBrainstormer.cached_ideas_for_state(...)` so skipped
  brainstorming can reuse cached alternatives without another LLM call.

## Fallback Behavior

If the gate call fails or returns invalid JSON, the agent falls back
conservatively:

- legacy environmental-change detection runs,
- stored situation detection runs,
- affordance brainstorming runs.

This preserves the previous behavior when the new gate is unavailable.

## Follow-Up Polish

- Initialized the auxiliary gate debug fields in `LLMClient.__init__`.
- Added a dedicated per-experiment `auxiliary_gate_log.txt` with the gate
  prompt, raw response, parsed decision, fallback status, and environmental
  change detail for each step where the gate runs.
- Extended the gate response with a `summary_triggers` section so
  `navigation`, `environmental`, and `narrative` summaries are LLM-routed.
  The old local summary-trigger checks remain only as a gate-failure fallback.
- Added fuller affordance context for brainstorming: unproductive commands,
  same-state tried commands, and pending carryover commands are now supplied to
  the brainstorm LLM before it generates new ideas.
- Replaced the main action prompt's raw brainstorm list with an explicit
  affordance agenda that separates pending commands from commands already tried
  in the same state/location.

## Verification

The patched snapshot was syntax-checked locally with:

```text
python -m compileall lplh2
```

The gate prompt was also checked with sample `.format(...)` inputs.
