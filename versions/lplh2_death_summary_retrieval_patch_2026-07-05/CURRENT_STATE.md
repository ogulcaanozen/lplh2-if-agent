# Current State

Last updated: 2026-07-05

This folder is the current death-summary / retrieval experiment:

```text
versions/lplh2_death_summary_retrieval_patch_2026-07-05
```

It is built on:

```text
versions/lplh2_attempt_ledger_agenda_patch_2026-07-04
```

## Included

- KG/world-state reliability fixes from the July 3 snapshot.
- Action space enabled.
- Dedicated inventory reconciliation.
- Newest logging/timing files.
- Attempt ledger rendered to the main LLM and logs.
- Per-location affordance agenda cache with tried-count annotations.
- Experience retrieval headers with summary kind/use labels.
- Main LLM repeat self-check.
- Evidence-first score-loss/death summarization.
- Score-gain summary dedup across epochs.
- Epoch/step staleness headers for retrieved experiences.
- Retrieval over-fetching and diversity selection.

## Not Included

- Full dynamic situation manager.
- Active planning.
- BFS navigation execution.
- Deterministic repeat bans.

## Runtime Defaults

The root convenience notebook should point to this folder. It is configured for:

- `1` epoch,
- `250` max steps,
- main LLM `Qwen/Qwen2.5-14B-Instruct`,
- auxiliary modules through local LLM fallback unless OpenAI model env vars are
  explicitly set,
- FM model `Qwen/Qwen2.5-1.5B-Instruct` plus the Drive LoRA adapter.

## Logs To Inspect

Each experiment folder should include:

- `run_log.txt`
- `summary_module_log.txt`
- `situation_memory_log.txt`
- `affordance_brainstorm_log.txt`
- `action_failure_memory_log.txt`
- `action_generation_log.txt`
- `attempt_ledger_log.txt`
- `auxiliary_gate_log.txt`
- `kg_location_log.txt`
- `module_timing_log.txt`
- JSON step log

The most relevant file for this patch is `action_generation_log.txt`, which
includes command history in the room and the main LLM repeat self-check.
Use `summary_module_log.txt` to audit the new score-loss summaries and
`steplog.json` to inspect `retrieval_debug`, duplicate score-gain skips, and
experience headers.
