# Current State

Last updated: 2026-07-07

This folder is the current grounded-memory / retrieval experiment:

```text
versions/lplh2_grounded_memory_retrieval_patch_2026-07-07
```

It is built on:

```text
versions/lplh2_death_summary_retrieval_patch_2026-07-05
```

## Included

- KG/world-state reliability fixes.
- Action space enabled.
- Dedicated inventory reconciliation.
- Restored failed-command memory plus attempt ledger and same-state memory.
- Newest logging/timing files.
- Grounded score summaries with authoritative exact scoring command/location.
- Grounded death summaries with authoritative exact fatal command and death
  location fields.
- Current-epoch achievement filtering so already re-earned rewards stop being
  shown as active targets.
- Dynamic experience retrieval cap prioritized by decision value:
  unearned nearby achievements, relevant death warnings, object-anchored state
  changes/syntax lessons/clues, then only novel route memories as filler.
- Existing affordance agenda carryover strengthened so untried local ideas do
  not vanish just because the compact state changed.
- Movement-confirmed KG location updates.
- Same-title room fingerprinting for repeated room names.

## Not Included

- Full dynamic situation manager.
- Active planning.
- BFS navigation execution.
- Dedicated parser noun-canonicalization fix.
- Deterministic repeat bans.

## Runtime Defaults

The root convenience notebook and this version's notebook point to this folder.
They are configured for the Zork1 experiment setup with:

- `10` epochs,
- `250` max steps per epoch,
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
- `action_generation_log.txt`
- `action_failure_memory_log.txt`
- `attempt_ledger_log.txt`
- `auxiliary_gate_log.txt`
- `kg_location_log.txt`
- `module_timing_log.txt`
- JSON step log

Use `summary_module_log.txt` to audit authoritative score/death summaries,
`action_generation_log.txt` to inspect selected retrieval headers and agenda
context, and `kg_location_log.txt` to inspect false-location prevention and
same-title room splitting.
