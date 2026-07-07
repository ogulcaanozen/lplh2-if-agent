# Current State

Last updated: 2026-07-07

This folder is the current enabler / agenda retrieval experiment:

```text
versions/lplh2_enabler_agenda_retrieval_patch_2026-07-07
```

It is built on:

```text
versions/lplh2_grounded_memory_retrieval_patch_2026-07-07
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
- Reward enabler summaries for state-changing setup commands that preceded
  score gains.
- Dynamic suppression of enabler summaries once their linked reward is earned
  in the current epoch.
- Sparse experience retrieval with up to 5 useful summaries.
- Existing affordance agenda carryover with completion consumption and
  observation-based relevance.
- Movement-confirmed KG location updates.
- Same-title room fingerprinting, including initial-room fingerprint seeding.
- Confirmed blocked-exit tracking plus one-shot regeneration before executing a
  repeated blocked direction.
- Dedicated retrieved-summary logging for every action-generation prompt.

## Not Included

- Full dynamic situation manager.
- Active planning.
- BFS navigation execution.
- Dedicated parser noun-canonicalization fix.

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
- `retrieved_summaries_log.txt`
- `action_failure_memory_log.txt`
- `attempt_ledger_log.txt`
- `auxiliary_gate_log.txt`
- `kg_location_log.txt`
- `module_timing_log.txt`
- JSON step log

Use `summary_module_log.txt` to audit achievement, death, and enabler
summaries; `retrieved_summaries_log.txt` to inspect which summaries were
selected for each action-generation prompt; `affordance_brainstorm_log.txt` and
`action_generation_log.txt` to inspect agenda lifecycle and blocked-direction
regeneration; and `kg_location_log.txt` to inspect location resolution and
same-title room splitting.
