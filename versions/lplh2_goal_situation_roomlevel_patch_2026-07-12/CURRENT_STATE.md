# Current State

Last updated: 2026-07-12

This folder is the room-level goal-situation experimental version:

```text
versions/lplh2_goal_situation_roomlevel_patch_2026-07-12
```

It is built on:

```text
versions/lplh2_goal_situation_patch_2026-07-11
```

## Included

- Persistent room-level death counters and compact fatal-action evidence.
- Goal creation after two deaths in one room, independent of fatal command key.
- Concrete hypothesis item keywords for inventory matching.
- Auditable unprepared survivals and reversible false confirmations.
- Merged per-room death-warning retrieval headers.

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
- Terminal defeat/victory classification that does not depend on a score loss.
- Deterministic early-epoch retrieval of unearned starting achievements.
- Visit-scoped repeated blocked-navigation adjudication and bounded alternative
  generation, while preserving explicit route hints from game text.
- Event-key normalization that preserves bracketed room titles such as
  `<< Outside >>` and distinguishes fingerprinted same-title rooms.
- Exact physical-room experience retrieval: achievements, enablers, death
  warnings, routes, and neutral memories are not borrowed from neighboring or
  destination rooms.
- Persistent goal-class situations inferred from repeated identical deaths,
  including room-identity merge, gateway evidence, refutation, confirmation,
  decline, avoid, and five-open-goal cap behavior.
- Cross-producer suppression so an open goal replaces redundant observation
  situations for the same hazard room.
- Precondition-hypothesis prompts in `summary_module_log.txt` and complete goal
  transitions in `situation_memory_log.txt`.

## Not Included

- Full dynamic situation manager.
- Active planning.
- BFS navigation execution.
- Dedicated parser noun-canonicalization fix.

## Runtime Defaults

The root convenience notebook and this version's notebook point to this folder.
The current notebook is configured for the Detective experiment with:

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
