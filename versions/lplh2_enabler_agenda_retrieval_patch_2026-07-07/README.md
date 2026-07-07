# LPLH2 Enabler / Agenda Retrieval Patch

Experimental LPLH2 framework for text-based interactive fiction agents.

This snapshot is `lplh2_enabler_agenda_retrieval_patch_2026-07-07`. It starts
from `lplh2_grounded_memory_retrieval_patch_2026-07-07` and keeps the restored
failed-command memory, attempt ledger, action-space context, inventory
reconciliation, grounded score/death summaries, and timing/logging setup.

New in this version:

- filtered summary retrieval returns an empty result when a query succeeds but
  finds no matches, instead of falling back to unrelated recent memories;
- experience retrieval is a true cap of up to `5` useful summaries, not a forced
  quota;
- reward enabler summaries are stored from recent state-changing setup actions
  before a score gain, linked to the same score event key as the achievement;
- enabler summaries are hidden for the current epoch once the linked reward has
  been earned;
- completed pending agenda commands are consumed when the ledger shows they
  already caused a state change, movement, or score;
- brainstorm carryover relevance uses the current observation text plus visible
  objects, instead of letting inventory keep stale acquisition ideas alive;
- fresh brainstorm ideas outrank carried ideas when merged;
- confirmed blocked directions are tracked in the KG and trigger one action
  regeneration if the main LLM selects the same blocked direction again;
- the initial room receives a description fingerprint, and empty same-title
  fingerprints are adopted instead of creating spurious `#2` rooms.

Not included in this version:

- active planning/BFS execution;
- a new separate agenda module;
- parser noun canonicalization as a dedicated fix.

The repo excludes game ROMs, Chroma databases, logs, Drive data, API keys, and
LoRA adapter weights.
