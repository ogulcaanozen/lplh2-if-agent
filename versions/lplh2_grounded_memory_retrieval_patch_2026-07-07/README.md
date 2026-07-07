# LPLH2 Grounded Memory / Retrieval Patch

Experimental LPLH2 framework for text-based interactive fiction agents.

This snapshot is `lplh2_grounded_memory_retrieval_patch_2026-07-07`. It starts
from `lplh2_death_summary_retrieval_patch_2026-07-05` and keeps the restored
failed-command memory, attempt ledger, action-space context, inventory
reconciliation, and timing/logging setup.

New in this version:

- score summaries receive authoritative `scoring_action`, scoring location,
  location-after, reward, epoch, and step fields;
- score summaries are retried once, then prefixed with a factual scoring line if
  the LLM omits the exact command/location;
- death summaries explicitly receive the exact fatal command, fatal-command
  issue location, and after-loss location;
- retrieved achievements are filtered out once the same score event has already
  been re-earned in the current epoch;
- experience retrieval is a cap, not a quota: nearby unearned achievements,
  relevant death warnings, and object-anchored mechanics outrank redundant
  route memories;
- existing affordance carryover is more persistent for still-visible local
  objects/inventory/conditions;
- KG location updates are movement-confirmed, so a room merely seen through a
  window no longer becomes the current location;
- KG can split same-title rooms with different description fingerprints, such
  as separate `Clearing` nodes.

Not included in this version:

- active planning/BFS execution;
- a new separate agenda module;
- parser noun canonicalization as a dedicated fix;
- deterministic hard bans on repeated commands.

The repo excludes game ROMs, Chroma databases, logs, Drive data, API keys, and
LoRA adapter weights.
