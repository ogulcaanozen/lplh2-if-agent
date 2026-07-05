# LPLH2 Death Summary / Retrieval Patch

Created from:

```text
versions/lplh2_attempt_ledger_agenda_patch_2026-07-04
```

This version keeps the attempt-ledger / agenda anti-loop layer and improves how
cross-epoch experience memories are created and rendered. It does not add a new
module: death summaries still use the experience library, and dedup still uses
the Chroma-independent event index.

## Changes

1. Evidence-first death summaries
   - Score losses call `LOSS_EXPERIENCE_SUMMARIZATION_PROMPT`.
   - The prompt requires the last fatal exchanges, proximate cause, confirmed
     mechanics, fatal-action assessment, retry condition, and one fenced
     untested idea.
   - The current room's attempt ledger is included before the fatal step is
     recorded, so repeated combat/random outcomes can be described factually.

2. Epoch-aware experience metadata
   - Score and neutral summaries now store `epoch` and `step`.
   - Retrieved experience headers render staleness, e.g. `stored=epoch 2,
     step 94`.

3. Score-gain dedup across epochs
   - Positive score memories are keyed as `score:v1:gain:<location>:<action>:<delta>`.
   - If the same score gain is already indexed, the run logs a skipped duplicate
     instead of storing another near-identical achievement summary.

4. Retrieval diversity
   - The experience library can fetch more candidates than it renders.
   - The agent selects a compact mixed set, limits repeated kinds, reserves a
     non-route slot when available, and annotates route memories already present
     in the current KG map.

5. Separate death prompt notebook
   - `lplh2/test_death_summary_prompt_colab.ipynb` tests this prompt on fixed
     troll-death histories without running a full game.

## Intentionally Not Included

- Deterministic bans on repeated commands.
- Active situation planning or BFS navigation execution.
- Dynamic situation manager reintroduction.
