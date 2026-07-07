# LPLH2 Grounded Memory / Retrieval Patch

Created from:

```text
versions/lplh2_death_summary_retrieval_patch_2026-07-05
```

## Purpose

The previous multi-epoch run showed that epoch 1 could solve valuable chains,
but later epochs sometimes failed because the retrieved memory was wrong,
irrelevant, or crowded out by route facts. This patch keeps the existing module
layout and makes cross-epoch memory more factual and more selective.

## Changes

1. Grounded score summaries
   - `LLMClient.summarize_experience` now receives authoritative
     `scoring_action`, `location_issued`, and `location_after`.
   - The prompt says the exact scoring command caused the reward and must not
     be replaced by an earlier setup command.
   - The agent validates positive-score summaries, retries once if the exact
     action/location are missing, and prepends an authoritative fact if the
     retry still fails.

2. Grounded death summaries
   - `summarize_loss_experience` now receives the fatal command issue location
     separately from the after-loss location.
   - The prompt records `death_location`, exact `fatal_action`, and generic
     `unsafe_condition_evidence` without adding a game-specific hazard module.

3. Current-epoch achievement filtering
   - The agent tracks score event keys earned in the current epoch.
   - Achievement memories remain stored across epochs but stop rendering as
     nearby reward targets once re-earned in the current epoch.

4. Decision-value retrieval cap
   - Retrieval is no longer a forced quota of three similar summaries.
   - Selection prefers unearned nearby achievements, relevant death warnings,
     object-anchored state changes/syntax lessons/clues, then novel routes.
   - Route memories already represented in the current KG map are not used as
     padding.

5. More durable existing agenda carryover
   - The existing affordance carryover keeps stale-state ideas if they still
     mention visible objects, inventory items, or condition context.
   - This is an adjustment to the current pending-command system, not a new
     planning module.

6. KG location reliability
   - Room-title location updates are accepted only after movement-like commands,
     when current location is unknown, or when the title matches the current
     room.
   - Non-movement observations that merely mention another room no longer move
     the current location.
   - Movement-confirmed arrivals use title+description fingerprints to split
     repeated room names such as `Clearing` / `Clearing #2`.

## Validation

Run:

```text
python -m compileall versions\lplh2_grounded_memory_retrieval_patch_2026-07-07\lplh2
```

Also run the included notebook for a full behavioral smoke test.
