# Main LPLH2 Version

Current main version as of 2026-07-03:

```text
versions/lplh2_inv_gate_worldstate_selective_2026-07-02
```

This is the thesis/testing baseline unless a newer version is explicitly marked
as main.

## Why This Version Is Main

This version gave strong Zork1 performance while staying simpler than the
dynamic-planning branches. It starts from the higher-performing
`lplh2_inventory_gate_patch_2026-06-28` baseline and selectively adds only the
reliability pieces that helped without adding too much context noise.

Included:

- KG/world-state reliability fixes.
- Dedicated inventory reconciliation.
- Newest logging and module timing.
- Action space enabled and supplied to both the main action LLM and the
  affordance brainstormer.
- Passive stored-situation detection/resolution.
- Affordance brainstorming with carryover and failed/same-state context.
- Experience summaries for score changes and selected neutral events.

Not included:

- Full dynamic situation manager.
- Active situation planning.
- BFS navigation or forced route execution.

## Runtime LLM Roles

- Main action LLM: usually `Qwen/Qwen2.5-14B-Instruct`.
- FM model: `Qwen/Qwen2.5-1.5B-Instruct` plus the
  `fm_adapter_v4_autoplay_failures` LoRA adapter.
- Auxiliary/summarization LLM: local main-LLM fallback unless
  `LPLH_LLM_ES_MODEL` is explicitly set.
- Affordance brainstorming LLM: local main-LLM fallback unless
  `LPLH_BRAINSTORM_MODEL` is explicitly set.

## Main Mechanisms

1. **KG Map**
   - Stores current location, visited rooms, confirmed navigation graph,
     visible room objects, object relations, frontier directions, known object
     locations, and inventory.
   - Uses FM relation extraction, plus conservative room-title correction when
     the observation starts with a clear room title.

2. **Action Space**
   - Uses the FM split task after valid actions.
   - Stores learned valid verb/object templates.
   - Feeds learned action-space context to the action prompt and affordance
     brainstorm prompt.

3. **Auxiliary Gate**
   - One LLM routing call per completed step.
   - Decides command outcome, summary triggers, inventory reconciliation route,
     stored-situation detection route, and affordance brainstorming route.
   - Does not choose game commands.

4. **Inventory Reconciliation**
   - Dedicated LLM call only when routed by the auxiliary gate.
   - Adds/removes/sets carried inventory from semantic observation evidence.
   - Removes taken items from room object lists.

5. **Affordance Brainstorming**
   - Suggests concrete object/inventory/situation commands.
   - Receives visible objects, inventory, stored situations, learned action
     space, known failed commands, same-state tried commands, and recent
     command outcomes.
   - Advisory only; the main action LLM still chooses the final command.

6. **Experience Memory**
   - Stores score-change summaries.
   - Stores selected neutral summaries: navigation, narrative, environmental
     changes, and error-correction.
   - Uses duplicate keys to avoid repeated neutral memories across epochs.

7. **Stored Situations**
   - Passive compact memory of unresolved blockers/hazards/future-return
     situations.
   - Current format:

```json
{
  "location": "where the issue exists",
  "situation": "short unresolved problem"
}
```

## Logs To Inspect

Each experiment should create:

- `run_log.txt`
- `steplog.json`
- `summary_module_log.txt`
- `situation_memory_log.txt`
- `affordance_brainstorm_log.txt`
- `action_failure_memory_log.txt`
- `action_generation_log.txt`
- `auxiliary_gate_log.txt`
- `kg_location_log.txt`
- `module_timing_log.txt`

The root Colab notebook currently points to this version.
