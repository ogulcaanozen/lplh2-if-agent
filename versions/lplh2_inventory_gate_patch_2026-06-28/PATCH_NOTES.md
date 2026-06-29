# LPLH2 Inventory Gate Patch

Date: 2026-06-28

This folder is a separate snapshot of the latest LPLH2 auxiliary-gate version
with a new LLM-gated inventory reconciliation patch. The previous snapshot under
`versions/lplh2_auxiliary_gate_patch_2026-06-26/` is left unchanged.

## Purpose

The old inventory update path used command-prefix logic such as `take`, `drop`,
`eat`, `drink`, and `give`. This snapshot moves inventory correction into the
existing auxiliary gate so inventory updates are decided from the action,
observation, current inventory, and inventory-before-step context.

The gate still does not choose the next game command. It routes helper modules
and now also emits structured inventory evidence.

## Main Changes

- Added `command_outcome` to `AUXILIARY_MODULE_GATE_PROMPT`.
  - `accepted`
  - `rejected`
  - `no_effect`
  - `unknown`
- Added `inventory_update` to the auxiliary gate output:
  - `changed`
  - `authoritative`
  - `items_now_carried`
  - `items_added`
  - `items_removed`
  - `reason`
- Added `Inventory Before This Step` to the gate prompt.
- Removed the old agent-side hard-coded inventory mutation based on command
  prefixes.
- Added `KGMap.apply_inventory_update(...)` plus helper methods for:
  - authoritative inventory replacement,
  - add/remove deltas,
  - alias alignment such as `brass lantern` -> existing `lantern`.
- Applied gate inventory reconciliation immediately after the auxiliary gate and
  before situation, summary, and brainstorm modules.
- Added inventory reconciliation details to detailed run logs and
  `auxiliary_gate_log.txt`.
- Guarded environmental summaries with `command_outcome` so rejected/no-effect
  commands are not stored as successful environmental memories.
- Updated failed-command memory so repeated failures of the same command/location
  refresh stale observations/reasons instead of leaving the first failure as the
  only explanation.

## Design Notes

- There is no `inventory_uncertain` state in this version.
- Vague theft/loss text should not make the gate guess a concrete missing item.
- An explicit inventory listing can be authoritative.
- Concrete take/drop/eat/drink/give/loss evidence can produce delta updates.
- The system remains advisory and LLM-centered; it does not hard-block commands.

## Verification

Checked locally with:

```text
python -m compileall versions\lplh2_inventory_gate_patch_2026-06-28\lplh2
```

Also smoke-tested:

- gate prompt `.format(...)`,
- authoritative inventory reconciliation,
- delta inventory removal,
- alias alignment for carried items,
- duplicate failed-command refresh.

## Planned Design: Active Situation Plan

Date: 2026-06-29

This design is documented for the next LPLH2 improvement, but it is not
implemented in this snapshot.

### Problem

`SituationMemory` currently stores unresolved future-return situations, for
example:

```json
{
  "location": "Kitchen / upstairs",
  "situation": "dark upstairs area may require light"
}
```

This is useful passive memory, but it does not preserve an intention across
multiple steps. In the latest Zork1 experiment, the agent stored dark-area
situations, later acquired a lantern, and the affordance agenda correctly
connected the lantern to the stored dark situations. The agent turned on the
lantern and tried some chimney-related commands, but it did not reliably pursue
the stored target to completion and did not return to every earlier dark
location.

The failure has two parts:

1. Intention persistence: the agent may form the correct high-level idea, then
   forget it after one or two actions.
2. Navigation execution: reaching the target location may require several
   intermediate moves, and the main LLM currently re-derives each move from the
   map text every step.

### Proposed Solution

Add a singleton `Active Situation Plan` adjacent to situation memory. This
should be an advisory intention pointer, not a hard rule and not a command
queue.

Recommended fields:

```json
{
  "source_situation_key": "stable key for the stored situation",
  "target_location": "exact KG-map node to return to",
  "goal": "test whether the lantern helps with the dark upstairs area",
  "created_step": 88,
  "attempts_at_target": 0
}
```

Do not persist fixed `candidate_commands` in the plan. Commands should be
generated fresh by affordance brainstorming using the active plan as focus, so
failed-command memory and same-state tried memory can filter stale attempts.

### Intended Flow

1. Situation memory stores an unresolved problem.
2. A later event makes the situation plausibly actionable, such as inventory
   changing after taking a relevant item.
3. The situation update/resolution path creates or updates one active plan.
   The auxiliary gate should not synthesize the full plan because it already
   handles command outcome, inventory reconciliation, summary routing,
   situation routing, and affordance routing.
4. The main action prompt receives the active plan every step as soft guidance:
   continue the plan if plausible, but abandon it if a stronger immediate
   opportunity appears.
5. The affordance brainstormer receives the active plan goal as focus and
   proposes fresh commands when the target is relevant.
6. The plan is cleared when the source situation is resolved, when it becomes
   stale, or when repeated attempts at the target show no progress.

### Concrete Example

Stored situation:

```json
{
  "location": "Kitchen / upstairs",
  "situation": "dark upstairs area may require light"
}
```

Later observation:

```text
Command: take lantern
Observation: Taken.
Inventory: lantern
```

Active plan:

```json
{
  "source_situation_key": "kitchen_upstairs_dark_upstairs_area_may_require_light",
  "target_location": "Kitchen",
  "goal": "test whether the lantern helps with the dark upstairs area",
  "created_step": 88,
  "attempts_at_target": 0
}
```

The main LLM sees this intention while navigating. Once at the target, the
brainstormer can propose fresh commands such as `turn on lantern`, `up`, or
`climb stairs`, subject to current failed-command and same-state-tried context.

### Optional Route Hint

The active plan solves intention persistence, not navigation by itself. A future
patch may add an advisory KG-map route hint, such as:

```text
Target Kitchen appears reachable; a known route starts with east.
```

This should remain advisory map support, not forced command execution.
