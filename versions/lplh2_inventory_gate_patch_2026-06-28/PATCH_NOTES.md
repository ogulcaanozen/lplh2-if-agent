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
