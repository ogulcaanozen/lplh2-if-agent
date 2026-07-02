# LPLH2 Selective Worldstate Patch

Date: 2026-07-02

This snapshot starts from the higher-scoring inventory-gate baseline
(`lplh2_inventory_gate_patch_2026-06-28`) and selectively ports only the
mechanics we still want to test.

## Included

- KG/world-state reliability fixes:
  - cleaner current-room JSON,
  - canonical room aliases,
  - real destination names in confirmed exits,
  - room-title fallback/override when FM misses or hallucinates the location,
  - KG location audit log.
- Dedicated inventory reconciliation:
  - the auxiliary gate routes whether reconciliation should run,
  - a separate inventory LLM prompt performs add/remove/authoritative updates,
  - mixed observations such as "revealed a grating. Taken." are handled by the
    inventory reconciler instead of hard-coded command-prefix rules.
- Newest logging/timing:
  - auxiliary gate log,
  - KG location log,
  - module timing log,
  - action generation log,
  - inventory reconciliation details.
- Action space enabled from the start:
  - valid FM-split commands are stored,
  - learned verb/object templates are shown in run logs,
  - the main action prompt and affordance brainstormer receive action-space
    context.

## Intentionally Not Included

- Full dynamic situation manager.
- Active situation planning.
- BFS navigation or forced route execution.

The older passive situation detector/resolver remains, because it existed in
the selected baseline and is less invasive than the dynamic manager.

## Verification

Checked locally with:

```text
python -m compileall versions\lplh2_inv_gate_worldstate_selective_2026-07-02\lplh2
```

Also smoke-checked:

- prompt `.format(...)` for action generation, auxiliary gate, inventory
  reconciliation, and affordance brainstorming,
- KG confirmed route storage,
- inventory add/update via `KGMap.apply_inventory_update(...)`.
