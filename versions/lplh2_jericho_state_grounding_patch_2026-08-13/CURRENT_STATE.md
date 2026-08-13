# Current State

Last updated: 2026-08-13

This folder adds optional Jericho state grounding:

```text
versions/lplh2_jericho_state_grounding_patch_2026-08-13
```

**Status: experimental child of the 2026-08-02 reward-resource snapshot.**

## Added In This Snapshot

- Jericho room object numbers anchor stable KG room identity when supported.
- Jericho inventory replaces inferred carried state after every live step,
  including objects contained inside carried containers.
- The grounded room and inventory reach the auxiliary gate before routing and
  then flow through the existing KG context to every memory, brainstorming,
  retrieval, and main-decision prompt.
- Unsupported or failed APIs fall back independently to the prior LLM-based
  location and inventory paths.
- Logs expose the grounding source and authoritative inventory reconciliation.

## Parent Snapshot

The remaining sections describe the inherited 2026-08-02 mechanisms.

This folder is the reward-resource, mixed-inventory, and situation-provenance
experiment:

```text
versions/lplh2_reward_resource_inventory_situation_patch_2026-08-02
```

**Status: experimental child of the latest purpose-aware snapshot.**

It is built on:

```text
versions/lplh2_purpose_death_situation_patch_2026-08-01
```

## Added In This Snapshot

- Existing reward context now highlights visible, uncarried resources named by
  exact setup commands for unearned rewards.
- Existing auxiliary and inventory prompts handle simultaneous acquisition and
  state changes without another runtime module or retry call.
- Persistent ordinary situations retain their original creation epoch and step
  for logs while gameplay prompts remain compact.
- The existing situation detector compares candidates against all learned
  situations, including current-epoch resolved records, before adding one.

## Inherited From The Parent Snapshot

- Object interactions are grounded from the selected agenda item, FM action
  split, or visible/inventory object names, so parser-rejected syntax still
  reaches the existing `InteractionStats` tracker.
- Object exhaustion uses distinct no-progress commands. Score, lasting state
  change, and genuinely new information protect productive puzzle objects.
- Brainstorm ideas may name `target_object` and `preparation_resource`.
  `preparation_for` receives automatic priority only when a grounded resource
  is actually acquired, equipped, carried, filled, loaded, lit, or activated.
  Invalid claims are demoted rather than removed.
- Agenda entries with equivalent pending-command sets merge before the
  five-entry cap, retaining their supporting situations.
- The main prompt prefers grounded untried agenda commands before invented
  commands for descriptive text, without forcing the final action.
- Logs record target provenance, object-futility state, preparation
  keep/demote decisions, agenda deduplication, selected agenda rank, and
  commands skipped earlier in the selected entry.

All purpose-aware acquisition, preparation validation, room-danger evidence,
transit, location identity, summary retrieval, goal-situation, KG, and command
memory behavior is inherited unchanged from the parent snapshot.

## Runtime

The root convenience notebook and this version's notebook point to this folder
and run Zork1 for `3` epochs x `250` steps with Qwen2.5-14B-Instruct.

Inspect `affordance_brainstorm_log.txt`, `action_generation_log.txt`,
`attempt_ledger_log.txt`, and `steplog.json` for the new behavior.
