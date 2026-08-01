# Current State

Last updated: 2026-08-01

This folder is the purpose-aware acquisition, room-danger, and persistent
situation experiment:

```text
versions/lplh2_purpose_death_situation_patch_2026-08-01
```

**Status: experimental child of the current agenda-grounding snapshot.**

It is built on:

```text
versions/lplh2_agenda_grounding_patch_2026-07-31
```

## Added In This Snapshot

- Main and brainstorming prompts distinguish purposeful resource acquisition
  from collecting every visible object, including explicit capacity-pressure
  guidance.
- `preparation_for` priority now requires an LLM-written
  `preparation_relation` that connects the grounded resource to a separate
  situation. Code performs structural grounding checks and logs keep/demote
  decisions.
- Persistent death evidence is rendered as room-level danger history for both
  reasoning models, without turning the room into a deterministic action ban.
- Compact three-field situation records persist across epochs. Resolution
  hides a record for the current epoch, and the next epoch reactivates it.
- Action and situation logs expose acquisition purpose, preparation relation,
  room danger history, persistent records, and current-epoch resolution state.

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

All transit, location-identity, world-state extraction, summary retrieval,
goal situation, reward directory, KG, command-memory, and inventory behavior
is inherited unchanged from the parent snapshot.

## Runtime

The root convenience notebook and this version's notebook point to this folder
and run Zork1 for `3` epochs x `250` steps with Qwen2.5-14B-Instruct.

Inspect `affordance_brainstorm_log.txt`, `action_generation_log.txt`,
`attempt_ledger_log.txt`, and `steplog.json` for the new behavior.
