# LPLH2 Attempt Ledger / Agenda Patch

Created from:

```text
versions/lplh2_kg_state_situation_patch_2026-07-03
```

This version keeps the KG/world-state reliability work and adds an LLM-centered
anti-loop layer. It does not hard-ban commands. Instead, it makes prior attempts
and agenda lifecycle visible to the brainstormer and main action LLM.

## Changes

1. Attempt ledger
   - Records every command by normalized room and command.
   - Tracks attempt count, last step, outcome class, last observation,
     destination if movement occurred, and whether outcomes varied.
   - Renders a compact "Command History In This Room" prompt block.

2. Per-location affordance agenda cache
   - Brainstormed affordance ideas are cached by location and compact state.
   - Pending ideas survive within the same local context so the main LLM can
     consider alternatives after the first suggestion fails.
   - The agenda annotates each proposed command with room-level tried counts and
     last outcomes.

3. Summary kind tags and retrieval headers
   - Stored experiences now include metadata such as `achievement`, `route`,
     `state_change`, `clue`, `syntax_lesson`, and `death_warning`.
   - Retrieved experiences are rendered with headers so score-gain memories are
     not treated as automatic repeat targets.

4. Main LLM repeat self-check
   - The action prompt asks for a `<repeat>` JSON field explaining whether the
     selected command is a repeat and why retrying is justified.
   - The parser ignores the `<repeat>` tag for command extraction, and logs the
     self-check in `action_generation_log.txt`.

## Intentionally Not Included

- Deterministic bans on repeated commands.
- Active situation planning or BFS navigation execution.
- Dynamic situation manager reintroduction.

## Post-Review Fixes

- Room visit counts now increment only on actual location changes; repeated
  commands inside the same room no longer inflate the visit count.
- Positive score summaries are always tagged as `achievement`, even if the game
  also ends on that step.
- Removed an unused experience-rendering config flag that was not wired to any
  behavior.
