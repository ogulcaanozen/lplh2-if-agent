# Current State

Last updated: 2026-06-29

This file is a handoff note for continuing the LPLH2 research project from another computer or a fresh Codex session.

## Repository Scope

This public repo contains the portable LPLH2 code and Colab notebook:

- `lplh2/` - the LPLH2 agent package.
- `lplh2/run_zork1_lplh2_smoke_colab.ipynb` - current Colab experiment notebook.
- `requirements.txt` - dependency list.

This folder is the `lplh2_inventory_gate_patch_2026-06-28` snapshot. It is based
on the auxiliary-gate version, with inventory reconciliation added to the gate.

This repo intentionally excludes:

- Z-machine ROMs such as `zork1.z5`.
- Google Drive experiment data.
- Chroma DBs and experience indexes.
- Run logs and step logs.
- FM LoRA adapter weights.
- API keys and local `.env` files.

## External Files Needed

For Colab runs, keep these in Google Drive:

```text
MyDrive/lplh/
  fm_adapter_v4_autoplay_failures/
  games/
    zork1.z5
```

The notebook can also use a project checkout under:

```text
MyDrive/lplh/IFGames/
```

`OPENAI_API_KEY` is optional in the current experiment. `LPLH_LLM_ES_MODEL`
is intentionally empty, so auxiliary modules use the local main LLM
(`Qwen/Qwen2.5-14B-Instruct`) instead of `o3-mini`.

## Current Runtime Design

Main LPLH2 modules:

- `KGMap`: tracks rooms, exits, visible objects, inventory, and simple relation triples.
- `ActionSpace`: tracks learned valid verb/object templates.
- `ExperienceLib`: stores score-change and neutral-event summaries in Chroma.
- `SituationMemory`: stores unresolved future-return situations such as darkness, locked access, or missing-condition blockers.
- `AffordanceBrainstormer`: proposes object/inventory/stored-situation commands.
- `FailedActionMemory`: stores invalid failed commands by location.
- `StateScopedActionMemory`: stores commands that were unproductive in the exact pre-action state.
- `FmClient`: fine-tuned Qwen2.5-1.5B + LoRA for action validation, relation extraction, and action splitting.
- `LLMClient`: main action LLM plus optional OpenAI auxiliary LLM calls. In
  this snapshot, auxiliary calls default to local LLM_a/Qwen14B.

Typical Colab experiment setup:

- Main action LLM: `Qwen/Qwen2.5-14B-Instruct`.
- FM model: `Qwen/Qwen2.5-1.5B-Instruct` plus `fm_adapter_v4_autoplay_failures/`.
- Auxiliary LLM: local `Qwen/Qwen2.5-14B-Instruct` fallback for summaries,
  situation memory, environmental-change detection, affordance brainstorming,
  failure explanation, and repetition evaluation. Set `LPLH_LLM_ES_MODEL`
  only to deliberately re-enable an OpenAI auxiliary model such as `o3-mini`.

## Recent Fixes

Important LPLH2 fixes already applied:

- Robust command parsing for malformed main-LLM tags.
- Initial KG seed from the initial game observation.
- Separate logs per experiment folder.
- Summary module log with state type, prompt, and summary.
- Situation memory module and log.
- Affordance brainstorming module and log.
- Action failure memory module and log.
- Main action-generation log with raw response and extracted reasoning.
- Environmental-change detection moved to an auxiliary LLM decision.
- Same-state repetition memory added:
  - stores the pre-action state snapshot,
  - evaluates no-progress actions with an LLM,
  - shows same-state tried commands to the main action LLM,
  - filters exact same-state repeats out of brainstorm suggestions,
  - does not hard-block the main LLM.
- Inventory reconciliation moved into the auxiliary gate:
  - the gate emits `command_outcome` and structured `inventory_update`,
  - the old hard-coded `take/drop/eat/drink/give` command-prefix inventory
    mutation has been removed from the agent,
  - explicit inventory listings can authoritatively replace carried items,
  - concrete take/drop/loss evidence can add or remove carried items,
  - inventory reconciliation is logged in detailed run logs and
    `auxiliary_gate_log.txt`.
- Condition-aware affordance trigger added:
  - the auxiliary gate receives recent same-location command outcomes,
  - the gate can trigger fresh brainstorming when several different commands
    produce similarly abnormal, garbled, repeated, obscured, blocked, or
    mismatched observations,
  - affordance brainstorming can emit optional `kind: "condition"` ideas,
  - the action prompt treats condition ideas as advisory context, not forced
    commands.
- Auxiliary gate logging now writes `auxiliary_gate_log.txt` with:
  - step/action/location metadata,
  - observation,
  - prompt,
  - raw LLM response,
  - parsed response body,
  - normalized decision,
  - inventory reconciliation,
  - environmental-change detail,
  - recent failures, same-state tried commands, active situations, and recent
    command outcomes.

## Current Known Problem

The agent still tends to repeat object commands or low-value interactions in Zork1, especially around objects like:

- `leaflet`
- `tree`
- `egg`
- `canary`

The latest same-state repetition and condition-aware brainstorming patches
should reduce this, but they still need validation in a fresh experiment.

## Next Planned Patch

The next design idea is an `Active Situation Plan`, documented in
`PATCH_NOTES.md`. Situation memory is currently passive: it remembers unresolved
future-return problems, but the agent can forget a multi-step intention after
finding a possible solution item. The planned design is a singleton advisory
plan such as:

```json
{
  "source_situation_key": "kitchen_upstairs_dark_upstairs_area_may_require_light",
  "target_location": "Kitchen",
  "goal": "test whether the lantern helps with the dark upstairs area",
  "created_step": 88,
  "attempts_at_target": 0
}
```

This plan should persist intention across navigation steps, feed the
brainstormer as focus, and remain advisory. It should not store fixed command
lists because those can go stale and create loops. Opus also noted that this
fixes intention persistence, not navigation execution; a later advisory KG-map
route hint may be useful.

## Research Preference

For LPLH2, avoid hard-coded gameplay rules and forced command vetoes. Prefer:

- LLM-based evaluation,
- structured memory,
- compact high-signal context,
- advisory prompts,
- and state-aware caches.

The user wants the framework to remain LLM-centered rather than becoming a rule-based Zork solver.

## Suggested Future Experiments

Useful next experiments:

- 1 epoch x 250 steps on Zork1 with the latest inventory-gate snapshot.
- Compare run logs before/after condition-aware brainstorming.
- Inspect:
  - `auxiliary_gate_log.txt`
  - `action_generation_log.txt`
  - `affordance_brainstorm_log.txt`
  - `action_failure_memory_log.txt`
  - JSON step log
- Then patch selective brainstorming / pending alternatives.

Possible model comparison:

- Keep the same LPLH2 modules.
- Run Qwen2.5-14B as the main LLM.
- Later run GPT-5-class, Claude Sonnet-class, or Gemini-class model as an upper-bound main LLM.
- If a frontier model succeeds with the same modules, Qwen14B is likely the bottleneck.
- If a frontier model still loops, the memory/context design is likely the bottleneck.

## Git Workflow

From another computer:

```bash
git clone https://github.com/ogulcaanozen/lplh2-if-agent.git
cd lplh2-if-agent
```

After edits:

```bash
git add .
git commit -m "Describe patch"
git push
```

In Colab, pull/reclone the latest repo before running experiments.
