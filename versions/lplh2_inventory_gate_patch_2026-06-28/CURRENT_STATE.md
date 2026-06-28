# Current State

Last updated: 2026-06-28

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

Set `OPENAI_API_KEY` in Colab Secrets if using the `o3-mini` auxiliary modules.

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
- `LLMClient`: main action LLM plus optional OpenAI auxiliary LLM calls.

Typical Colab experiment setup:

- Main action LLM: `Qwen/Qwen2.5-14B-Instruct`.
- FM model: `Qwen/Qwen2.5-1.5B-Instruct` plus `fm_adapter_v4_autoplay_failures/`.
- Auxiliary LLM: `o3-mini` for summaries, situation memory, environmental-change detection, affordance brainstorming, failure explanation, and repetition evaluation.

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

## Current Known Problem

The agent still tends to repeat object commands or low-value interactions in Zork1, especially around objects like:

- `leaflet`
- `tree`
- `egg`
- `canary`

The latest same-state repetition patch should reduce this, but it has not yet been validated in a full experiment.

## Next Planned Patch

The next design idea is to optimize and stabilize affordance brainstorming:

1. Do not call the brainstormer every step.
2. Run brainstorming only when there is likely affordance material:
   - visible objects/features,
   - inventory items,
   - active stored situations,
   - object/syntax failures needing alternatives,
   - or a meaningfully changed state.
3. Cache brainstorm ideas by exact state.
4. Track pending vs tried brainstorm commands.
5. Show the main LLM:

```text
Pending brainstormed alternatives:
- move leaves
- look under leaves

Already tried in this same state:
- take leaves -> The leaves are of no use.
```

The main LLM should still decide freely. The system should provide better context, not force or block actions.

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

- 1 epoch x 150 steps after same-state repetition memory patch.
- Compare run logs before/after repetition memory.
- Inspect:
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
