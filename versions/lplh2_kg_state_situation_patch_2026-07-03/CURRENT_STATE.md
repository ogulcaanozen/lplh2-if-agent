# Current State

Last updated: 2026-07-03

This folder is an experimental KG-map/state patch built on top of the current
main version:

```text
versions/lplh2_inv_gate_worldstate_selective_2026-07-02
```

It keeps the main version's selective design and adds:

- LLM-gated non-cardinal action transitions in KG,
- `possible_solution` on stored situations,
- gated Qwen-14B object/world-state extraction.
- object-state location clamping so the extractor cannot create fake rooms.

It deliberately does not include:

- dynamic situation manager,
- active planning,
- BFS navigation.

## Runtime Defaults

The Colab notebook in this folder and the root convenience notebook should
point to:

```text
versions/lplh2_kg_state_situation_patch_2026-07-03
```

The notebook is configured for:

- `1` epoch,
- `250` max steps,
- main LLM `Qwen/Qwen2.5-14B-Instruct`,
- auxiliary modules through local LLM fallback unless OpenAI model env vars are
  explicitly set,
- FM model `Qwen/Qwen2.5-1.5B-Instruct` plus the Drive LoRA adapter.

## Logs To Inspect

Each experiment folder should include:

- `run_log.txt`
- `summary_module_log.txt`
- `situation_memory_log.txt`
- `affordance_brainstorm_log.txt`
- `action_failure_memory_log.txt`
- `action_generation_log.txt`
- `auxiliary_gate_log.txt`
- `kg_location_log.txt`
- `module_timing_log.txt`
- JSON step log

## External Files

Keep outside GitHub, usually under Google Drive:

```text
MyDrive/lplh/
  fm_adapter_v4_autoplay_failures/
  games/
    zork1.z5
```

Generated logs, Chroma DBs, experience indexes, ROMs, adapters, and API keys are
not part of the repo.
