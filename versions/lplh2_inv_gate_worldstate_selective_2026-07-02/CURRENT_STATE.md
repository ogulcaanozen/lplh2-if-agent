# Current State

Last updated: 2026-07-02

This folder is the selective LPLH2 version for the next Zork1 experiment.

It is based on the 45-point inventory-gate snapshot and adds only:

- KG/world-state reliability fixes.
- Dedicated inventory reconciliation.
- Newer logging and module timing.
- Action space restored/enabled.

It deliberately does not include:

- dynamic situation manager,
- active planning,
- BFS navigation.

## Runtime Defaults

The Colab notebook in this folder and the root convenience notebook both point
to:

```text
versions/lplh2_inv_gate_worldstate_selective_2026-07-02
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
