# LPLH2 IF Agent

Experimental LPLH2 framework for text-based interactive fiction agents.

This snapshot is `lplh2_kg_state_situation_patch_2026-07-03`: it starts from
the current main selective world-state version and adds a focused KG-map test
patch.

New in this version:

- LLM-gated non-cardinal KG action transitions such as
  `enter window -> Kitchen`,
- compact `possible_solution` text on stored situations,
- gated Qwen-14B object/world-state extraction for durable states such as
  `window=open`, `rug=moved`, and `trap door=revealed, closed`.
  The extractor cannot create new rooms; unknown location strings are clamped
  to the current known room.

It intentionally does not include later-room-title recovery, the dynamic
situation manager, active planning, or BFS navigation experiments.

This repo contains the `lplh2` package and the Colab experiment notebook used for the Zork1 runs. It intentionally does **not** include game ROMs, Chroma databases, run logs, Drive data, API keys, or LoRA adapter weights.

## Contents

- `lplh2/` - agent, KG map, action space, experience library, stored-situation memory, affordance brainstorming, failed-action memory, and prompts.
- `lplh2/run_zork1_lplh2_smoke_colab.ipynb` - Colab experiment notebook.
- `requirements.txt` - Python dependencies used by local/Colab runs.

## External Files Needed

Keep these outside GitHub, usually in Google Drive:

- Z-machine ROMs such as `zork1.z5`.
- FM LoRA adapter, currently expected in Drive as `fm_adapter_v4_autoplay_failures/` by the notebook.
- Experiment data/logs under `data/`.
- `OPENAI_API_KEY` is optional for the current experiment. `LPLH_LLM_ES_MODEL` and `LPLH_BRAINSTORM_MODEL` are left empty so the auxiliary modules, including affordance brainstorming, use the local main LLM (`Qwen/Qwen2.5-14B-Instruct`).

## Typical Colab Layout

```text
MyDrive/lplh/
  IFGames/
    lplh2/
  fm_adapter_v4_autoplay_failures/
  games/
    zork1.z5
```

The notebook can also accept ROMs under `IFGames/games/` depending on the configured candidate paths.

## Main Runtime Roles

- Main action LLM: usually `Qwen/Qwen2.5-14B-Instruct` in Colab.
- FM model: `Qwen/Qwen2.5-1.5B-Instruct` plus the LoRA adapter for validation, relation extraction, and action splitting.
- Auxiliary LLM: local `Qwen/Qwen2.5-14B-Instruct` fallback for summaries, situation memory, environmental-change detection, failed-command reasoning, and repetition evaluation.
- Affordance brainstorming: local `Qwen/Qwen2.5-14B-Instruct` fallback via empty `LPLH_BRAINSTORM_MODEL`.

## Notes

The repo is public-safe by design: it excludes nonredistributable ROMs, generated logs/results, Chroma indexes, and model adapter artifacts.
