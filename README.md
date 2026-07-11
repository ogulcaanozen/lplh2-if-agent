# LPLH2 IF Agent

Experimental LPLH2 framework for text-based interactive fiction agents.

This repo contains the `lplh2` package and Colab experiment notebooks used for
interactive-fiction runs. It intentionally does **not** include game ROMs,
Chroma databases, run logs, Drive data, API keys, or LoRA adapter weights.

## Current Main Version

The current thesis/testing baseline is:

```text
versions/lplh2_enabler_agenda_retrieval_patch_2026-07-07
```

See `MAIN_VERSION.md` for the authoritative designation and included patch
level. The root Colab notebook selects this version from GitHub.

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
