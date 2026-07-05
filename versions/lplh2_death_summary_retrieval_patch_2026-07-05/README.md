# LPLH2 IF Agent

Experimental LPLH2 framework for text-based interactive fiction agents.

This snapshot is `lplh2_death_summary_retrieval_patch_2026-07-05`: it starts
from the attempt-ledger / agenda version and adds cleaner cross-epoch death
memory plus less repetitive experience retrieval.

New in this version:

- factual attempt ledger rendered as "Command History In This Room",
- per-location affordance agenda cache with tried-count annotations,
- summary kind tags and retrieval headers,
- main LLM repeat self-check in `<repeat>...</repeat>`,
- evidence-first score-loss/death summaries with local attempt-ledger context,
- epoch/step staleness headers for retrieved experiences,
- score-gain summary dedup across epochs via the Chroma-independent event index,
- retrieval over-fetching with a small diversity selector,
- KG/world-state reliability fixes from the July 3 snapshot.

It intentionally does not hard-ban repeated commands and does not include the
dynamic situation manager, active planning, or BFS navigation experiments.

This repo contains the `lplh2` package and the Colab experiment notebook used for the Zork1 runs. It intentionally does **not** include game ROMs, Chroma databases, run logs, Drive data, API keys, or LoRA adapter weights.

## Contents

- `lplh2/` - agent, KG map, action space, experience library, stored-situation memory, affordance brainstorming, attempt ledger, failed-action memory, and prompts.
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
