# Current Main LPLH2 Version

Current main version as of 2026-07-11:

```text
versions/lplh2_enabler_agenda_retrieval_patch_2026-07-07
```

This is the thesis/testing baseline and the version selected by the root Colab
experiment notebook. Older folders under `versions/` are preserved as
historical snapshots and should not be modified when changing the main version.

The main-version designation includes all patches through commit `e271db5`:

- grounded achievement, death, and reward-enabler memories;
- sparse, dynamic experience retrieval with exact physical-room relevance;
- current-epoch suppression of already-earned achievements and linked enablers;
- affordance brainstorming with a per-location carryover agenda;
- failed-command memory, attempt ledger, and same-state repetition memory;
- dedicated inventory reconciliation and KG/world-state updates;
- visit-scoped repeated-navigation enforcement;
- preserved `<< Room Title >>` identities and fingerprint-aware event keys;
- terminal defeat/victory detection and deterministic start-memory retrieval;
- detailed summary, retrieval, gate, action, KG, agenda, and timing logs.

The main version intentionally does not include active planning or forced BFS
navigation.
