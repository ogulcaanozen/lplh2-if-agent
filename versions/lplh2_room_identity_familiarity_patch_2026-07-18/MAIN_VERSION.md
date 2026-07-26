# Main LPLH2 Version

This folder is the current main LPLH2 thesis/testing baseline as of
2026-07-26:

```text
versions/lplh2_room_identity_familiarity_patch_2026-07-18
```

It was promoted after the completed 10-epoch Detective experiment
`detective_20260719_084125`.

- Per-epoch raw scores: `30, 30, 80, 90, 110, 180, 210, 190, 180, 170`
- Maximum raw score: `210`
- Final-three-epoch average: `180`

The designation records the best validated research baseline; it does not
claim that every mechanism is solved. Known limitations include imperfect
same-title room identity, familiarity labels that remained mostly `FRESH`,
and occasional LLM disregard of repetition or hazard advice.

See `CURRENT_STATE.md` for the included mechanisms and runtime configuration,
and `PATCH_NOTES.md` for the complete implementation history.
