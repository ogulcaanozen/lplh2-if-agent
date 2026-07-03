# LPLH2 KG State / Situation Patch

Created from the current main version:

```text
versions/lplh2_inv_gate_worldstate_selective_2026-07-02
```

This version is for testing KG-map improvements without reintroducing active
planning or BFS route forcing.

## Changes

1. LLM-gated non-cardinal transitions
   - KG now stores confirmed non-direction transitions separately from cardinal
     exits.
   - Code only detects the transition candidate. The auxiliary gate decides
     whether it is a reusable action transition before KG records it.
   - Example:

```json
{
  "action_transitions": {
    "Behind House": {
      "enter window": "Kitchen"
    },
    "Forest Path": {
      "climb tree": "Up a Tree"
    }
  }
}
```

2. Situation possible solution
   - Stored situations now optionally include:

```json
{
  "location": "Kitchen / upstairs",
  "situation": "dark upstairs area may require light",
  "possible_solution": "a light source may help explore upstairs safely"
}
```

   - This stays compact and advisory; there is still no active plan memory.

3. Qwen-14B object/world-state extraction
   - The auxiliary gate can route a dedicated object-state extraction pass.
   - This pass intentionally uses the main LLM fallback path, so in Colab it
     tests `Qwen/Qwen2.5-14B-Instruct` rather than o3-mini.
   - The KG can apply durable object states such as:

```json
{
  "object_state_updates": [
    {"object": "window", "location": "Behind House", "state": "open"},
    {"object": "trap door", "location": "Living Room", "state": "revealed, closed"}
  ],
  "new_objects": [
    {"object": "trap door", "location": "Living Room"}
  ]
}
```

   - Object-state extraction is not allowed to create new rooms. If the
     extractor names an unknown subarea or guessed location, KG stores the
     update under the current known room instead.

## Intentionally Not Included

- Later-room-title recovery after death/respawn text.
- Active situation planning.
- BFS navigation execution.
- Dynamic situation manager add/update/remove format.
