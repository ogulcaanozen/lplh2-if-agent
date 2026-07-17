# LPLH2 Reward Directory and Prompt Slimming Patch

## 2026-07-17: persistent reward targets and compact decision context

Created from
`versions/lplh2_memory_grounding_visitcount_patch_2026-07-16` without modifying
that baseline.

This snapshot adds a persistent reward directory populated only from grounded
score summaries and observed movement. Each score event retains its exact
scoring command, same-room setup commands, a cycle-compressed route hint, and
current-epoch earned status. Grounded setup commands also revive reward-enabler
storage when the attempt ledger classified a useful setup as information.

Cross-epoch interaction statistics produce advisory notes for visible objects
that have accumulated many commands without score or lasting state change.
Route lessons are deduplicated during storage and retrieval.

The action and brainstorm prompts share one compact `TRIED HERE` context.
Brainstorming sees only relevant achievements, enablers, and death warnings,
plus the reward directory and object-history notes. The KG prompt keeps current
room exits but replaces the full per-room frontier with room labels that still
have unexplored exits. Main action history is capped at six turns.

## 2026-07-16: grounded room variants, preparation, and loop advisory

Created from `versions/lplh2_location_refinement_patch_2026-07-14` without
modifying that baseline. Persistent room entries now retain up to six
resolver-confirmed description signatures and provide candidate cards to fresh
epochs, so portable-object changes and transient ambience do not repeatedly
split one physical room.

Unseen-area situations and death goals are grounded to the last known room and
the command that entered the area. Death summarization keeps repeated unseen
observations in addition to the final turns, and a grounded
`wrong_in_that_state` lesson may create a concrete preparation goal after the
first death.

The affordance agenda supports explicit `preparation_for` ideas, consumes
equivalent command phrasings, and keeps relevant preparation alive while its
situation remains active. Epoch-local visit counts, destination annotations,
and an eight-step recent path give the main LLM advisory evidence about stale
rooms and two-room oscillation without banning navigation.

# Earlier: LPLH2 Location Identity Refinement Patch

## 2026-07-14: stable signatures and uncertainty-safe context

Created from `versions/lplh2_llm_location_identity_patch_2026-07-13` while
leaving the main marker unchanged. Exact registry description matches now reuse
stable room identities before the resolver LLM, ambiguous identical matches use
known topology or visit recency, and resolver-created rooms retain registry
deduplication. Mint descriptions are immutable; current-visit descriptions are
stored separately and seed contradiction-split siblings.

While the destination is unseen, prompt-facing room state keeps inventory but
hides all stale local objects and exits, records the entering action, and shows
a likely inverse route. Action and brainstorm prompts now make inventory the
authoritative possession record. Grounded death titles appearing only after a
three-asterisk death/respawn marker are rejected.

## 2026-07-13: text-grounded KG room identity

Created from `versions/lplh2_goal_situation_roomlevel_patch_2026-07-12`.
The main marker remains unchanged. The auxiliary gate now emits a grounded
movement/title verdict, same-title arrivals are resolved by the auxiliary LLM,
and a persistent text-derived registry stabilizes room labels across epochs.
FM location triples are inert in the strict agent KG, confirmed edges come only
from resolved movement, and contradictory edges can split a merged room belief.

The runner performs state-preserving `look` probes and records Jericho room ids
only in the steplog for offline evaluation. They are never put in agent `info`,
prompts, memories, event keys, or the registry. Terminal room titles are copied
by the death summarizer and substring-validated before hazard memories use them.

Post-review corrections keep movement-death warnings retrievable from the room
where the fatal command is issued while storing the grounded hazard room in
separate destination fields. Navigation facts are suppressed for the first
visible step after an unknown dark location because the remembered source room
is stale. Legacy fallback mode again uses the baseline subject resolver and
gate-approved transition repair. Canonical decoration stripping and registry
naming remain intentionally always-on bookkeeping normalization, so fallback
labels can differ cosmetically from the baseline even though its movement and
triple semantics are restored.

## 2026-07-12: room-level lifecycle and merged warnings

Created from:

```text
versions/lplh2_goal_situation_patch_2026-07-11
```

The main version marker is intentionally unchanged. This experimental snapshot
counts every death by stable hazard-room identity, retains compact room-level
fatal evidence, and runs at most one precondition-hypothesis call on a death.
Goals are created after the second room death even when fatal commands differ.

Hypotheses now provide concrete inventory `item_keywords`. Alive retreats only
confirm a goal when entry inventory matches those anchors; other survivals are
recorded and keep the warning active. A later contradictory death reopens a
confirmed goal while preserving the old inventory in `false_confirmations`.

Retrieval still selects one death-warning slot, but its header now merges up to
five sibling fatal actions and shows the persistent room death count. Episodic
event keys, Chroma storage, exact-room selection, and command generation remain
unchanged.

## 2026-07-11: persistent preparation goals

Created from:

```text
versions/lplh2_enabler_agenda_retrieval_patch_2026-07-07
```

This patch extends the existing `SituationMemory` rather than adding a new
module. Observation situations remain epoch-local. Repeated identical deaths
can create a bounded, persistent, advisory preparation goal that is available
non-locally through the existing situations context. Goal records merge by
hazard-room identity and support refutation, confirmation, decline, and avoid
lifecycle states.

The patch does not change Chroma retrieval, event keys, or command selection,
and it adds no game-specific rules.

Confirmation requires the inventory captured on goal-room entry to match the
current hypothesized requirement. Leaving alive without that preparation is
treated as a retreat, not proof that the hazard was solved. A later death
reopens any mistakenly confirmed goal. For fatal movement, the destination is
the hazard room and the source room/action is stored as its gateway.

## Earlier enabler / agenda notes

Created from:

```text
versions/lplh2_grounded_memory_retrieval_patch_2026-07-07
```

## Purpose

The previous run showed that summary grounding improved, but the agent still
missed setup actions such as opening a window before climbing through it or
keeping rug/trap-door ideas alive in the Living Room. This patch keeps the
current module layout and improves memory retrieval, agenda lifecycle, and KG
blocked-exit/fingerprint behavior without adding a new planner.

## Changes

1. Sparse retrieval
   - `retrieve_relevant_structured` returns `[]` when Chroma succeeds but a
     filtered query has no matches.
   - `EXPERIENCE_TOP_K` is now `5`.
   - The selector no longer pads weak non-route memories just to fill all slots.

2. Reward enablers
   - Positive score gains now store up to two recent `state_change` setup
     commands from a short lookback window.
   - Enablers are templated from harness facts, not generated by another LLM
     call.
   - Each enabler stores `enables_event_key`, linking it to the achievement
     score event.
   - Enablers are selected before achievements when the linked reward has not
     been earned in the current epoch.

3. Agenda lifecycle
   - The brainstorm agenda consumes commands whose ledger outcome already shows
     `state_change`, `moved`, or `scored`.
   - Completed commands are not kept as pending.
   - The brainstorm prompt tells the LLM not to propose acquiring objects
     already listed in inventory.

4. Carryover relevance
   - Stale carryover relevance uses visible objects plus current observation
     text.
   - Inventory alone no longer keeps stale acquisition ideas alive.
   - Fresh brainstorm ideas are merged before carried ideas.

5. Blocked exits
   - KG nodes track `blocked_directions`.
   - Prompt-facing map JSON includes `blocked_exits` for the current room.
   - If the main LLM selects a confirmed blocked direction, action generation
     regenerates once with a correction. If the retry repeats the blocked
     direction, it falls back to `look`.

6. Initial-room fingerprint
   - Initial KG seeding now records a room fingerprint from the opening
     observation.
   - Same-title candidates with empty fingerprints adopt a later fingerprint
     instead of creating spurious `#2` rooms.

## Post-review hotfixes

- Affordance exact-state keys drop the observation field before serialization,
  so observation text can help carryover relevance without fragmenting
  valid-but-unproductive command memory.
- Initial-room fingerprint seeding slices the initial observation from the room
  title before fingerprinting, preventing banner/copyright text from becoming
  the room fingerprint.
- Filtered retrieval returns `[]` even when a filtered Chroma query raises an
  exception, instead of falling back to unrelated recent memories.
- Current-epoch reward completion is also tracked by `(location, reward)` so
  achievements/enablers are suppressed when the same reward is re-earned with a
  wording variant of the original scoring command.

## Validation

Run from this version folder:

```text
python -m compileall lplh2
```

Also run the included notebook for behavioral validation. Inspect:

- `retrieved_summaries_log.txt` for enabler/achievement selection;
- `summary_module_log.txt` for stored `score_enabler` entries;
- `affordance_brainstorm_log.txt` for completed-command consumption;
- `action_generation_log.txt` for `blocked_direction_guard`;
- `kg_location_log.txt` for fingerprint/location behavior.
