# LPLH2 Location Identity Refinement Patch

Experimental LPLH2 framework for text-based interactive fiction agents.

This snapshot is `lplh2_location_refinement_patch_2026-07-14`. It is an
experimental child of `lplh2_llm_location_identity_patch_2026-07-13`; the
repository's current main version remains unchanged.

New or corrected in this version:

- exact registered description matches bypass repeated room-identity calls;
- registry dedup prevents erroneous resolver splits from becoming permanent;
- immutable mint descriptions keep same-title candidate cards stable;
- unresolved dark locations hide the stale room's objects and exits;
- main and brainstorming prompts treat inventory as authoritative possession;
- respawn titles printed after a death marker cannot become hazard rooms;
- room movement/title verdicts come from grounded auxiliary-LLM output;
- a second LLM decision disambiguates same-title arrivals from textual evidence;
- FM triples cannot move the strict agent KG or create free-text destinations;
- text-derived registry ids stabilize room/event identity across epochs;
- contradictory traversal evidence can split a previously merged room node;
- engine room ids are logged only for offline map evaluation;
- every death updates a persistent physical-room death count, so two deaths in
  one room can trigger a goal even when their final commands differ;
- compact per-room evidence preserves distinct fatal actions and gateways until
  the room-level goal is created, then merges later deaths into that goal;
- movement-entry deaths use the destination as the hazard and the issuing
  room/action as the gateway;
- precondition hypotheses include concrete `item_keywords`, allowing conceptual
  requirements such as protection to match literal inventory names;
- alive exits only confirm a goal when entry inventory matches those keywords;
  unmatched retreats remain logged while the goal stays active;
- deaths after confirmation reopen the goal and retain the previous confirmation
  in `false_confirmations`;
- one retrieved death warning summarizes sibling fatal actions and the room-level
  death count without changing episodic storage or event keys;

- `SituationMemory` has a second `goal` class for persistent preparation
  requirements inferred from repeated deaths in one physical room;
- the first death still creates the normal place-bound warning, while the
  second room-level death can trigger one precondition-hypothesis LLM call;
- open goals are advisory and visible through the existing situations channel
  everywhere until confirmed, declined, or downgraded to avoid;
- goals merge by hazard-room fingerprint, retain gateway/fatal-action evidence,
  survive normal epoch resets, and are capped at five open entries;
- leaving a goal room alive confirms the goal and removes it from the active
  feed only when the entry inventory matches the hypothesized item keywords;
- an unprepared retreat leaves the goal open, while a later contradictory
  death reopens a previously confirmed goal and revises its hypothesis;
- fatal movement events key the goal to the destination hazard and preserve
  the issuing room/action as the gateway;
- goal transitions and hypothesis prompt/response records are written to the
  existing situation and summary logs;

- filtered summary retrieval returns an empty result when a query succeeds but
  finds no matches, instead of falling back to unrelated recent memories;
- experience retrieval is a true cap of up to `5` useful summaries, not a forced
  quota;
- reward enabler summaries are stored from recent state-changing setup actions
  before a score gain, linked to the same score event key as the achievement;
- enabler summaries are hidden for the current epoch once the linked reward has
  been earned;
- completed pending agenda commands are consumed when the ledger shows they
  already caused a state change, movement, or score;
- brainstorm carryover relevance uses the current observation text plus visible
  objects, instead of letting inventory keep stale acquisition ideas alive;
- fresh brainstorm ideas outrank carried ideas when merged;
- confirmed blocked directions are tracked in the KG and trigger one action
  regeneration if the main LLM selects the same blocked direction again;
- the initial room receives a description fingerprint, and empty same-title
  fingerprints are adopted instead of creating spurious `#2` rooms.

Still not included in this version:

- active planning/BFS execution;
- a new separate agenda module;
- parser noun canonicalization as a dedicated fix.

The repo excludes game ROMs, Chroma databases, logs, Drive data, API keys, and
LoRA adapter weights.
