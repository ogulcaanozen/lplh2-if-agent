"""LPLH2 Agent - Enhanced zero-shot decision-making.

The main agent that ties all three modules together:
1. Dynamic KG-Map (spatial reasoning)
2. Action Space Learning (verb-object discovery)
3. Experience Library (reflective learning via RAG)

At each step, it integrates all module outputs with the current
observation to generate the next game command via zero-shot prompting.

LPLH2 keeps the paper-faithful LPLH runtime fixes from ``lplh`` and adds
neutral-state experience storage for useful non-score events.
"""

import re
import logging
import hashlib
import json
import time
import copy
from .kg_map import KGMap, canonical_room_display
from .action_space import ActionSpace
from .experience_lib import ExperienceLib
from .opportunity_module import SituationMemory
from .affordance_brainstormer import AffordanceBrainstormer
from .action_memory import FailedActionMemory, StateScopedActionMemory
from .attempt_ledger import AttemptLedger
from .command_keys import normalize_command_key, normalize_location_key
from .llm_client import LLMClient
from .fm_client import FmClient
from .prompts import LPLH_ACTION_GENERATION_PROMPT
from . import config

logger = logging.getLogger(__name__)


class LPLHAgent:
    """LPLH Agent for playing Interactive Fiction games.

    Implements the full LPLH pipeline from the paper (Section 3.5):
    1. Update KG-map with extracted relations
    2. Validate & store action in action space
    3. Summarize experience on score change
    4. Retrieve relevant experiences
    5. Generate next command via zero-shot LLM
    """

    def __init__(self, llm_client: LLMClient = None, fm_client: FmClient = None):
        # LLM_a (action generation + experience summarization fallback)
        self.llm = llm_client or LLMClient()
        # fm (3 paper-faithful structured tasks: validate / extract / split)
        self.fm = fm_client or FmClient()
        self.kg_map = KGMap(strict_location_authority=True)
        self.action_space = ActionSpace()
        self.experience_lib = ExperienceLib()
        self.situation_memory = SituationMemory()
        self.affordance_brainstormer = AffordanceBrainstormer()
        self.failed_action_memory = FailedActionMemory()
        self.state_action_memory = StateScopedActionMemory()
        self.attempt_ledger = AttemptLedger()

        # History tracking
        self.history = []          # list of (action, observation) tuples
        self.prev_action = None
        self.prev_score = 0
        self.total_score = 0
        self.step_count = 0
        self.current_epoch = 1

        # Neutral-state tracking
        self.consecutive_failures = 0
        self.recent_failed_actions = []

        # Per-step detail log for tracking
        self.step_details = []
        self.pending_generation = None
        self.initial_observation_processed = False
        self.last_retrieval_debug = {}
        self.earned_score_event_keys_this_epoch = set()
        self.earned_score_location_reward_keys_this_epoch = set()
        self._recent_outcomes = []
        self._visit_direction_failures = {}
        self._goal_visit_inventory = {}
        self._hazard_room_deaths = {}
        self._hazard_room_evidence = {}
        self._location_resolver_cache = {}
        self._contradiction_splits_this_epoch = set()

    def reset(self, keep_experiences: bool = True):
        """Reset the agent for a new epoch.

        Args:
            keep_experiences: If True, keep the Experience Library and learned
                            preparation goals across epochs. All epoch-local
                            world/action bookkeeping and observation situations
                            are reset. If False, clear both persistent stores.
        """
        self.kg_map.reset(full=not keep_experiences)
        self.action_space.reset()
        self.situation_memory.reset(full=not keep_experiences)
        self.failed_action_memory.reset()
        self.state_action_memory.reset()
        self.attempt_ledger.reset()
        if not keep_experiences:
            self.experience_lib.reset()
        self.affordance_brainstormer.reset()
        self.history = []
        self.prev_action = None
        self.prev_score = 0
        self.total_score = 0
        self.step_count = 0
        self.consecutive_failures = 0
        self.recent_failed_actions = []
        self.step_details = []
        self.pending_generation = None
        self.initial_observation_processed = False
        self.last_retrieval_debug = {}
        self.earned_score_event_keys_this_epoch = set()
        self.earned_score_location_reward_keys_this_epoch = set()
        self._recent_outcomes = []
        self._visit_direction_failures = {}
        self._goal_visit_inventory = {}
        self._location_resolver_cache = {}
        self._contradiction_splits_this_epoch = set()
        if not keep_experiences:
            self._hazard_room_deaths = {}
            self._hazard_room_evidence = {}
        logger.info(f"Agent reset (keep_experiences={keep_experiences})")

    def act(self, observation: str, score: int, done: bool, info: dict,
            generate_next: bool = True) -> str:
        """Decide the next action given the current game state.

        This is the main method called at each game step.

        Args:
            observation: Current text observation from the game
            score: Current game score
            done: Whether the game has ended
            info: Additional info from Jericho
            generate_next: If False, process the previous action result but do
                not spend an LLM call choosing an unused next command.

        Returns:
            The next command string to send to the game
        """
        # First call in an epoch: no previous action has produced an observation
        # yet, so choose the first command without creating a completed-step log.
        if self.prev_action is None:
            if done or not generate_next:
                self.prev_score = score
                self.total_score = score
                return ""
            self._initialize_kg_from_observation(observation, info=info)
            self.attempt_ledger.record_room_visit(
                self.kg_map.current_location or "unknown",
                step=0,
                epoch=self.current_epoch,
            )
            print("  Initial action generation: preparing first command...", flush=True)
            command, generation = self._generate_command(observation, score)
            print(f"  Initial action generation: first command = {command}", flush=True)
            self.prev_action = command
            self.pending_generation = generation
            self.prev_score = score
            self.total_score = score
            logger.info(f"Initial command: '{command}'")
            return command

        self.step_count += 1
        step_started = time.perf_counter()
        module_timings: dict[str, float] = {}

        def record_timing(name: str, started_at: float):
            module_timings[name] = round(time.perf_counter() - started_at, 4)

        reward_change = score - self.prev_score
        completed_action = self.prev_action
        source_generation = self.pending_generation or {}
        source_state_snapshot = source_generation.get("state_snapshot_at_generation") or {}

        # observation is the game's response to completed_action, so
        # (completed_action, observation) is a correct completed pair.
        self.history.append((completed_action, observation))
        if len(self.history) > config.HISTORY_LENGTH:
            self.history = self.history[-config.HISTORY_LENGTH:]

        detail = {
            "epoch": self.current_epoch,
            "step": self.step_count,
            "observation": observation,
            "score": score,
            "reward_change": reward_change,
            "prev_action": completed_action,
            "executed_action": completed_action,
            "modules": {},
        }

        visited_rooms_before = set(self.kg_map.visited_rooms)
        prev_location = self.kg_map.current_location
        inventory_before = set(self.kg_map.inventory)

        action_valid = None
        action_split = None
        timer = time.perf_counter()
        try:
            is_valid = self.fm.validate_action(completed_action, observation)
            action_valid = is_valid
            if is_valid:
                split = self.fm.split_action(completed_action)
                action_split = split
                self.action_space.store_action(split["verb"], split["objects"])
                logger.debug(f"Valid action stored: {split}")
        except Exception as e:
            logger.warning(f"Action validation/splitting failed: {e}")
            action_valid = f"ERROR: {e}"
        record_timing("fm_action_validation_and_split", timer)

        detail["modules"]["action_space"] = {
            "prev_action_valid": action_valid,
            "action_split": action_split,
            "total_verbs": len(self.action_space.verbs),
            "total_actions_learned": self.action_space.num_actions(),
            "all_verbs": list(self.action_space.verbs.keys()),
        }
        pure_rejection = self._is_pure_rejected_observation(action_valid, observation)
        detail["pure_rejection_skip"] = pure_rejection

        timer = time.perf_counter()
        auxiliary_gate = self._run_auxiliary_gate(
            action=completed_action,
            observation=observation,
            done=done,
            score=score,
            reward_change=reward_change,
            action_valid=action_valid,
            prev_location=prev_location,
            inventory_before=inventory_before,
            visited_rooms_before=visited_rooms_before,
            look_probe_text=info.get("look_probe_text", ""),
        )
        record_timing("auxiliary_gate", timer)
        detail["modules"]["auxiliary_gate"] = auxiliary_gate

        source_visit_id = getattr(self.kg_map, "_current_visit_id", 0)
        kg_location_resolution = self._resolve_step_location(
            auxiliary_gate=auxiliary_gate,
            action=completed_action,
            observation=observation,
            look_probe_text=info.get("look_probe_text", ""),
            previous_location=prev_location,
            done=done,
        )

        extracted_triples = []
        applied_triples = []
        timer = time.perf_counter()
        if pure_rejection or self.kg_map.location_uncertain:
            kg_location_resolution["location_after_update"] = self.kg_map.current_location
            kg_location_resolution["action_transition_status"] = (
                "skipped_location_uncertain"
                if self.kg_map.location_uncertain else "skipped_pure_rejection"
            )
            logger.debug("KG relation extraction skipped for rejection/uncertain room")
        else:
            try:
                extracted_triples = self.fm.extract_relations(completed_action, observation)
                room_title = (
                    "" if (
                        config.AUX_GATE_LOCATION_VERDICT
                        and not kg_location_resolution.get(
                            "use_legacy_location_pipeline"
                        )
                    )
                    else self._extract_observation_room_title(observation)
                )
                fm_location = self._location_from_triples(extracted_triples)
                kg_location_resolution["raw_room_title"] = room_title
                kg_location_resolution["fm_location"] = fm_location
                applied_triples = extracted_triples
                prev_lower = completed_action.lower().strip()
                movement_location_candidate = self._is_location_change_action(
                    completed_action,
                    action_valid,
                )
                if action_valid is False and prev_lower in self.kg_map._direction_set():
                    applied_triples = self._filter_failed_movement_triples(extracted_triples)
                if room_title:
                    if movement_location_candidate:
                        room_title = self.kg_map.resolve_arrival_location(
                            room_title,
                            observation=observation,
                            from_location=prev_location or "",
                            action=completed_action,
                        )
                    title_key = normalize_location_key(room_title) or "unknown"
                    fm_key = normalize_location_key(fm_location) or "unknown"
                    current_key = (
                        normalize_location_key(self.kg_map.current_location or "")
                        or "unknown"
                    )
                    may_apply_title = (
                        movement_location_candidate
                        or not self.kg_map.current_location
                        or title_key == current_key
                    )
                    should_use_title = may_apply_title and (not fm_location or title_key != fm_key)
                    if should_use_title:
                        applied_triples = self._with_authoritative_location_triple(
                            applied_triples,
                            room_title,
                        )
                        kg_location_resolution["title_fallback_used"] = not bool(fm_location)
                        kg_location_resolution["title_overrode_fm"] = bool(fm_location)
                        kg_location_resolution["chosen_location_hint"] = room_title
                    elif not may_apply_title:
                        applied_triples = self._without_location_triples(applied_triples)
                        kg_location_resolution["remote_room_title_ignored"] = room_title
                elif fm_location and not movement_location_candidate:
                    fm_key = normalize_location_key(fm_location) or "unknown"
                    current_key = (
                        normalize_location_key(self.kg_map.current_location or "")
                        or "unknown"
                    )
                    if self.kg_map.current_location and fm_key != current_key:
                        applied_triples = self._without_location_triples(applied_triples)
                        kg_location_resolution["remote_fm_location_ignored"] = fm_location
                if (kg_location_resolution.get("use_legacy_location_pipeline")
                        or not config.AUX_GATE_LOCATION_VERDICT):
                    self.kg_map._legacy_update(applied_triples, completed_action)
                else:
                    self.kg_map.update(applied_triples, completed_action)
                kg_location_resolution["location_after_update"] = self.kg_map.current_location
                gate_accepted = (
                    auxiliary_gate.get("decision", {})
                    .get("command_outcome", {})
                    .get("status") == "accepted"
                )
                movement_direction = self._blocked_direction_for_command(
                    completed_action
                )
                if ((action_valid is True or gate_accepted)
                        and prev_location
                        and self.kg_map.current_location
                        and self.kg_map.current_location != prev_location
                        and not done
                        and movement_direction):
                    split_key = (
                        prev_location,
                        movement_direction,
                    )
                    room_split = self.kg_map.confirm_direction(
                        from_location=prev_location,
                        direction=movement_direction,
                        to_location=self.kg_map.current_location,
                        epoch=self.current_epoch,
                        step=self.step_count,
                        source_visit_id=source_visit_id,
                        allow_split=(
                            config.CONTRADICTION_SPLITTER
                            and split_key not in self._contradiction_splits_this_epoch
                        ),
                        preserve_conflict=(
                            config.CONTRADICTION_SPLITTER
                            and split_key in self._contradiction_splits_this_epoch
                        ),
                    )
                    if room_split:
                        self._contradiction_splits_this_epoch.add(split_key)
                        kg_location_resolution["room_split"] = room_split
                    kg_location_resolution["confirmed_transition_type"] = "direction"
                    kg_location_resolution["confirmed_transition"] = {
                        "from": prev_location,
                        "command": movement_direction,
                        "to": self.kg_map.current_location,
                    }
                elif ((action_valid is True or gate_accepted)
                        and prev_location
                        and self.kg_map.current_location
                        and self.kg_map.current_location != prev_location
                        and not done
                        and not movement_direction):
                    kg_location_resolution["confirmed_transition_type"] = "action_candidate"
                    kg_location_resolution["action_transition_candidate"] = {
                        "from": prev_location,
                        "command": completed_action,
                        "to": self.kg_map.current_location,
                    }
                logger.debug(f"KG-map updated with {len(applied_triples)} triples")
            except Exception as e:
                logger.warning(f"Relation extraction failed: {e}")
                extracted_triples = [("ERROR", str(e), "")]
                applied_triples = extracted_triples
                kg_location_resolution["location_after_update"] = self.kg_map.current_location
        record_timing("fm_relation_extraction_and_kg_update", timer)

        detail["modules"]["kg_map"] = {
            "extracted_triples": [(s, r, o) for s, r, o in extracted_triples],
            "applied_triples": [(s, r, o) for s, r, o in applied_triples],
            "location_resolution": kg_location_resolution,
            "current_location": self.kg_map.current_location,
            "rooms_visited": list(self.kg_map.visited_rooms),
            "inventory": list(self.kg_map.inventory),
            "room_info": copy.deepcopy(self.kg_map.get_current_room_info()),
            "kg_map_context": self.kg_map.to_prompt_string(),
        }

        command_outcome = auxiliary_gate.get("decision", {}).get("command_outcome", {})
        terminal_status = (
            auxiliary_gate.get("decision", {}).get("terminal_outcome", "none")
        )
        is_death = self._classify_death(
            done=done,
            reward_change=reward_change,
            terminal_status=terminal_status,
            observation=observation,
        )
        if command_outcome.get("status") == "accepted" and action_valid is not True:
            raw_action_valid = action_valid
            action_valid = True
            pure_rejection = False
            detail["pure_rejection_skip"] = False
            action_space_detail = detail["modules"].get("action_space", {})
            action_space_detail["fm_prev_action_valid"] = raw_action_valid
            action_space_detail["prev_action_valid"] = True
            action_space_detail["validity_override"] = {
                "source": "auxiliary_gate_command_outcome",
                "status": "accepted",
                "reason": command_outcome.get("reason", ""),
            }
            if action_split is None:
                try:
                    split = self.fm.split_action(completed_action)
                    action_split = split
                    self.action_space.store_action(split["verb"], split["objects"])
                    action_space_detail["action_split"] = action_split
                    action_space_detail["total_verbs"] = len(self.action_space.verbs)
                    action_space_detail["total_actions_learned"] = self.action_space.num_actions()
                    action_space_detail["all_verbs"] = list(self.action_space.verbs.keys())
                except Exception as e:
                    action_space_detail["validity_override_split_error"] = str(e)

        timer = time.perf_counter()
        if pure_rejection:
            action_transition_result = {
                "status": "skipped_pure_rejection",
                "candidate": auxiliary_gate.get("action_transition_candidate", {}),
                "decision": auxiliary_gate.get("decision", {}).get("kg_action_transition", {}),
                "applied": False,
                "reason": "latest command was a clear parser/world rejection",
            }
        else:
            action_transition_result = self._apply_gate_action_transition(
                auxiliary_gate=auxiliary_gate,
            )
        record_timing("kg_action_transition_gate", timer)
        detail["modules"]["kg_action_transition"] = action_transition_result
        kg_location_resolution["action_transition_gate_decision"] = (
            action_transition_result.get("decision", {})
        )
        kg_location_resolution["action_transition_applied"] = bool(
            action_transition_result.get("applied")
        )
        kg_location_resolution["action_transition_status"] = (
            action_transition_result.get("status", "")
        )
        verdict_moved = (
            kg_location_resolution.get("gate_location_verdict", {}).get("moved")
            == "yes"
        )
        if (verdict_moved
                and action_transition_result.get("candidate")
                and not action_transition_result.get("applied")):
            kg_location_resolution["transition_gate_disagreement"] = {
                "location_verdict": "moved",
                "edge_recording": action_transition_result.get("status", "not_applied"),
                "resolution": "location_verdict_wins",
            }
        if action_transition_result.get("applied"):
            kg_location_resolution["confirmed_transition_type"] = "action"
            kg_location_resolution["confirmed_transition"] = (
                action_transition_result.get("candidate", {})
            )
            kg_location_resolution["location_after_update"] = self.kg_map.current_location
            detail["modules"]["kg_map"]["current_location"] = self.kg_map.current_location
            detail["modules"]["kg_map"]["rooms_visited"] = list(self.kg_map.visited_rooms)
            detail["modules"]["kg_map"]["room_info"] = copy.deepcopy(
                self.kg_map.get_current_room_info()
            )
            detail["modules"]["kg_map"]["kg_map_context"] = self.kg_map.to_prompt_string()
        timer = time.perf_counter()
        if pure_rejection:
            inventory_reconciliation = self._skipped_module_result(
                "inventory_reconciliation",
                "skipped_pure_rejection",
            )
            inventory_reconciliation["before"] = list(self.kg_map.inventory)
            inventory_reconciliation["after"] = list(self.kg_map.inventory)
        else:
            inventory_reconciliation = self._apply_gate_inventory_update(
                auxiliary_gate=auxiliary_gate,
                inventory_before=inventory_before,
            )
        record_timing("inventory_reconciliation", timer)
        detail["modules"]["inventory_reconciliation"] = inventory_reconciliation
        if inventory_reconciliation.get("applied"):
            detail["modules"]["kg_map"]["inventory"] = list(self.kg_map.inventory)
            detail["modules"]["kg_map"]["room_info"] = copy.deepcopy(
                self.kg_map.get_current_room_info()
            )
            detail["modules"]["kg_map"]["kg_map_context"] = self.kg_map.to_prompt_string()

        timer = time.perf_counter()
        if pure_rejection:
            world_state_extraction = self._skipped_module_result(
                "world_state_extraction",
                "skipped_pure_rejection",
            )
        else:
            world_state_extraction = self._apply_gate_world_state_update(
                auxiliary_gate=auxiliary_gate,
            )
        record_timing("world_state_extraction", timer)
        detail["modules"]["world_state_extraction"] = world_state_extraction
        if world_state_extraction.get("applied"):
            detail["modules"]["kg_map"]["room_info"] = copy.deepcopy(
                self.kg_map.get_current_room_info()
            )
            detail["modules"]["kg_map"]["kg_map_context"] = self.kg_map.to_prompt_string()

        environmental_change_detection = (
            auxiliary_gate.get("environmental_change_detection") or {}
        )
        if pure_rejection:
            environmental_change_detection = {
                "status": "skipped_pure_rejection",
                "environmental_change": False,
                "evidence": "",
                "source": "pure_rejection_skip",
            }
            module_timings["legacy_environmental_detection"] = 0.0
        elif auxiliary_gate.get("use_legacy_environmental_detection"):
            timer = time.perf_counter()
            environmental_change_detection = self._detect_environmental_change_with_llm(
                action=completed_action,
                observation=observation,
                action_valid=action_valid,
            )
            environmental_change_detection["source"] = "legacy_fallback_after_gate_failure"
            record_timing("legacy_environmental_detection", timer)
        else:
            module_timings["legacy_environmental_detection"] = 0.0
        detail["modules"]["environmental_change_detection"] = environmental_change_detection

        goal_lifecycle_transitions = self._update_goal_visit_lifecycle(
            previous_location=prev_location,
            current_location=self.kg_map.current_location or "unknown",
            done=done,
            inventory=list(self.kg_map.inventory),
        )

        timer = time.perf_counter()
        if pure_rejection:
            situation_resolution = {
                "status": "skipped_pure_rejection",
                "removed_situations": [],
                "reason": "latest command was a clear parser/world rejection",
            }
            situation_detail = self._skipped_situation_detection_result(
                action=completed_action,
                just_resolved=[],
                gate_decision={
                    "run": False,
                    "reason": "latest command was a clear parser/world rejection",
                },
            )
        else:
            situation_resolution = self._resolve_stored_situations(
                action=completed_action,
                observation=observation,
                score=score,
                reward_change=reward_change,
                action_valid=action_valid,
                prev_location=prev_location,
                environmental_change=environmental_change_detection,
            )
            if (auxiliary_gate.get("decision", {})
                    .get("stored_situation_detection", {})
                    .get("run", True)):
                situation_detail = self._detect_and_store_situation(
                    action=completed_action,
                    observation=observation,
                    just_resolved=situation_resolution.get("removed_situations", []),
                )
            else:
                situation_detail = self._skipped_situation_detection_result(
                    action=completed_action,
                    just_resolved=situation_resolution.get("removed_situations", []),
                    gate_decision=auxiliary_gate.get("decision", {}).get(
                        "stored_situation_detection", {}
                    ),
                )
        situation_detail["resolution"] = situation_resolution
        situation_detail["goal_transitions"] = goal_lifecycle_transitions
        situation_detail["goal_situations"] = self.situation_memory.goal_situations()
        situation_detail["active_goal_count"] = self.situation_memory.open_goal_count()
        situation_detail["active_situations_after"] = self.situation_memory.active_situations()
        detail["modules"]["situation_memory"] = situation_detail
        record_timing("stored_situation_detection_resolution", timer)

        timer = time.perf_counter()
        experience_summary = None
        score_summary_skipped = None
        summary_log_entries = []
        experience_triggered = bool(done or (reward_change != 0 and action_valid is True))
        if experience_triggered:
            try:
                history_text = self._format_history()
                score_location = self._first_known_location(
                    prev_location,
                    self.kg_map.current_location,
                    "Starting Location",
                )
                location_after = self._first_known_location(
                    self.kg_map.current_location,
                    score_location,
                    "Starting Location",
                )
                score_location_fingerprint = self._location_fingerprint_hash(
                    score_location
                )
                score_experience_kind = (
                    "death_warning" if is_death
                    else "achievement" if reward_change > 0
                    else "terminal"
                )
                score_event_key = ""
                death_event_key = ""
                precomputed_loss_summary = ""
                grounded_death_room = ""
                death_room_grounding = {}
                goal_hazard = None
                if is_death:
                    ledger_block = self.attempt_ledger.format_room_block(
                        score_location
                    )
                    precomputed_loss_summary = self.llm.summarize_loss_experience(
                        history=history_text,
                        reward_change=reward_change,
                        current_score=score,
                        fatal_action=completed_action,
                        location=location_after,
                        location_issued=score_location,
                        command_history_block=ledger_block,
                    )
                    grounded_death_room, death_room_grounding = (
                        self._ground_death_room_from_summary(
                            precomputed_loss_summary,
                            observation,
                        )
                    )
                    goal_hazard = self._goal_hazard_context(
                        action=completed_action,
                        action_valid=action_valid,
                        location_issued=score_location,
                        location_after=location_after,
                        grounded_death_room=grounded_death_room,
                    )
                    death_event_key = self._death_event_key(
                        action=completed_action,
                        location=goal_hazard["hazard_location"],
                        location_fingerprint=goal_hazard["hazard_fingerprint"],
                    )
                elif reward_change > 0:
                    score_event_key = self._score_event_key(
                        trigger="gain",
                        action=completed_action,
                        location=score_location,
                        reward_change=reward_change,
                        location_fingerprint=score_location_fingerprint,
                    )
                    self.earned_score_event_keys_this_epoch.add(score_event_key)
                    self.earned_score_location_reward_keys_this_epoch.add(
                        self._score_location_reward_key(
                            score_location,
                            reward_change,
                            location_fingerprint=score_location_fingerprint,
                        )
                    )

                event_key = death_event_key or score_event_key
                if event_key and self.experience_lib.event_seen(event_key):
                    score_summary_skipped = {
                        "trigger": "death" if is_death else "score_change",
                        "event_key": event_key,
                        "reason": "duplicate_death_warning" if is_death else "duplicate_score_gain",
                        "kind": score_experience_kind,
                        "epoch": self.current_epoch,
                        "step": self.step_count,
                        "score_change": reward_change,
                        "current_score": score,
                        "location": score_location,
                        "location_fingerprint": score_location_fingerprint,
                        "action": completed_action,
                        "terminal": done,
                        "terminal_status": terminal_status,
                    }
                    if is_death and goal_hazard:
                        score_summary_skipped.update({
                            "location": goal_hazard["hazard_location"],
                            "location_after": goal_hazard["hazard_location"],
                            "location_fingerprint": goal_hazard["hazard_fingerprint"],
                            "location_registry_id": goal_hazard.get(
                                "hazard_registry_id", ""
                            ),
                            "death_room_title": grounded_death_room,
                            "death_room_grounding": death_room_grounding,
                        })
                    self.experience_lib.record_event(
                        event_key,
                        metadata=score_summary_skipped,
                    )
                    logger.info(
                        f"Experience skipped as duplicate {score_experience_kind}: "
                        f"{completed_action}"
                    )
                else:
                    validation = {}
                    if is_death:
                        exp_summary = precomputed_loss_summary
                    else:
                        exp_summary = self.llm.summarize_experience(
                            history=history_text,
                            reward_change=reward_change,
                            current_score=score,
                            scoring_action=completed_action,
                            location_issued=score_location,
                            location_after=location_after,
                        )
                        if reward_change > 0:
                            exp_summary, validation = self._validate_score_summary(
                                summary=exp_summary,
                                history_text=history_text,
                                reward_change=reward_change,
                                current_score=score,
                                scoring_action=completed_action,
                                location_issued=score_location,
                                location_after=location_after,
                            )
                    experience_summary = exp_summary
                    score_summary_prompt = self.llm.last_summary_prompt or ""
                    score_metadata = {
                        "kind": score_experience_kind,
                        "trigger": "death" if is_death else "score_change",
                        "score_change": reward_change,
                        "current_score": score,
                        "epoch": self.current_epoch,
                        "step": self.step_count,
                        "location": score_location,
                        "location_issued": score_location,
                        "location_fingerprint": score_location_fingerprint,
                        "location_registry_id": self.kg_map.registry_id_for(
                            score_location
                        ),
                        "location_after": location_after,
                        "action": completed_action,
                        "scoring_action": (
                            completed_action if reward_change > 0 and not is_death else ""
                        ),
                        "fatal_action": completed_action if is_death else "",
                        "terminal": done,
                        "terminal_status": terminal_status,
                        "is_death": is_death,
                    }
                    if is_death and goal_hazard:
                        score_metadata.update({
                            "location": goal_hazard["hazard_location"],
                            "location_after": goal_hazard["hazard_location"],
                            "location_fingerprint": goal_hazard["hazard_fingerprint"],
                            "location_registry_id": goal_hazard.get(
                                "hazard_registry_id", ""
                            ),
                            "death_room_title": grounded_death_room,
                            "death_room_grounding": death_room_grounding,
                        })
                    if reward_change > 0 and not is_death:
                        score_metadata["summary_validation"] = validation
                    if event_key:
                        score_metadata["event_key"] = event_key
                    self.experience_lib.store_experience(
                        experience_text=exp_summary,
                        metadata=score_metadata,
                    )
                    if event_key:
                        self.experience_lib.record_event(
                            event_key,
                            metadata={
                                **score_metadata,
                                "status": "stored",
                            },
                        )
                    summary_log_entries.append({
                        "state_type": (
                            "score_loss" if is_death and reward_change < 0
                            else "terminal_defeat" if is_death
                            else "terminal" if done and reward_change == 0
                            else "score_change"
                        ),
                        "prompt": score_summary_prompt,
                        "summary": exp_summary,
                        "raw_response": self.llm.last_summary_raw_response or "",
                        "metadata": score_metadata,
                    })
                    logger.info(f"Experience stored: score change {reward_change:+d}")

                if is_death:
                    goal_hazard = goal_hazard or self._goal_hazard_context(
                        action=completed_action,
                        action_valid=action_valid,
                        location_issued=score_location,
                        location_after=location_after,
                    )
                    goal_transition, hypothesis_log = (
                        self._handle_death_goal_lifecycle(
                            event_key=event_key,
                            hazard_location=goal_hazard["hazard_location"],
                            hazard_fingerprint=goal_hazard["hazard_fingerprint"],
                            fatal_action=completed_action,
                            death_observation=observation,
                            inventory_at_death=list(self.kg_map.inventory),
                            gateway=goal_hazard.get("gateway"),
                        )
                    )
                    if goal_transition:
                        situation_detail.setdefault("goal_transitions", []).append(
                            goal_transition
                        )
                        situation_detail["goal_situations"] = (
                            self.situation_memory.goal_situations()
                        )
                        situation_detail["active_goal_count"] = (
                            self.situation_memory.open_goal_count()
                        )
                        situation_detail["active_situations_after"] = (
                            self.situation_memory.active_situations()
                        )
                    if hypothesis_log:
                        summary_log_entries.append(hypothesis_log)

                if (not is_death) and reward_change > 0 and score_event_key:
                    enabler_entries = self._store_reward_enabler_experiences(
                        score_event_key=score_event_key,
                        reward_change=reward_change,
                        scoring_action=completed_action,
                        scoring_location=score_location,
                        location_after=location_after,
                    )
                    summary_log_entries.extend(enabler_entries)
            except Exception as e:
                logger.warning(f"Experience summarization failed: {e}")
                experience_summary = f"ERROR: {e}"

        neutral_triggers = []
        neutral_summaries = []
        neutral_summaries_skipped = []
        neutral_event_keys = []
        if reward_change == 0 and not done and not pure_rejection:
            neutral_triggers = self._detect_neutral_triggers(
                observation=observation,
                action_valid=action_valid,
                visited_rooms_before=visited_rooms_before,
                prev_location=prev_location,
                environmental_change=environmental_change_detection,
                auxiliary_gate=auxiliary_gate,
            )
            for trigger_type, trigger_meta in neutral_triggers:
                try:
                    neutral_location = self.kg_map.current_location or "unknown"
                    neutral_issued_location = (
                        trigger_meta.get("prev_location")
                        if trigger_type == "navigation"
                        else neutral_location
                    ) or neutral_location
                    neutral_location_fingerprint = self._location_fingerprint_hash(
                        neutral_issued_location
                    )
                    neutral_location_registry_id = self.kg_map.registry_id_for(
                        neutral_issued_location
                    )
                    event_key = self._neutral_event_key(
                        trigger=trigger_type,
                        action=completed_action,
                        observation=observation,
                        location=neutral_location,
                        prev_location=trigger_meta.get("prev_location"),
                        failed_attempts=trigger_meta.get("failed_attempts"),
                    )
                    if self.experience_lib.neutral_event_seen(event_key):
                        neutral_summaries_skipped.append({
                            "trigger": trigger_type,
                            "event_key": event_key,
                            "reason": "duplicate_neutral_event",
                        })
                        neutral_event_keys.append({
                            "trigger": trigger_type,
                            "event_key": event_key,
                            "status": "skipped_duplicate",
                        })
                        logger.info(f"Neutral experience skipped as duplicate: {trigger_type}")
                        continue

                    summary = self.llm.summarize_neutral_experience(
                        trigger=trigger_type,
                        action=completed_action,
                        observation=observation,
                        location=self.kg_map.current_location or "unknown",
                        prev_location=trigger_meta.get("prev_location"),
                        failed_attempts=trigger_meta.get("failed_attempts"),
                    )
                    if self._is_empty_experience_summary(summary):
                        self.experience_lib.record_neutral_event(
                            event_key,
                            metadata={
                                "trigger": trigger_type,
                                "action": completed_action,
                                "location": neutral_location,
                                "location_issued": neutral_issued_location,
                                "location_fingerprint": neutral_location_fingerprint,
                                "location_registry_id": neutral_location_registry_id,
                                "prev_location": trigger_meta.get("prev_location"),
                                "failed_attempts": trigger_meta.get("failed_attempts") or [],
                                "gate_reason": trigger_meta.get("gate_reason", ""),
                                "gate_evidence": trigger_meta.get("gate_evidence", ""),
                                "epoch": self.current_epoch,
                                "step": self.step_count,
                                "score": score,
                                "status": "summary_none",
                            },
                        )
                        neutral_summaries_skipped.append({
                            "trigger": trigger_type,
                            "event_key": event_key,
                            "reason": "summary_none",
                        })
                        neutral_event_keys.append({
                            "trigger": trigger_type,
                            "event_key": event_key,
                            "status": "skipped_none",
                        })
                        logger.info(f"Neutral experience skipped as none: {trigger_type}")
                        continue

                    if summary:
                        neutral_summary_prompt = self.llm.last_summary_prompt or ""
                        neutral_kind = self._experience_kind_for_trigger(trigger_type)
                        self.experience_lib.store_experience(
                            experience_text=summary,
                            metadata={
                                "kind": neutral_kind,
                                "trigger": trigger_type,
                                "score_change": 0,
                                "current_score": score,
                                "epoch": self.current_epoch,
                                "step": self.step_count,
                                "location": neutral_location,
                                "location_issued": neutral_issued_location,
                                "location_fingerprint": neutral_location_fingerprint,
                                "location_registry_id": neutral_location_registry_id,
                                "prev_location": trigger_meta.get("prev_location"),
                                "action": completed_action,
                                "event_key": event_key,
                            },
                        )
                        self.experience_lib.record_neutral_event(
                            event_key,
                            metadata={
                                "trigger": trigger_type,
                                "action": completed_action,
                                "location": neutral_location,
                                "location_issued": neutral_issued_location,
                                "location_fingerprint": neutral_location_fingerprint,
                                "location_registry_id": neutral_location_registry_id,
                                "prev_location": trigger_meta.get("prev_location"),
                                "failed_attempts": trigger_meta.get("failed_attempts") or [],
                                "gate_reason": trigger_meta.get("gate_reason", ""),
                                "gate_evidence": trigger_meta.get("gate_evidence", ""),
                                "epoch": self.current_epoch,
                                "step": self.step_count,
                                "score": score,
                            },
                        )
                        neutral_summaries.append((trigger_type, summary))
                        neutral_event_keys.append({
                            "trigger": trigger_type,
                            "event_key": event_key,
                            "status": "stored",
                        })
                        summary_log_entries.append({
                            "state_type": trigger_type,
                            "prompt": neutral_summary_prompt,
                            "summary": summary,
                            "raw_response": self.llm.last_summary_raw_response or "",
                            "metadata": {
                                "kind": neutral_kind,
                                "score_change": 0,
                                "current_score": score,
                                "epoch": self.current_epoch,
                                "step": self.step_count,
                                "location": neutral_location,
                                "location_issued": neutral_issued_location,
                                "location_fingerprint": neutral_location_fingerprint,
                                "prev_location": trigger_meta.get("prev_location"),
                                "action": completed_action,
                                "event_key": event_key,
                                "gate_reason": trigger_meta.get("gate_reason", ""),
                                "gate_evidence": trigger_meta.get("gate_evidence", ""),
                            },
                        })
                        logger.info(f"Neutral experience stored: {trigger_type}")
                except Exception as e:
                    logger.warning(f"Neutral experience failed ({trigger_type}): {e}")
        record_timing("experience_summary_and_storage", timer)

        timer = time.perf_counter()
        source_location_for_attempt = prev_location or self.kg_map.current_location or "unknown"
        action_failure_memory = {
            "status": "not_applicable",
            "source_location": source_location_for_attempt,
            "command": completed_action,
            "stored_failure": None,
            "removed_failure": None,
            "known_failures_here_after": [],
            "world_signature": {},
            "failure_reason_prompt": "",
            "failure_reason_raw_response": "",
            "error": "",
        }

        if action_valid is True:
            self.consecutive_failures = 0
            self.recent_failed_actions = []
            removed_failure = self.failed_action_memory.remove(
                action_failure_memory["source_location"],
                completed_action,
            )
            if removed_failure:
                action_failure_memory["status"] = "removed_after_success"
                action_failure_memory["removed_failure"] = removed_failure
            else:
                action_failure_memory["status"] = "valid_no_prior_failure"
        elif action_valid is False:
            self.consecutive_failures += 1
            self.recent_failed_actions.append(completed_action)
            if len(self.recent_failed_actions) > 5:
                self.recent_failed_actions = self.recent_failed_actions[-5:]
            failure_location = action_failure_memory["source_location"]
            world_signature = self._world_signature(failure_location, score)
            action_failure_memory["world_signature"] = world_signature
            fallback_reason = self._fallback_failure_reason(observation)
            failure_reason = fallback_reason
            try:
                reason = self.llm.explain_action_failure(
                    location=failure_location,
                    command=completed_action,
                    observation=observation,
                    world_signature=world_signature,
                )
                action_failure_memory["failure_reason_prompt"] = (
                    self.llm.last_failure_reason_prompt or ""
                )
                action_failure_memory["failure_reason_raw_response"] = (
                    self.llm.last_failure_reason_raw_response or ""
                )
                if reason:
                    failure_reason = reason
            except Exception as e:
                logger.warning(f"Action failure explanation failed: {e}")
                action_failure_memory["error"] = str(e)
                action_failure_memory["failure_reason_prompt"] = (
                    self.llm.last_failure_reason_prompt or ""
                )
                action_failure_memory["failure_reason_raw_response"] = (
                    self.llm.last_failure_reason_raw_response or ""
                )
            failure_status, failure_record = self.failed_action_memory.record(
                location=failure_location,
                command=completed_action,
                observation=observation,
                failure_reason=failure_reason,
                world_signature=world_signature,
            )
            action_failure_memory["status"] = failure_status
            action_failure_memory["stored_failure"] = failure_record
        elif action_valid is not None:
            action_failure_memory["status"] = "validation_unknown"

        action_failure_memory["known_failures_here_after"] = (
            self.failed_action_memory.records_for_location(
                action_failure_memory["source_location"]
            )
        )
        detail["modules"]["action_failure_memory"] = action_failure_memory
        record_timing("action_failure_memory", timer)

        timer = time.perf_counter()
        location_changed_for_affordance = (
            bool(prev_location)
            and bool(self.kg_map.current_location)
            and self.kg_map.current_location != prev_location
        )
        completed_direction = self._blocked_direction_for_command(completed_action)
        if location_changed_for_affordance:
            self._reset_direction_visit_budget(
                previous_location=prev_location,
                current_location=self.kg_map.current_location,
            )
        elif (not is_death) and action_valid is False and completed_direction:
            failure_location = source_location_for_attempt
            self.kg_map.mark_direction_tried_at(
                completed_direction,
                failure_location,
            )
            self._record_visit_direction_failure(
                location=failure_location,
                direction=completed_direction,
                observation=observation,
                step=self.step_count,
            )
            detail["modules"]["kg_map"]["room_info"] = copy.deepcopy(
                self.kg_map.get_current_room_info()
            )
            detail["modules"]["kg_map"]["kg_map_context"] = self.kg_map.to_prompt_string()
        inventory_changed_for_repetition = set(self.kg_map.inventory) != inventory_before
        if not source_state_snapshot:
            source_location = source_location_for_attempt
            source_state_snapshot = self._repetition_state_snapshot(
                location=source_location,
                observation=source_generation.get("observation_at_generation", ""),
                visible_objects=self._visible_objects_for_location(source_location),
                inventory=list(inventory_before),
                score=source_generation.get("score_at_generation", self.prev_score),
            )

        progress_signals = {
            "action_valid": action_valid,
            "score_changed": reward_change != 0,
            "location_changed": location_changed_for_affordance,
            "inventory_changed": inventory_changed_for_repetition,
            "environment_changed": bool(environmental_change_detection.get("environmental_change")),
            "terminal": bool(done),
        }
        state_repetition_memory = {
            "status": "not_applicable",
            "command": completed_action,
            "source_state_snapshot": source_state_snapshot,
            "progress_signals": progress_signals,
            "prompt": "",
            "llm_raw_response": "",
            "response_body": "",
            "parsed": {},
            "stored_record": None,
            "removed_records": [],
            "same_state_records_after": [],
            "error": "",
        }
        source_memory_location = source_state_snapshot.get(
            "location",
            source_location_for_attempt,
        )
        progress_was_useful = bool(
            reward_change != 0
            or location_changed_for_affordance
            or inventory_changed_for_repetition
            or environmental_change_detection.get("environmental_change")
            or done
        )

        if action_valid is False:
            failure_reason = self._fallback_failure_reason(observation)
            stored, repetition_record = self.state_action_memory.record(
                state_snapshot=source_state_snapshot,
                command=completed_action,
                result_observation=observation,
                reason=failure_reason,
                source="fm_invalid",
            )
            state_repetition_memory["status"] = "stored_invalid" if stored else "duplicate_invalid"
            state_repetition_memory["stored_record"] = repetition_record
        elif action_valid is True and progress_was_useful:
            removed_records = self.state_action_memory.remove_command_records(
                source_memory_location,
                completed_action,
            )
            state_repetition_memory["status"] = (
                "removed_after_useful_progress" if removed_records else "useful_progress"
            )
            state_repetition_memory["removed_records"] = removed_records
        elif action_valid is True:
            try:
                response_body = self.llm.evaluate_action_repetition(
                    state_snapshot=source_state_snapshot,
                    command=completed_action,
                    observation=observation,
                    progress_signals=progress_signals,
                )
                state_repetition_memory["prompt"] = self.llm.last_repetition_eval_prompt or ""
                state_repetition_memory["llm_raw_response"] = (
                    self.llm.last_repetition_eval_raw_response or ""
                )
                state_repetition_memory["response_body"] = response_body
                parsed, parse_error = self._parse_repetition_evaluation_response(response_body)
                if parse_error:
                    state_repetition_memory["status"] = "eval_parse_error"
                    state_repetition_memory["error"] = parse_error
                else:
                    state_repetition_memory["parsed"] = parsed
                    if parsed.get("remember"):
                        stored, repetition_record = self.state_action_memory.record(
                            state_snapshot=source_state_snapshot,
                            command=completed_action,
                            result_observation=observation,
                            reason=parsed.get("reason", ""),
                            source="llm_unproductive_eval",
                        )
                        state_repetition_memory["status"] = (
                            "stored_unproductive" if stored else "duplicate_unproductive"
                        )
                        state_repetition_memory["stored_record"] = repetition_record
                    else:
                        state_repetition_memory["status"] = "not_remembered_useful_or_new_info"
            except Exception as e:
                logger.warning(f"Action repetition evaluation failed: {e}")
                state_repetition_memory["status"] = "eval_error"
                state_repetition_memory["error"] = str(e)
                state_repetition_memory["prompt"] = self.llm.last_repetition_eval_prompt or ""
                state_repetition_memory["llm_raw_response"] = (
                    self.llm.last_repetition_eval_raw_response or ""
                )

        state_repetition_memory["same_state_records_after"] = (
            self.state_action_memory.records_for_state(source_state_snapshot)
        )
        detail["modules"]["state_repetition_memory"] = state_repetition_memory
        record_timing("state_repetition_memory", timer)

        timer = time.perf_counter()
        room_visit_record = None
        if location_changed_for_affordance:
            room_visit_record = self.attempt_ledger.record_room_visit(
                self.kg_map.current_location or "unknown",
                step=self.step_count,
                epoch=self.current_epoch,
            )
        attempt_ledger_detail = self.attempt_ledger.record_step(
            location=source_memory_location,
            command=completed_action,
            observation=observation,
            action_valid=action_valid if isinstance(action_valid, bool) else None,
            reward_change=reward_change,
            location_changed=location_changed_for_affordance,
            destination=self.kg_map.current_location or "",
            inventory_changed=inventory_changed_for_repetition,
            environment_changed=bool(environmental_change_detection.get("environmental_change")),
            repetition_status=state_repetition_memory.get("status", ""),
            terminal_defeat=is_death,
            state_key=self._snapshot_key(source_state_snapshot),
            step=self.step_count,
            epoch=self.current_epoch,
        )
        attempt_ledger_detail["current_room_visit"] = room_visit_record
        attempt_ledger_detail["room_context_after"] = self.attempt_ledger.format_room_block(
            self.kg_map.current_location or "unknown",
            current_state_key=self._snapshot_key(self._repetition_state_snapshot(
                location=self.kg_map.current_location or "unknown",
                observation=observation,
                visible_objects=self._visible_objects_for_location(self.kg_map.current_location or "unknown"),
                inventory=list(self.kg_map.inventory),
                score=score,
            )),
        )
        detail["modules"]["attempt_ledger"] = attempt_ledger_detail
        if attempt_ledger_detail.get("status") == "recorded":
            self._recent_outcomes.append({
                "epoch": int(self.current_epoch),
                "step": int(self.step_count),
                "location": source_memory_location,
                "destination": self.kg_map.current_location or "",
                "command": completed_action,
                "observation": self._clean_text(observation)[:300],
                "outcome_class": attempt_ledger_detail.get("outcome_class", ""),
                "reward_change": int(reward_change or 0),
            })
            self._recent_outcomes = self._recent_outcomes[-8:]
        record_timing("attempt_ledger", timer)

        timer = time.perf_counter()
        affordance_attempt_memory = {
            "status": "not_applicable",
            "command": completed_action,
            "location": source_memory_location,
            "useful": None,
            "state_signature": {},
        }
        if action_valid is True:
            source_affordance_signature = self.affordance_brainstormer.state_signature(
                location=source_memory_location,
                visible_objects=source_state_snapshot.get("visible_objects", []),
                inventory=source_state_snapshot.get("inventory", []),
                score=source_state_snapshot.get("score", score),
                observation=source_state_snapshot.get("observation", ""),
            )
            useful_for_affordance = not bool(
                state_repetition_memory.get("stored_record")
                and state_repetition_memory.get("status") in {
                    "stored_unproductive",
                    "duplicate_unproductive",
                }
            )
            self.affordance_brainstormer.record_attempt_result(
                location=source_memory_location,
                state_signature=source_affordance_signature,
                command=completed_action,
                useful=useful_for_affordance,
            )
            affordance_attempt_memory = {
                "status": "ignored_useful" if useful_for_affordance else "recorded_unproductive",
                "command": completed_action,
                "location": source_memory_location,
                "useful": useful_for_affordance,
                "state_signature": source_affordance_signature,
            }
        detail["modules"]["affordance_attempt_memory"] = affordance_attempt_memory
        record_timing("affordance_attempt_memory", timer)

        timer = time.perf_counter()
        query = f"Location: {self.kg_map.current_location}. Observation: {observation[:200]}"
        experiences = self._retrieve_experiences_for_prompt(query)
        record_timing("experience_retrieval", timer)

        detail["modules"]["experience_lib"] = {
            "score_changed": reward_change != 0,
            "terminal": done,
            "experience_triggered": experience_triggered,
            "new_experience_summary": experience_summary,
            "score_summary_skipped": score_summary_skipped,
            "neutral_triggers_fired": [t for t, _ in neutral_triggers],
            "neutral_summaries": neutral_summaries,
            "neutral_summaries_skipped": neutral_summaries_skipped,
            "neutral_event_keys": neutral_event_keys,
            "summary_log_entries": summary_log_entries,
            "retrieved_experiences": experiences,
            "retrieval_debug": self.last_retrieval_debug,
            "total_experiences": self.experience_lib.num_experiences(),
        }

        room_info = self.kg_map.get_current_room_info()
        current_objects = room_info.get("objects", [])
        detail["modules"]["action_space"]["action_space_context"] = (
            self.action_space.to_prompt_string(current_objects)
        )
        detail["modules"]["action_generation"] = self.pending_generation or {
            "parsed_command": completed_action,
        }
        detail["final_command"] = completed_action

        if done or not generate_next:
            module_timings["completed_step_total"] = round(
                time.perf_counter() - step_started,
                4,
            )
            detail["modules"]["module_timings"] = module_timings
            self.prev_action = None
            self.pending_generation = None
            self.prev_score = score
            self.total_score = score
            self.step_details.append(detail)
            logger.info(f"Completed step {self.step_count}: score={score} "
                        f"({reward_change:+d}) cmd='{completed_action}' "
                        f"loc='{self.kg_map.current_location}'")
            return ""

        timer = time.perf_counter()
        command, generation = self._generate_command(
            observation,
            score,
            experiences,
            reset_affordance_cache=bool(
                reward_change != 0
                or environmental_change_detection.get("environmental_change")
            ),
            affordance_gate_decision=auxiliary_gate.get("decision", {}).get(
                "affordance_brainstorming", {}
            ),
        )
        record_timing("next_action_generation_total", timer)
        for key, value in (generation.get("timings") or {}).items():
            module_timings[f"next_action_generation.{key}"] = value
        module_timings["completed_step_total"] = round(
            time.perf_counter() - step_started,
            4,
        )
        detail["modules"]["module_timings"] = module_timings
        detail["next_command"] = command
        detail["modules"]["next_action_generation"] = generation
        self.prev_action = command
        self.pending_generation = generation
        self.prev_score = score
        self.total_score = score
        self.step_details.append(detail)

        logger.info(f"Completed step {self.step_count}: score={score} "
                    f"({reward_change:+d}) cmd='{completed_action}' "
                    f"next='{command}' loc='{self.kg_map.current_location}'")
        return command

    def _initialize_kg_from_observation(self, observation: str, info: dict = None):
        """Seed the KG-map from the game's initial observation before step 1."""
        if self.initial_observation_processed:
            return
        self.initial_observation_processed = True
        if not observation:
            return
        info = info or {}
        probe_text = str(info.get("look_probe_text", "") or "")
        try:
            print("  Initial KG seed: extracting initial room/object facts...", flush=True)
            triples = self.fm.extract_relations("look", observation, max_new=192)
            room_title = ""
            if config.AUX_GATE_LOCATION_VERDICT:
                try:
                    response = self.llm.gate_auxiliary_modules(
                        location="unknown",
                        previous_location="unknown",
                        action="look",
                        action_valid=True,
                        observation=observation,
                        done=False,
                        score=0,
                        reward_change=0,
                        rooms_visited_before=[],
                        inventory_before=[],
                        inventory=[],
                        visible_objects=[],
                        active_situations=[],
                        recent_failed_commands=[],
                        known_failed_commands_here="[]",
                        problem_attempts_here="[]",
                        recent_command_outcomes=[],
                        same_state_tried_commands=[],
                        action_transition_candidate={},
                        cached_affordance_ideas_available=0,
                        look_probe_text=probe_text,
                    )
                    parsed, parse_error = self._parse_auxiliary_gate_response(response)
                    if not parse_error:
                        raw_verdict = parsed.get("location", {})
                        if not isinstance(raw_verdict, dict):
                            raw_verdict = {}
                        raw_title = self._clean_text(raw_verdict.get("room_title", ""))
                        room_title, _ = self._ground_room_title(
                            raw_title, observation, probe_text
                        )
                except Exception as e:
                    logger.warning("Initial location gate failed: %s", e)
            room_title = (
                room_title
                or self._extract_probe_room_title(probe_text)
                or self._extract_observation_room_title(observation, scan_all_lines=True)
            )
            if not room_title:
                room_title = "Starting Location"
            room_title, _ = self.kg_map.mint_room(
                room_title,
                observation=probe_text or observation,
                epoch=self.current_epoch,
            )
            self.kg_map.confirm_arrival(
                room_title,
                observation=probe_text or observation,
                from_location="",
                action="look",
            )
            self.kg_map.update(triples, "look")
            title_index = observation.lower().find(room_title.lower())
            seed_observation = (
                observation[title_index:]
                if title_index >= 0 else observation
            )
            self.kg_map.seed_room_fingerprint(room_title, seed_observation)
            print(f"  Initial KG seed: {len(triples)} triple(s) extracted.", flush=True)
            logger.info(f"Initial KG-map seeded with {len(triples)} triples")
        except Exception as e:
            logger.warning(f"Initial KG-map extraction failed: {e}")
            print(f"  Initial KG seed skipped: {e}", flush=True)

    def _resolve_step_location(self, auxiliary_gate: dict, action: str,
                               observation: str, look_probe_text: str,
                               previous_location: str, done: bool) -> dict:
        """Resolve one arrival from grounded gate text without engine identity."""
        decision = (auxiliary_gate or {}).get("decision", {})
        verdict = decision.get("location_verdict", {})
        moved = self._clean_text(verdict.get("moved", "unclear")).lower()
        if moved not in {"yes", "no", "unclear"}:
            moved = "unclear"
        raw_title = self._clean_text(verdict.get("room_title", ""))
        grounded_title, grounded = self._ground_room_title(
            raw_title, observation, look_probe_text
        )
        probe_title = self._extract_probe_room_title(look_probe_text)
        fallback_title = (
            probe_title
            or self._extract_observation_room_title(observation, scan_all_lines=True)
        )
        gate_ok = (
            config.AUX_GATE_LOCATION_VERDICT
            and (auxiliary_gate or {}).get("status") == "routed"
        )
        result = {
            "raw_room_title": raw_title,
            "fm_location": "",
            "title_fallback_used": False,
            "title_overrode_fm": False,
            "chosen_location_hint": "",
            "previous_location": previous_location,
            "location_after_update": self.kg_map.current_location,
            "confirmed_transition_type": "",
            "confirmed_transition": {},
            "action_transition_candidate": {},
            "action_transition_gate_decision": {},
            "action_transition_applied": False,
            "action_transition_status": "",
            "gate_location_verdict": copy.deepcopy(verdict),
            "verdict_validated": bool(grounded),
            "probe_title": probe_title,
            "resolver_invoked": False,
            "resolver_decision": "",
            "resolver_confidence": "",
            "registry_id": self.kg_map.registry_id_for(
                self.kg_map.current_location or ""
            ),
            "resolution_mode": "gate" if gate_ok else "text_fallback",
            "location_uncertain": False,
            "room_split": {},
            "use_legacy_location_pipeline": False,
        }
        if not config.AUX_GATE_LOCATION_VERDICT:
            return result
        if gate_ok and moved == "no" and not self.kg_map.location_uncertain:
            result["action_transition_status"] = "gate_no_move"
            return result

        selected_title = grounded_title if grounded else fallback_title
        if selected_title and not grounded:
            result["title_fallback_used"] = True
            result["resolution_mode"] = "text_fallback"
        if done:
            result["chosen_location_hint"] = canonical_room_display(selected_title)
            result["action_transition_status"] = "terminal_location_not_applied"
            return result
        if gate_ok and moved == "yes" and not selected_title:
            self.kg_map.set_location_uncertain(True)
            result["location_uncertain"] = True
            result["action_transition_status"] = "moved_destination_unseen"
            return result
        if not selected_title:
            if not gate_ok or moved == "unclear":
                result["use_legacy_location_pipeline"] = True
            return result

        should_arrive = (
            moved in {"yes", "unclear"} or self.kg_map.location_uncertain
        ) if gate_ok else (
            self._is_location_change_action(action, True)
        )
        if not should_arrive:
            return result
        description = look_probe_text or observation
        resolved, resolver_log = self._resolve_arrival_identity(
            title=selected_title,
            description=description,
            action=action,
            from_location=previous_location,
        )
        result.update(resolver_log)
        if not resolved:
            return result
        self.kg_map.confirm_arrival(
            resolved,
            observation=description,
            from_location=previous_location,
            action=action,
        )
        result["chosen_location_hint"] = resolved
        result["location_after_update"] = resolved
        result["registry_id"] = self.kg_map.registry_id_for(resolved)
        candidate = (auxiliary_gate or {}).get("action_transition_candidate", {})
        if isinstance(candidate, dict) and previous_location != resolved:
            candidate["from"] = previous_location
            candidate["to"] = resolved
        return result

    def _resolve_arrival_identity(self, title: str, description: str,
                                  action: str, from_location: str) -> tuple[str, dict]:
        title = canonical_room_display(title)
        candidates = self.kg_map.room_candidates(title)
        log = {
            "resolver_invoked": False,
            "resolver_decision": "",
            "resolver_confidence": "",
            "resolver_prompt": "",
            "resolver_raw_response": "",
        }
        if not config.LLM_LOCATION_RESOLVER:
            return self.kg_map.resolve_arrival_location(
                title, description, from_location, action
            ), log
        if not candidates:
            label, _ = self.kg_map.mint_room(
                title, description, epoch=self.current_epoch
            )
            log.update({"resolver_decision": "new", "resolver_confidence": "high"})
            return label, log
        signature = self.kg_map.description_signature(title, description)
        cache_key = (
            from_location or "", normalize_command_key(action),
            self.kg_map._base_location_key(title), signature,
        )
        cached = self._location_resolver_cache.get(cache_key)
        if cached:
            return cached["label"], {**log, **cached["log"], "resolver_cached": True}

        cards = self.kg_map.candidate_cards(title, limit=4)
        map_evidence = self.kg_map.known_edge_evidence(
            from_location, action, title
        )
        parsed = {}
        parse_error = ""
        for _ in range(2):
            try:
                body = self.llm.resolve_location_identity(
                    title=title,
                    description=description,
                    action=action,
                    from_location=from_location,
                    candidate_cards=cards,
                    map_evidence=map_evidence,
                )
                log["resolver_invoked"] = True
                log["resolver_prompt"] = self.llm.last_location_resolver_prompt or ""
                log["resolver_raw_response"] = (
                    self.llm.last_location_resolver_raw_response or ""
                )
                parsed, parse_error = self._parse_location_resolver_response(
                    body, [card["label"] for card in cards]
                )
                if not parse_error:
                    break
            except Exception as e:
                parse_error = str(e)
        confidence = self._clean_text(parsed.get("confidence", "low")).lower()
        decision = self._clean_text(parsed.get("decision", "new")).lower()
        if parse_error or confidence == "low" or decision != "existing":
            label, _ = self.kg_map.mint_room(
                title, description, epoch=self.current_epoch, force_new=True
            )
            decision = "new"
        else:
            label = parsed.get("match_label", "")
        log["resolver_decision"] = decision
        log["resolver_confidence"] = confidence or "low"
        if parse_error:
            log["resolver_error"] = parse_error
        self._location_resolver_cache[cache_key] = {"label": label, "log": dict(log)}
        return label, log

    def _parse_location_resolver_response(self, response_body: str,
                                          offered_labels: list[str]) -> tuple[dict, str]:
        body = str(response_body or "").strip()
        match = re.search(r"\|start\|\s*(.*?)\s*\|end\|", body, re.DOTALL)
        if match:
            body = match.group(1).strip()
        try:
            parsed = json.loads(body)
        except Exception:
            return {}, "resolver response was not valid JSON"
        decision = self._clean_text(parsed.get("decision", "")).lower()
        confidence = self._clean_text(parsed.get("confidence", "")).lower()
        label = self._clean_text(parsed.get("match_label", ""))
        if decision not in {"existing", "new"}:
            return {}, "resolver decision was invalid"
        if confidence not in {"high", "medium", "low"}:
            return {}, "resolver confidence was invalid"
        if decision == "existing" and label not in offered_labels:
            return {}, "resolver selected an unoffered candidate"
        return {
            "decision": decision,
            "match_label": label,
            "confidence": confidence,
            "reason": self._clean_text(parsed.get("reason", "")),
        }, ""

    def _ground_room_title(self, raw_title: str, observation: str,
                           look_probe_text: str) -> tuple[str, bool]:
        raw = str(raw_title or "").strip()
        if not raw:
            return "", False
        if raw in str(observation or "") or raw in str(look_probe_text or ""):
            return canonical_room_display(raw), True
        return "", False

    def _extract_probe_room_title(self, probe_text: str) -> str:
        return self._extract_observation_room_title(probe_text, scan_all_lines=True)

    def _observation_is_dark(self, observation: str) -> bool:
        text = str(observation or "").lower()
        return any(phrase in text for phrase in (
            "pitch black", "too dark to see", "can't see a thing",
            "cannot see a thing", "unable to see", "darkness",
        ))

    def _location_from_triples(self, triples: list) -> str:
        for subj, rel, obj in triples or []:
            if str(subj).strip().lower() == "you" and str(rel).strip().lower() == "in":
                return self._clean_text(obj)
        return ""

    def _first_known_location(self, *candidates: str) -> str:
        for candidate in candidates:
            text = self._clean_text(candidate)
            location_key = normalize_location_key(text)
            if text and location_key and location_key != "unknown":
                return text
        return "Starting Location"

    def _with_authoritative_location_triple(self, triples: list, location: str) -> list:
        """Replace/inject <You, in, location> when the observation title is clear."""
        cleaned_location = self._clean_text(location)
        if not cleaned_location:
            return list(triples or [])
        output = []
        replaced = False
        for subj, rel, obj in triples or []:
            if str(subj).strip().lower() == "you" and str(rel).strip().lower() == "in":
                output.append(("You", "in", cleaned_location))
                replaced = True
            else:
                output.append((subj, rel, obj))
        if not replaced:
            output.insert(0, ("You", "in", cleaned_location))
        return output

    def _extract_observation_room_title(self, observation: str,
                                        scan_all_lines: bool = False) -> str:
        """Conservative parser for IF room-title headers."""
        text = str(observation or "").strip()
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if scan_all_lines:
            for line in lines:
                if re.match(
                    r"^\s*[^A-Za-z0-9]+\s*[A-Za-z0-9].*[A-Za-z0-9]\s*[^A-Za-z0-9]+\s*$",
                    line,
                ):
                    candidate = canonical_room_display(line)
                    if candidate and len(candidate.split()) <= 8:
                        return candidate
        first_line = lines[0] if lines else ""
        if not first_line:
            return ""
        blocked_prefixes = (
            "you ", "i ", "it ", "that ", "there ", "taken", "dropped", "opened",
            "closed", "done", "ok", "okay", "nothing", "the ", "\"", "'",
        )
        lower_line = first_line.lower()
        if lower_line.startswith(blocked_prefixes):
            if not re.match(
                r"^the\s+[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,4}\b",
                first_line,
                flags=re.IGNORECASE,
            ):
                return ""

        candidate = first_line
        if len(candidate.split()) > 8 or candidate.endswith((".", "!", "?", ":")):
            candidate = ""
        if not candidate:
            match = re.match(
                r"^((?:The\s+)?[A-Z][A-Za-z0-9'-]*(?:\s+(?:of|in|on|at|to|and|the|a|an|[A-Z][A-Za-z0-9'-]*)){0,5})"
                r"(?=\s+(?:You|This|There|A|An|On|In|Here)\b)",
                first_line,
            )
            candidate = match.group(1).strip() if match else ""
        if not candidate:
            return ""
        words = candidate.split()
        if len(words) > 6:
            return ""
        connectors = {"of", "in", "on", "at", "to", "and", "the", "a", "an"}
        if not any(word[:1].isupper() for word in words):
            return ""
        for word in words:
            clean = re.sub(r"[^A-Za-z0-9'-]", "", word)
            if not clean:
                continue
            if clean.lower() in connectors:
                continue
            if not clean[:1].isupper():
                return ""
        return canonical_room_display(candidate)

    def _generate_command(self, observation: str, score: int,
                          experiences: str = None,
                          reset_affordance_cache: bool = False,
                          affordance_gate_decision: dict = None) -> tuple:
        """Generate the next command and return it with prompt metadata."""
        generation_started = time.perf_counter()
        timings: dict[str, float] = {}

        def record_timing(name: str, started_at: float):
            timings[name] = round(time.perf_counter() - started_at, 4)

        initial_generation = self.step_count == 0 and self.prev_action is None
        room_info = self.kg_map.get_current_room_info()
        current_objects = room_info.get("objects", [])
        current_objects_with_state = self._visible_objects_with_state_for_location(
            self.kg_map.current_location or "unknown"
        )
        current_state_snapshot = self._repetition_state_snapshot(
            location=self.kg_map.current_location or "unknown",
            observation=observation,
            visible_objects=current_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
        )
        current_state_key = self._snapshot_key(current_state_snapshot)
        command_history_context = self.attempt_ledger.format_room_block(
            self.kg_map.current_location or "unknown",
            current_state_key=current_state_key,
        )
        same_state_tried_commands = self.state_action_memory.commands_for_state(
            current_state_snapshot
        )
        same_state_tried_context = self.state_action_memory.format_for_prompt(
            current_state_snapshot
        )
        known_failed_here = self.failed_action_memory.format_for_prompt(
            self.kg_map.current_location or "unknown"
        )
        problem_attempts_here = self.attempt_ledger.format_problem_attempts_for_prompt(
            self.kg_map.current_location or "unknown"
        )
        retrieval_started = time.perf_counter()
        if experiences is None:
            if initial_generation:
                print("  Initial action generation: retrieving experiences...", flush=True)
            visible = ", ".join(str(obj) for obj in current_objects[:8])
            query = (
                f"Location: {self.kg_map.current_location}. "
                f"Visible: {visible}. Observation: {observation[:200]}"
            )
            experiences = self._retrieve_experiences_for_prompt(query)
            if initial_generation:
                print("  Initial action generation: experience retrieval done.", flush=True)
        record_timing("experience_retrieval_for_prompt", retrieval_started)
        stored_situations = self.situation_memory.format_for_prompt()
        action_space_context = self.action_space.to_prompt_string(current_objects)
        if initial_generation:
            print("  Initial action generation: running affordance brainstorm...", flush=True)
        brainstorm_started = time.perf_counter()
        affordance_result = self._brainstorm_affordances(
            observation=observation,
            current_objects=current_objects,
            current_objects_with_state=current_objects_with_state,
            stored_situations=self.situation_memory.active_situations(),
            known_failed_here=known_failed_here,
            problem_attempts_here=problem_attempts_here,
            command_history_context=command_history_context,
            same_state_tried_commands=same_state_tried_commands,
            experiences=experiences,
            score=score,
            reset_cache=reset_affordance_cache,
            gate_decision=affordance_gate_decision,
        )
        record_timing("affordance_brainstorming", brainstorm_started)
        if initial_generation:
            print(
                "  Initial action generation: affordance brainstorm "
                f"{affordance_result.get('status', 'done')}.",
                flush=True,
            )
        brainstormed_command_ideas = affordance_result.get("ideas_for_prompt", "[]")
        kg_map_context = self._prompt_kg_map_context()

        prompt = LPLH_ACTION_GENERATION_PROMPT.format(
            kg_map=kg_map_context,
            action_pairs=action_space_context,
            experiences=experiences,
            stored_situations=stored_situations,
            brainstormed_command_ideas=brainstormed_command_ideas,
            known_failed_commands_here=known_failed_here,
            problem_attempts_here=problem_attempts_here,
            command_history_here=command_history_context,
            same_state_tried_commands=same_state_tried_context,
            history=self._format_history(),
            history_length=config.HISTORY_LENGTH,
            score=score,
            observation=observation,
        )

        raw_llm_response = ""
        blocked_direction_guard = self._empty_navigation_guard_debug()
        main_llm_started = time.perf_counter()
        try:
            if initial_generation:
                print("  Initial action generation: asking main LLM for command...", flush=True)
            raw_llm_response = self.llm.chat(
                system_prompt="You are an expert player of text-based interactive fiction games.",
                user_prompt=prompt,
                think=True,
            )
            command = self._parse_command(raw_llm_response)
            repeat_check = self._parse_repeat_check(raw_llm_response)
            (
                command,
                raw_llm_response,
                repeat_check,
                blocked_direction_guard,
            ) = self._apply_navigation_enforcement(
                command=command,
                raw_llm_response=raw_llm_response,
                repeat_check=repeat_check,
                prompt=prompt,
                observation=observation,
                affordance_agenda=affordance_result.get("affordance_agenda", []),
            )
            attempted_before_here = self.attempt_ledger.count(
                self.kg_map.current_location or "unknown",
                command,
            )
            if initial_generation:
                print("  Initial action generation: main LLM returned.", flush=True)
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["connect", "refused", "unreachable", "failed to connect"]):
                raise RuntimeError(f"Ollama server unreachable: {e}") from e
            logger.error(f"Action generation failed: {e}")
            raw_llm_response = f"ERROR: {e}"
            command = "look"
            repeat_check = {}
            attempted_before_here = self.attempt_ledger.count(
                self.kg_map.current_location or "unknown",
                command,
            )
        record_timing("main_llm_action_selection", main_llm_started)
        timings["total"] = round(time.perf_counter() - generation_started, 4)

        generation = {
            "kg_map_context": kg_map_context,
            "action_space_context": action_space_context,
            "retrieved_experiences": experiences,
            "retrieval_debug": self.last_retrieval_debug,
            "stored_situations_context": stored_situations,
            "brainstormed_command_ideas": brainstormed_command_ideas,
            "affordance_agenda": brainstormed_command_ideas,
            "known_failed_commands_here": known_failed_here,
            "problem_attempts_here": problem_attempts_here,
            "command_history_here": command_history_context,
            "same_state_tried_commands": same_state_tried_context,
            "same_state_tried_command_list": same_state_tried_commands,
            "state_snapshot_at_generation": current_state_snapshot,
            "state_key_at_generation": current_state_key,
            "observation_at_generation": observation,
            "affordance_brainstorming": affordance_result,
            "llm_raw_response": raw_llm_response,
            "parsed_command": command,
            "blocked_direction_guard": blocked_direction_guard,
            "repeat_self_check": repeat_check,
            "attempt_count_before_command_here": attempted_before_here,
            "score_at_generation": score,
            "timings": timings,
        }
        return command, generation

    def _brainstorm_affordances(self, observation: str, current_objects: list,
                                current_objects_with_state: list,
                                stored_situations: list,
                                known_failed_here: str,
                                problem_attempts_here: str,
                                command_history_context: str,
                                same_state_tried_commands: list[str],
                                experiences: str,
                                score: int,
                                reset_cache: bool = False,
                                gate_decision: dict = None) -> dict:
        """Run LPLH2 affordance brainstorming for the next action prompt."""
        failure_context = self.affordance_brainstormer.failure_context(
            recent_failed_commands=self.recent_failed_actions,
            known_failed_commands_here=known_failed_here,
        )
        ledger_failure_context = self.affordance_brainstormer.failure_context(
            recent_failed_commands=[],
            known_failed_commands_here=problem_attempts_here,
        )
        state_signature = self.affordance_brainstormer.state_signature(
            location=self.kg_map.current_location or "unknown",
            visible_objects=current_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
            observation=observation,
        )
        unproductive_commands = self.affordance_brainstormer.unproductive_commands(
            location=self.kg_map.current_location or "unknown",
            state_signature=state_signature,
        )
        filtered_commands = (
            failure_context["failed_commands"]
            + ledger_failure_context["failed_commands"]
            + unproductive_commands
            + list(same_state_tried_commands or [])
        )
        recent_command_outcomes = self._recent_same_location_outcomes(
            self.kg_map.current_location or "unknown"
        )
        same_state_snapshot = self._repetition_state_snapshot(
            location=self.kg_map.current_location or "unknown",
            observation=observation,
            visible_objects=current_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
        )
        same_state_records = self.state_action_memory.records_for_state(
            same_state_snapshot
        )
        failed_records_here = self.failed_action_memory.records_for_location(
            self.kg_map.current_location or "unknown"
        )
        ledger_problem_records_here = self.attempt_ledger.problem_attempts_for_location(
            self.kg_map.current_location or "unknown"
        )
        attempt_counts_here = self.attempt_ledger.counts_for_location(
            self.kg_map.current_location or "unknown"
        )
        cache_status = self.affordance_brainstormer.cache_status(
            location=self.kg_map.current_location or "unknown",
            state_signature=state_signature,
        )
        result = {
            "status": "not_run",
            "location": self.kg_map.current_location or "unknown",
            "observation": observation,
            "visible_objects": list(current_objects_with_state or current_objects or []),
            "visible_object_names": list(current_objects or []),
            "inventory": list(self.kg_map.inventory),
            "recent_failed_commands": list(self.recent_failed_actions),
            "known_failed_commands_here": known_failed_here,
            "problem_attempts_here": problem_attempts_here,
            "command_history_context": command_history_context,
            "recent_command_outcomes": recent_command_outcomes,
            "failed_commands": filtered_commands,
            "unproductive_commands": unproductive_commands,
            "same_state_tried_commands": list(same_state_tried_commands or []),
            "same_state_tried_records": same_state_records,
            "failed_records_here": failed_records_here,
            "ledger_problem_records_here": ledger_problem_records_here,
            "attempt_counts_here": attempt_counts_here,
            "pending_carryover_commands": [],
            "failed_command_verbs": list(dict.fromkeys(
                failure_context["failed_verbs"] + ledger_failure_context["failed_verbs"]
            )),
            "active_situations": list(stored_situations or []),
            "action_space_context": "not_supplied",
            "score": score,
            "state_signature": state_signature,
            "reset_cache": reset_cache,
            "gate_decision": dict(gate_decision or {}),
            "gate_reason": (gate_decision or {}).get("reason", ""),
            "gate_focus": list((gate_decision or {}).get("focus", []) or []),
            "cached_ideas_available": 0,
            "cache_status": cache_status,
            "prompt": "",
            "llm_raw_response": "",
            "finish_reason": "",
            "response_body": "",
            "ideas": [],
            "fresh_ideas": [],
            "carried_ideas_before": [],
            "carried_ideas_after": [],
            "filtered_failed_commands": [],
            "affordance_agenda": [],
            "ideas_for_prompt": "[]",
            "error": "",
        }

        should_run = True
        if gate_decision is not None and "run" in gate_decision:
            should_run = self._coerce_gate_bool(gate_decision.get("run"), True)

        cached_ideas = []
        if reset_cache:
            self.affordance_brainstormer.merge_with_carryover(
                location=result["location"],
                fresh_ideas=[],
                failed_commands=filtered_commands,
                state_signature=state_signature,
                reset_cache=True,
                attempt_counts=attempt_counts_here,
            )
        else:
            cached_ideas = self.affordance_brainstormer.cached_ideas_for_state(
                location=result["location"],
                state_signature=state_signature,
                failed_commands=filtered_commands,
                attempt_counts=attempt_counts_here,
            )
        result["cached_ideas_available"] = len(cached_ideas)
        result["pending_carryover_commands"] = (
            self.affordance_brainstormer.pending_commands(cached_ideas)
        )

        if not should_run:
            result["ideas"] = cached_ideas
            result["carried_ideas_before"] = cached_ideas
            result["carried_ideas_after"] = cached_ideas
            result["filtered_failed_commands"] = list(dict.fromkeys(filtered_commands))
            if result["ideas"]:
                result["affordance_agenda"] = self.affordance_brainstormer.build_agenda(
                    result["ideas"],
                    tried_records=same_state_records,
                    failed_records=failed_records_here + ledger_problem_records_here,
                    attempt_counts=attempt_counts_here,
                )
                result["ideas_for_prompt"] = self.affordance_brainstormer.format_agenda_for_prompt(
                    result["affordance_agenda"]
                )
                result["status"] = "skipped_by_gate_with_cached_ideas"
            else:
                result["status"] = "skipped_by_gate_no_cached_ideas"
            return result

        try:
            response_body = self.llm.brainstorm_affordances(
                location=result["location"],
                observation=observation,
                visible_objects=result["visible_objects"],
                inventory=result["inventory"],
                recent_failed_commands=result["recent_failed_commands"],
                known_failed_commands_here=known_failed_here,
                problem_attempts_here=problem_attempts_here,
                command_history_here=command_history_context,
                recent_command_outcomes=recent_command_outcomes,
                failed_command_verbs=result["failed_command_verbs"],
                unproductive_commands_here=unproductive_commands,
                same_state_tried_commands=list(same_state_tried_commands or []),
                pending_carryover_commands=result["pending_carryover_commands"],
                stored_situations=result["active_situations"],
                experiences=experiences,
                score=score,
            )
            result["prompt"] = self.llm.last_affordance_prompt or ""
            result["llm_raw_response"] = self.llm.last_affordance_raw_response or ""
            result["finish_reason"] = self.llm.last_affordance_finish_reason or ""
            result["response_body"] = response_body

            ideas, parse_error = self.affordance_brainstormer.parse_response(response_body)
            fresh_ideas = [] if parse_error else ideas
            merge = self.affordance_brainstormer.merge_with_carryover(
                location=result["location"],
                fresh_ideas=fresh_ideas,
                failed_commands=filtered_commands,
                state_signature=state_signature,
                reset_cache=reset_cache,
                attempt_counts=attempt_counts_here,
            )
            result["fresh_ideas"] = merge["fresh_ideas"]
            result["carried_ideas_before"] = merge["carried_ideas_before"]
            result["carried_ideas_after"] = merge["carried_ideas_after"]
            result["filtered_failed_commands"] = merge["filtered_failed_commands"]
            result["ideas"] = merge["merged_ideas"]
            if result["ideas"]:
                result["affordance_agenda"] = self.affordance_brainstormer.build_agenda(
                result["ideas"],
                tried_records=same_state_records,
                failed_records=failed_records_here + ledger_problem_records_here,
                attempt_counts=attempt_counts_here,
            )
                result["ideas_for_prompt"] = self.affordance_brainstormer.format_agenda_for_prompt(
                    result["affordance_agenda"]
                )

            if parse_error:
                result["status"] = "carried_over_after_parse_error" if result["ideas"] else "parse_error"
                result["error"] = parse_error
            elif result["ideas"] and result["fresh_ideas"] and result["carried_ideas_before"]:
                result["status"] = "generated_with_carryover"
            elif result["ideas"] and result["fresh_ideas"]:
                result["status"] = "generated"
            elif result["ideas"]:
                result["status"] = "carried_over"
            elif ideas:
                result["status"] = "none_after_filter"
            else:
                result["status"] = "none"

        except Exception as e:
            logger.warning(f"Affordance brainstorming failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["prompt"] = self.llm.last_affordance_prompt or ""
            result["llm_raw_response"] = self.llm.last_affordance_raw_response or ""
            result["finish_reason"] = self.llm.last_affordance_finish_reason or ""

        return result

    def _run_auxiliary_gate(self, action: str, observation: str, done: bool,
                            score: int, reward_change: int, action_valid,
                            prev_location: str,
                            inventory_before: set = None,
                            visited_rooms_before: set = None,
                            look_probe_text: str = "") -> dict:
        """Use one aux LLM call to route selected expensive helper modules."""
        location = self.kg_map.current_location or "unknown"
        rooms_visited_before = sorted(str(room) for room in (visited_rooms_before or []))
        visible_objects = self._visible_objects_for_location(location)
        current_state_snapshot = self._repetition_state_snapshot(
            location=location,
            observation=observation,
            visible_objects=visible_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
        )
        same_state_tried_commands = self.state_action_memory.commands_for_state(
            current_state_snapshot
        )
        known_failed_here = self.failed_action_memory.format_for_prompt(location)
        problem_attempts_here = self.attempt_ledger.format_problem_attempts_for_prompt(location)
        recent_command_outcomes = self._recent_same_location_outcomes(location)
        recent_failed_for_gate = list(self.recent_failed_actions)
        if action_valid is False and action:
            recent_failed_for_gate.append(action)
        affordance_signature = self.affordance_brainstormer.state_signature(
            location=location,
            visible_objects=visible_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
            observation=observation,
        )
        failure_context = self.affordance_brainstormer.failure_context(
            recent_failed_commands=recent_failed_for_gate,
            known_failed_commands_here=known_failed_here,
        )
        ledger_failure_context = self.affordance_brainstormer.failure_context(
            recent_failed_commands=[],
            known_failed_commands_here=problem_attempts_here,
        )
        cached_affordance_ideas = self.affordance_brainstormer.cached_ideas_for_state(
            location=location,
            state_signature=affordance_signature,
            failed_commands=(
                failure_context["failed_commands"]
                + ledger_failure_context["failed_commands"]
                + list(same_state_tried_commands or [])
            ),
            attempt_counts=self.attempt_ledger.counts_for_location(location),
        )
        prev_lower = str(action or "").lower().strip()
        navigation_direction = self._blocked_direction_for_command(action)
        action_transition_candidate = {}
        if (action_valid is True
                and prev_location
                and location
                and location != prev_location
                and not navigation_direction):
            action_transition_candidate = {
                "from": prev_location,
                "command": action,
                "to": location,
                "source": "kg_location_change",
            }
        elif (not self._is_pure_rejected_observation(action_valid, observation)
              and prev_location
              and not navigation_direction):
            room_title = (
                self._extract_probe_room_title(look_probe_text)
                or self._extract_observation_room_title(
                    observation, scan_all_lines=True
                )
            )
            if room_title:
                current_key = normalize_location_key(location or "") or "unknown"
                title_key = normalize_location_key(room_title) or "unknown"
                if title_key and title_key != current_key:
                    action_transition_candidate = {
                        "from": prev_location,
                        "command": action,
                        "to": room_title,
                        "source": "observation_room_title",
                        "evidence": (
                            f"Observation begins with room title "
                            f"'{room_title}' while KG location remained '{location}'."
                        ),
                    }

        result = {
            "status": "not_run",
            "location": location,
            "previous_location": prev_location or "unknown",
            "action": action,
            "action_valid": action_valid,
            "observation": observation,
            "look_probe_text": look_probe_text,
            "done": bool(done),
            "score": score,
            "reward_change": reward_change,
            "rooms_visited_before": rooms_visited_before,
            "visible_objects": visible_objects,
            "inventory_before": sorted(str(item) for item in (inventory_before or [])),
            "inventory": list(self.kg_map.inventory),
            "active_situations": self.situation_memory.active_situations(),
            "recent_failed_commands": recent_failed_for_gate,
            "known_failed_commands_here": known_failed_here,
            "problem_attempts_here": problem_attempts_here,
            "recent_command_outcomes": recent_command_outcomes,
            "same_state_tried_commands": same_state_tried_commands,
            "action_transition_candidate": action_transition_candidate,
            "cached_affordance_ideas_available": len(cached_affordance_ideas),
            "prompt": "",
            "llm_raw_response": "",
            "finish_reason": "",
            "response_body": "",
            "decision": {},
            "environmental_change_detection": {},
            "use_legacy_environmental_detection": False,
            "use_legacy_summary_trigger_detection": False,
            "error": "",
        }

        try:
            response_body = self.llm.gate_auxiliary_modules(
                location=location,
                previous_location=prev_location or "unknown",
                action=action,
                action_valid=action_valid,
                observation=observation,
                done=bool(done),
                score=score,
                reward_change=reward_change,
                rooms_visited_before=rooms_visited_before,
                inventory_before=sorted(str(item) for item in (inventory_before or [])),
                inventory=list(self.kg_map.inventory),
                visible_objects=visible_objects,
                active_situations=result["active_situations"],
                recent_failed_commands=recent_failed_for_gate,
                known_failed_commands_here=known_failed_here,
                problem_attempts_here=problem_attempts_here,
                recent_command_outcomes=recent_command_outcomes,
                same_state_tried_commands=same_state_tried_commands,
                action_transition_candidate=action_transition_candidate,
                cached_affordance_ideas_available=len(cached_affordance_ideas),
                look_probe_text=look_probe_text,
            )
            result["prompt"] = self.llm.last_auxiliary_gate_prompt or ""
            result["llm_raw_response"] = self.llm.last_auxiliary_gate_raw_response or ""
            result["finish_reason"] = self.llm.last_auxiliary_gate_finish_reason or ""
            result["response_body"] = response_body

            parsed, parse_error = self._parse_auxiliary_gate_response(response_body)
            if parse_error:
                result["status"] = "parse_error"
                result["error"] = parse_error
                result["decision"] = self._fallback_auxiliary_gate_decision()
                result["use_legacy_environmental_detection"] = True
                result["use_legacy_summary_trigger_detection"] = True
            else:
                result["decision"] = self._normalize_auxiliary_gate_decision(
                    parsed=parsed,
                    action=action,
                    action_valid=action_valid,
                    observation=observation,
                )
                result["status"] = "routed"
        except Exception as e:
            logger.warning(f"Auxiliary module gate failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["prompt"] = self.llm.last_auxiliary_gate_prompt or ""
            result["llm_raw_response"] = self.llm.last_auxiliary_gate_raw_response or ""
            result["finish_reason"] = self.llm.last_auxiliary_gate_finish_reason or ""
            result["decision"] = self._fallback_auxiliary_gate_decision()
            result["use_legacy_environmental_detection"] = True
            result["use_legacy_summary_trigger_detection"] = True

        if not result.get("environmental_change_detection"):
            result["environmental_change_detection"] = (
                self._environmental_change_detail_from_gate(
                    gate_result=result,
                    action=action,
                    observation=observation,
                    action_valid=action_valid,
                )
            )
        return result

    def _apply_gate_inventory_update(self, auxiliary_gate: dict,
                                     inventory_before: set = None) -> dict:
        """Run dedicated inventory reconciliation when routed by the gate."""
        decision = (auxiliary_gate or {}).get("decision", {})
        route = decision.get("inventory_reconciliation", {})
        result = {
            "status": "noop",
            "gate_status": (auxiliary_gate or {}).get("status", "unknown"),
            "route": route,
            "raw_update": {},
            "before": list(self.kg_map.inventory),
            "after": list(self.kg_map.inventory),
            "applied": False,
            "reason": "",
            "prompt": "",
            "llm_raw_response": "",
            "finish_reason": "",
            "response_body": "",
            "error": "",
        }
        should_run = bool(route.get("run")) if isinstance(route, dict) else False
        if not should_run:
            result["reason"] = self._clean_text(route.get("reason", "")) if isinstance(route, dict) else ""
            return result
        try:
            response_body = self.llm.reconcile_inventory(
                location=self.kg_map.current_location or "unknown",
                action=(auxiliary_gate or {}).get("action", ""),
                action_valid=(auxiliary_gate or {}).get("action_valid"),
                command_outcome=decision.get("command_outcome", {}),
                observation=(auxiliary_gate or {}).get("observation", ""),
                inventory_before=sorted(str(item) for item in (inventory_before or [])),
                inventory=list(self.kg_map.inventory),
                visible_objects=self._visible_objects_for_location(
                    self.kg_map.current_location or "unknown"
                ),
                gate_reason=route.get("reason", "") if isinstance(route, dict) else "",
                gate_focus=route.get("focus", []) if isinstance(route, dict) else [],
            )
            result["prompt"] = self.llm.last_inventory_reconciliation_prompt or ""
            result["llm_raw_response"] = self.llm.last_inventory_reconciliation_raw_response or ""
            result["finish_reason"] = self.llm.last_inventory_reconciliation_finish_reason or ""
            result["response_body"] = response_body
            update, parse_error = self._parse_inventory_reconciliation_response(response_body)
            if parse_error:
                result["status"] = "parse_error"
                result["error"] = parse_error
                return result
            result["raw_update"] = update
            if not update.get("changed"):
                result["status"] = "no_inventory_change"
                result["reason"] = self._clean_text(update.get("reason", ""))
                return result
            applied = self.kg_map.apply_inventory_update(
                update,
                inventory_before=sorted(str(item) for item in (inventory_before or [])),
                action=(auxiliary_gate or {}).get("action", ""),
            )
            result.update(applied)
        except Exception as e:
            logger.warning(f"Inventory reconciliation failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["prompt"] = self.llm.last_inventory_reconciliation_prompt or ""
            result["llm_raw_response"] = self.llm.last_inventory_reconciliation_raw_response or ""
            result["finish_reason"] = self.llm.last_inventory_reconciliation_finish_reason or ""
        return result

    def _parse_inventory_reconciliation_response(self, response_body: str) -> tuple[dict, str]:
        body = str(response_body or "").strip()
        if not body:
            return {}, "empty response"
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", body, re.DOTALL)
        if m:
            body = m.group(1).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            start = body.find("{")
            end = body.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(body[start:end + 1])
                except json.JSONDecodeError:
                    return {}, "response was not valid JSON"
            else:
                return {}, "response was not valid JSON"
        if not isinstance(parsed, dict):
            return {}, "response did not contain an object"
        return self._normalize_gate_inventory_update(parsed), ""

    def _apply_gate_world_state_update(self, auxiliary_gate: dict) -> dict:
        """Run dedicated object/world-state extraction when routed by the gate."""
        decision = (auxiliary_gate or {}).get("decision", {})
        route = decision.get("world_state_extraction", {})
        result = {
            "status": "noop",
            "gate_status": (auxiliary_gate or {}).get("status", "unknown"),
            "route": route,
            "raw_update": {},
            "applied_update": {},
            "applied": False,
            "reason": "",
            "prompt": "",
            "llm_raw_response": "",
            "finish_reason": "",
            "response_body": "",
            "error": "",
        }
        should_run = bool(route.get("run")) if isinstance(route, dict) else False
        if not should_run:
            result["reason"] = self._clean_text(route.get("reason", "")) if isinstance(route, dict) else ""
            return result
        try:
            current_location = self.kg_map.current_location or "unknown"
            response_body = self.llm.extract_world_state_update(
                location=current_location,
                action=(auxiliary_gate or {}).get("action", ""),
                action_valid=(auxiliary_gate or {}).get("action_valid"),
                command_outcome=decision.get("command_outcome", {}),
                observation=(auxiliary_gate or {}).get("observation", ""),
                inventory=list(self.kg_map.inventory),
                visible_objects=self._visible_objects_for_location(current_location),
                current_room_state=self.kg_map.get_current_room_info(),
                gate_reason=route.get("reason", "") if isinstance(route, dict) else "",
                gate_focus=route.get("focus", []) if isinstance(route, dict) else [],
            )
            result["prompt"] = self.llm.last_world_state_extraction_prompt or ""
            result["llm_raw_response"] = self.llm.last_world_state_extraction_raw_response or ""
            result["finish_reason"] = self.llm.last_world_state_extraction_finish_reason or ""
            result["response_body"] = response_body
            update, parse_error = self._parse_world_state_update_response(response_body)
            if parse_error:
                result["status"] = "parse_error"
                result["error"] = parse_error
                return result
            result["raw_update"] = update
            if not update.get("changed"):
                result["status"] = "no_world_state_change"
                result["reason"] = self._clean_text(update.get("reason", ""))
                return result
            applied = self.kg_map.apply_world_state_update(
                update,
                default_location=current_location,
            )
            result["applied_update"] = applied
            result["applied"] = bool(applied.get("applied"))
            result["status"] = applied.get("status", "applied")
            result["reason"] = applied.get("reason", self._clean_text(update.get("reason", "")))
        except Exception as e:
            logger.warning(f"World-state extraction failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["prompt"] = self.llm.last_world_state_extraction_prompt or ""
            result["llm_raw_response"] = self.llm.last_world_state_extraction_raw_response or ""
            result["finish_reason"] = self.llm.last_world_state_extraction_finish_reason or ""
        return result

    def _apply_gate_action_transition(self, auxiliary_gate: dict) -> dict:
        """Record a non-cardinal action transition only if the aux gate approves."""
        decision = (auxiliary_gate or {}).get("decision", {})
        transition_decision = decision.get("kg_action_transition", {})
        candidate = (auxiliary_gate or {}).get("action_transition_candidate", {}) or {}
        result = {
            "status": "noop",
            "candidate": candidate,
            "decision": transition_decision,
            "applied": False,
            "reason": self._clean_text(transition_decision.get("reason", ""))
            if isinstance(transition_decision, dict) else "",
        }
        if not candidate:
            result["status"] = "no_candidate"
            return result
        if not isinstance(transition_decision, dict) or not transition_decision.get("record"):
            result["status"] = "gate_rejected"
            return result
        self.kg_map.confirm_action_transition(
            from_location=candidate.get("from", ""),
            action=candidate.get("command", ""),
            to_location=candidate.get("to", ""),
        )
        # If KG failed to apply a clear room-title arrival before the aux gate
        # reviewed it, the approved action transition is also authoritative for
        # current location. This repairs commands such as "go through window"
        # whose observation starts with the destination room title.
        self.kg_map.update(
            [("You", "in", candidate.get("to", ""))],
            candidate.get("command", ""),
        )
        result["applied"] = True
        result["status"] = "applied"
        return result

    def _parse_world_state_update_response(self, response_body: str) -> tuple[dict, str]:
        body = str(response_body or "").strip()
        if not body:
            return {}, "empty response"
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", body, re.DOTALL)
        if m:
            body = m.group(1).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            start = body.find("{")
            end = body.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(body[start:end + 1])
                except json.JSONDecodeError:
                    return {}, "response was not valid JSON"
            else:
                return {}, "response was not valid JSON"
        if not isinstance(parsed, dict):
            return {}, "response did not contain an object"
        return self._normalize_world_state_update(parsed), ""

    def _normalize_world_state_update(self, value: dict) -> dict:
        if not isinstance(value, dict):
            value = {}
        return {
            "changed": self._coerce_gate_bool(value.get("changed", False), False),
            "object_state_updates": self._clean_world_state_update_list(
                value.get("object_state_updates", [])
            ),
            "new_objects": self._clean_world_object_list(value.get("new_objects", [])),
            "removed_objects": self._clean_world_object_list(value.get("removed_objects", [])),
            "reason": self._clean_text(value.get("reason", "")),
        }

    def _clean_world_state_update_list(self, value) -> list[dict]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if not isinstance(item, dict):
                continue
            obj = self._clean_text(item.get("object", ""))
            state = self._clean_text(item.get("state", ""))
            loc = self._clean_text(item.get("location", ""))
            if not obj or not state:
                continue
            cleaned.append({
                "object": obj[:80],
                "location": loc[:120],
                "state": state[:120],
            })
        return cleaned[:8]

    def _clean_world_object_list(self, value) -> list:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                obj = self._clean_text(item.get("object", item.get("name", "")))
                loc = self._clean_text(item.get("location", ""))
                if obj:
                    cleaned.append({"object": obj[:80], "location": loc[:120]})
            else:
                obj = self._clean_text(str(item or ""))
                if obj:
                    cleaned.append(obj[:80])
        return cleaned[:8]

    def _parse_auxiliary_gate_response(self, response_body: str) -> tuple[dict, str]:
        body = str(response_body or "").strip()
        if not body:
            return {}, "empty response"
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", body, re.DOTALL)
        if m:
            body = m.group(1).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            start = body.find("{")
            end = body.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(body[start:end + 1])
                except json.JSONDecodeError:
                    return {}, "response was not valid JSON"
            else:
                return {}, "response was not valid JSON"
        if not isinstance(parsed, dict):
            return {}, "response did not contain an object"
        return parsed, ""

    def _normalize_terminal_outcome(self, value) -> str:
        text = self._normalize_event_piece(value)
        if text in {"defeat", "victory", "other", "none"}:
            return text
        if text in {"death", "dead", "lost", "lose", "failed", "failure", "bad ending"}:
            return "defeat"
        if text in {"win", "won", "success", "complete", "completed"}:
            return "victory"
        return "none"

    def _normalize_auxiliary_gate_decision(self, parsed: dict, action: str,
                                           action_valid, observation: str = "") -> dict:
        parsed = parsed if isinstance(parsed, dict) else {}
        note = self._clean_text(parsed.get("note", ""))
        focus = parsed.get("focus", [])
        if not isinstance(focus, list):
            focus = [str(focus)]
        compact_outcome = parsed.get("outcome")
        command_outcome_raw = parsed.get("command_outcome", {})
        if compact_outcome and not command_outcome_raw:
            command_outcome_raw = {
                "status": compact_outcome,
                "reason": note,
            }
        command_outcome = self._normalize_command_outcome(
            command_outcome_raw,
            action_valid=action_valid,
        )
        summary_raw = parsed.get("summary_triggers", {})
        compact_summary = parsed.get("summary", [])
        if not isinstance(compact_summary, list):
            compact_summary = []
        compact_summary_set = {
            self._normalize_event_piece(item) for item in compact_summary
        }
        if not isinstance(summary_raw, dict):
            summary_raw = {}

        navigation_summary = self._normalize_gate_summary_trigger(
            summary_raw.get(
                "navigation",
                parsed.get(
                    "navigation",
                    {"run": "navigation" in compact_summary_set, "evidence": note}
                    if compact_summary_set else {},
                ),
            ),
            default_run=False,
        )
        env_summary = self._normalize_gate_summary_trigger(
            summary_raw.get(
                "environmental",
                summary_raw.get(
                    "environmental_change",
                    parsed.get(
                        "environmental_change",
                        {"run": "environmental" in compact_summary_set, "evidence": note}
                        if compact_summary_set else {},
                    ),
                ),
            ),
            default_run=False,
        )
        narrative_summary = self._normalize_gate_summary_trigger(
            summary_raw.get(
                "narrative",
                parsed.get(
                    "narrative",
                    {"run": "narrative" in compact_summary_set, "evidence": note}
                    if compact_summary_set else {},
                ),
            ),
            default_run=False,
        )

        env_changed = bool(env_summary["run"])
        env_evidence = env_summary.get("evidence", "")
        command_accepted = command_outcome.get("status") == "accepted"
        if action_valid is not True or not command_accepted or self._is_movement_action(action):
            env_changed = False
            env_summary["run"] = False

        situation_raw = parsed.get(
            "stored_situation_detection",
            parsed.get(
                "situation_detection",
                {"run": parsed.get("situation"), "reason": note, "focus": focus}
                if "situation" in parsed else {},
            ),
        )
        affordance_raw = parsed.get(
            "affordance_brainstorming",
            {"run": parsed.get("brainstorm"), "reason": note, "focus": focus}
            if "brainstorm" in parsed else {},
        )
        inventory_route = self._normalize_gate_run_decision(
            parsed.get(
                "inventory_reconciliation",
                parsed.get(
                    "inventory_update",
                    {"run": parsed.get("inventory"), "reason": note, "focus": focus}
                    if "inventory" in parsed else {},
                ),
            ),
            default_run=False,
        )
        inventory_route = self._repair_inventory_route_if_needed(
            route=inventory_route,
            action=action,
            command_outcome=command_outcome,
            observation=observation,
        )
        return {
            "command_outcome": command_outcome,
            "terminal_outcome": self._normalize_terminal_outcome(
                parsed.get("terminal", parsed.get("terminal_outcome", "none"))
            ),
            "location_verdict": self._normalize_gate_location_verdict(
                parsed.get("location", parsed.get("location_verdict", {}))
            ),
            "environmental_change": {
                "changed": env_changed,
                "evidence": env_evidence,
            },
            "summary_triggers": {
                "navigation": navigation_summary,
                "environmental": env_summary,
                "narrative": narrative_summary,
            },
            "stored_situation_detection": self._normalize_gate_run_decision(
                situation_raw,
                default_run=True,
            ),
            "affordance_brainstorming": self._normalize_gate_run_decision(
                affordance_raw,
                default_run=True,
            ),
            "inventory_reconciliation": inventory_route,
            "world_state_extraction": self._normalize_gate_run_decision(
                parsed.get(
                    "world_state_extraction",
                    parsed.get(
                        "object_state_extraction",
                        {"run": parsed.get("world_state"), "reason": note, "focus": focus}
                        if "world_state" in parsed else {},
                    ),
                ),
                default_run=False,
            ),
            "kg_action_transition": self._normalize_gate_transition_decision(
                parsed.get(
                    "kg_action_transition",
                    parsed.get(
                        "action_transition",
                        {"record": parsed.get("transition"), "reason": note}
                        if "transition" in parsed else {},
                    ),
                )
            ),
            "inventory_update": self._normalize_gate_inventory_update(
                parsed.get("inventory_update", {}),
            ),
        }

    def _skipped_module_result(self, module_name: str, status: str) -> dict:
        return {
            "status": status,
            "module": module_name,
            "applied": False,
            "reason": "latest command was a clear parser/world rejection",
            "prompt": "",
            "llm_raw_response": "",
            "finish_reason": "",
            "response_body": "",
            "error": "",
        }

    def _is_pure_rejected_observation(self, action_valid, observation: str) -> bool:
        """Conservative cost gate for rejections with no extractable state.

        This never decides gameplay. It only avoids auxiliary extraction calls
        on observations that simply say the command/parser/world rejected the
        command. Mixed or informative failures are allowed through.
        """
        if action_valid is not False:
            return False
        text = self._normalize_event_piece(observation)
        if not text:
            return False

        informative_markers = (
            "but ",
            "however",
            "revealed",
            "appears",
            "opens",
            "closes",
            "locked",
            "unlocked",
            "already have",
            "you already have",
            "you don t have",
            "you don't have",
            "you do not have",
            "taken",
            "dropped",
            "inventory",
            "too heavy",
            "noticed",
            "notice",
            "pitch black",
            "grue",
            "dark",
            "score",
        )
        if any(marker in text for marker in informative_markers):
            return False

        pure_rejection_markers = (
            "i don t know the word",
            "i don't know the word",
            "i do not know the word",
            "you used the word",
            "in a way that i don't understand",
            "in a way that i don t understand",
            "you can t see any",
            "you can't see any",
            "you cannot see any",
            "you can t go that way",
            "you can't go that way",
            "you cannot go that way",
            "there is no way to go",
            "that sentence isn't one i recognize",
            "that sentence isn t one i recognize",
            "that sentence is not one i recognize",
            "i don't understand that sentence",
            "i don t understand that sentence",
            "i do not understand that sentence",
        )
        return any(marker in text for marker in pure_rejection_markers)

    def _repair_inventory_route_if_needed(self, route: dict, action: str,
                                          command_outcome: dict,
                                          observation: str = "") -> dict:
        """Fix self-contradictory gate routing for explicit inventory evidence."""
        if not isinstance(route, dict):
            route = {}
        if route.get("run"):
            return route

        outcome_status = str(command_outcome.get("status", "")).strip().lower()
        evidence = " ".join([
            str(action or ""),
            str(observation or ""),
            str(command_outcome.get("reason", "") or ""),
        ]).lower()

        accepted_inventory_markers = (
            "taken", "picked up", "acquired", "now carried", "already carried",
            "dropped", "eaten", "eats", "drunk", "drank", "given away",
            "gave", "lost", "stolen", "no longer carried",
        )
        repair_markers = (
            "you don't have that", "you do not have that",
            "already have that", "already have it",
        )
        should_repair = (
            outcome_status == "accepted"
            and any(marker in evidence for marker in accepted_inventory_markers)
        ) or any(marker in evidence for marker in repair_markers)

        if not should_repair:
            return route

        repaired = dict(route)
        repaired["run"] = True
        repaired["reason"] = (
            "Consistency repair: command outcome/observation contains explicit "
            "inventory evidence, so inventory reconciliation should run."
        )
        if not repaired.get("focus"):
            repaired["focus"] = self._inventory_focus_from_action(action)
        repaired["consistency_repair"] = True
        return repaired

    def _inventory_focus_from_action(self, action: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9']+", str(action or "").lower())
        if len(tokens) < 2:
            return []
        stop_words = {
            "to", "at", "with", "from", "into", "in", "on", "onto",
            "through", "using", "inside", "under", "over",
        }
        object_words = []
        for token in tokens[1:]:
            if token in stop_words:
                break
            object_words.append(token)
        focus = " ".join(object_words).strip()
        return [focus] if focus else []

    def _normalize_command_outcome(self, value, action_valid) -> dict:
        default_status = "accepted" if action_valid is True else "rejected"
        if action_valid is None:
            default_status = "unknown"
        if isinstance(value, str):
            status = value.strip().lower()
            reason = ""
        elif isinstance(value, dict):
            status = str(value.get("status", default_status)).strip().lower()
            reason = self._clean_text(value.get("reason", ""))
        else:
            status = default_status
            reason = ""
        aliases = {
            "success": "accepted",
            "successful": "accepted",
            "valid": "accepted",
            "failed": "rejected",
            "failure": "rejected",
            "invalid": "rejected",
            "parser_error": "rejected",
            "unchanged": "no_effect",
            "no effect": "no_effect",
            "none": "no_effect",
        }
        status = aliases.get(status, status)
        if status not in {"accepted", "rejected", "no_effect", "unknown"}:
            status = default_status
        return {"status": status, "reason": reason}

    def _normalize_gate_inventory_update(self, value) -> dict:
        if not isinstance(value, dict):
            value = {}
        return {
            "changed": self._coerce_gate_bool(value.get("changed", False), False),
            "authoritative": self._coerce_gate_bool(
                value.get("authoritative", value.get("authoritative_inventory", False)),
                False,
            ),
            "items_now_carried": self._clean_gate_item_list(value.get("items_now_carried", [])),
            "items_added": self._clean_gate_item_list(value.get("items_added", [])),
            "items_removed": self._clean_gate_item_list(value.get("items_removed", [])),
            "reason": self._clean_text(value.get("reason", "")),
        }

    def _clean_gate_item_list(self, value) -> list[str]:
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, list):
            raw_items = value
        else:
            raw_items = []
        cleaned = []
        for item in raw_items:
            text = self._clean_inventory_item(str(item or ""))
            if text and text.lower() not in [x.lower() for x in cleaned]:
                cleaned.append(text)
        return cleaned

    def _normalize_gate_summary_trigger(self, value, default_run: bool) -> dict:
        if isinstance(value, bool):
            return {"run": value, "reason": "", "evidence": ""}
        if not isinstance(value, dict):
            return {"run": default_run, "reason": "", "evidence": ""}
        evidence = self._clean_text(value.get("evidence", value.get("reason", "")))
        return {
            "run": self._coerce_gate_bool(
                value.get("run", value.get("trigger", value.get("changed", default_run))),
                default_run,
            ),
            "reason": self._clean_text(value.get("reason", evidence)),
            "evidence": evidence,
        }

    def _normalize_gate_run_decision(self, value, default_run: bool) -> dict:
        if isinstance(value, bool):
            return {"run": value, "reason": "", "focus": []}
        if not isinstance(value, dict):
            return {"run": default_run, "reason": "", "focus": []}
        focus = value.get("focus", [])
        if isinstance(focus, str):
            focus = [focus]
        elif not isinstance(focus, list):
            focus = []
        return {
            "run": self._coerce_gate_bool(value.get("run", default_run), default_run),
            "reason": self._clean_text(value.get("reason", "")),
            "focus": [
                self._clean_text(item)
                for item in focus
                if self._clean_text(item)
            ][:5],
        }

    def _normalize_gate_transition_decision(self, value) -> dict:
        if isinstance(value, bool):
            return {"record": value, "reason": ""}
        if not isinstance(value, dict):
            return {"record": False, "reason": ""}
        return {
            "record": self._coerce_gate_bool(value.get("record", False), False),
            "reason": self._clean_text(value.get("reason", "")),
        }

    def _fallback_auxiliary_gate_decision(self) -> dict:
        return {
            "command_outcome": {
                "status": "unknown",
                "reason": "Gate unavailable.",
            },
            "terminal_outcome": "none",
            "location_verdict": {
                "moved": "unclear",
                "room_title": "",
            },
            "environmental_change": {
                "changed": False,
                "evidence": "Gate unavailable; legacy environmental detector should run.",
            },
            "summary_triggers": {
                "navigation": {
                    "run": False,
                    "reason": "Gate unavailable; use legacy summary-trigger fallback.",
                    "evidence": "",
                },
                "environmental": {
                    "run": False,
                    "reason": "Gate unavailable; use legacy summary-trigger fallback.",
                    "evidence": "",
                },
                "narrative": {
                    "run": False,
                    "reason": "Gate unavailable; use legacy summary-trigger fallback.",
                    "evidence": "",
                },
            },
            "stored_situation_detection": {
                "run": True,
                "reason": "Gate unavailable; preserve previous situation detection behavior.",
                "focus": [],
            },
            "affordance_brainstorming": {
                "run": True,
                "reason": "Gate unavailable; preserve previous affordance brainstorming behavior.",
                "focus": [],
            },
            "inventory_reconciliation": {
                "run": False,
                "reason": "Gate unavailable.",
                "focus": [],
            },
            "world_state_extraction": {
                "run": False,
                "reason": "Gate unavailable.",
                "focus": [],
            },
            "kg_action_transition": {
                "record": False,
                "reason": "Gate unavailable.",
            },
            "inventory_update": {
                "changed": False,
                "authoritative": False,
                "items_now_carried": [],
                "items_added": [],
                "items_removed": [],
                "reason": "Gate unavailable.",
            },
        }

    def _normalize_gate_location_verdict(self, value) -> dict:
        if not isinstance(value, dict):
            value = {}
        moved = self._clean_text(value.get("moved", "unclear")).lower()
        if moved not in {"yes", "no", "unclear"}:
            moved = "unclear"
        return {
            "moved": moved,
            "room_title": self._clean_text(value.get("room_title", ""))[:160],
        }

    def _environmental_change_detail_from_gate(self, gate_result: dict,
                                               action: str, observation: str,
                                               action_valid) -> dict:
        env_decision = (
            gate_result.get("decision", {}).get("environmental_change", {})
        )
        command_outcome = (
            gate_result.get("decision", {}).get("command_outcome", {})
        )
        changed = bool(env_decision.get("changed"))
        if action_valid is not True:
            status = "skipped_invalid_action"
        elif command_outcome.get("status") not in {"accepted", None}:
            status = "skipped_rejected_by_gate"
            changed = False
        elif self._is_movement_action(action):
            status = "skipped_movement_action"
        else:
            status = "changed" if changed else "none"
        return {
            "status": status,
            "source": "auxiliary_gate",
            "location": self.kg_map.current_location or "unknown",
            "action": action,
            "observation": observation,
            "environmental_change": changed,
            "evidence": self._clean_text(env_decision.get("evidence", "")),
            "command_outcome": command_outcome,
            "prompt": gate_result.get("prompt", ""),
            "llm_raw_response": gate_result.get("llm_raw_response", ""),
            "response_body": gate_result.get("response_body", ""),
            "error": gate_result.get("error", ""),
        }

    def _skipped_situation_detection_result(self, action: str,
                                            just_resolved: list,
                                            gate_decision: dict) -> dict:
        active_before = self.situation_memory.active_situations()
        return {
            "status": "skipped_by_auxiliary_gate",
            "location": self.kg_map.current_location or "unknown",
            "action": action,
            "inventory": list(self.kg_map.inventory),
            "active_situations_before": active_before,
            "just_resolved_situations": just_resolved or [],
            "prompt": "",
            "llm_raw_response": "",
            "finish_reason": "",
            "response_body": "",
            "parsed_situation": None,
            "new_stored_situation": None,
            "active_situations_after": active_before,
            "gate_decision": dict(gate_decision or {}),
            "gate_reason": (gate_decision or {}).get("reason", ""),
            "error": "",
        }

    def _coerce_gate_bool(self, value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "run"}:
                return True
            if normalized in {"false", "no", "0", "skip"}:
                return False
        return bool(value)

    def _clean_text(self, value) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _recent_same_location_outcomes(self, location: str,
                                       max_items: int = 5) -> list[dict]:
        """Recent command->observation pairs issued from the same source room.

        The gate/brainstormer use this as advisory evidence for general
        condition-level problems where several different commands produce
        similarly distorted, blocked, or mismatched observations.
        """
        target = normalize_location_key(location or "unknown") or "unknown"
        outcomes: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add(command: str, observation: str, source_location: str):
            if not command or not observation:
                return
            if (normalize_location_key(source_location) or "unknown") != target:
                return
            clean_command = self._clean_text(command)[:80]
            clean_observation = self._clean_text(observation)[:180]
            if not clean_command or not clean_observation:
                return
            key = (clean_command.lower(), clean_observation.lower())
            if key in seen:
                return
            seen.add(key)
            outcomes.append({
                "command": clean_command,
                "observation": clean_observation,
            })

        for detail in self.step_details[-max_items * 4:]:
            if not isinstance(detail, dict):
                continue
            modules = detail.get("modules", {}) or {}
            generation = modules.get("action_generation", {}) or {}
            snapshot = generation.get("state_snapshot_at_generation", {}) or {}
            source_location = snapshot.get("location", "")
            command = detail.get("final_command") or generation.get("parsed_command", "")
            observation = detail.get("observation", "")
            add(command, observation, source_location)

        source_generation = self.pending_generation or {}
        source_snapshot = source_generation.get("state_snapshot_at_generation", {}) or {}
        if self.history:
            current_command, current_observation = self.history[-1]
            add(current_command, current_observation, source_snapshot.get("location", ""))

        return outcomes[-max_items:]

    def _world_signature(self, location: str, score: int) -> dict:
        """Compact state snapshot used to decide whether a failure may be stale."""
        loc = location or self.kg_map.current_location or "unknown"
        return {
            "location": loc,
            "inventory": list(self.kg_map.inventory),
            "visible_objects": self._visible_objects_for_location(loc),
            "score": score,
        }

    def _repetition_state_snapshot(self, location: str, observation: str,
                                   visible_objects: list, inventory: list,
                                   score: int) -> dict:
        """Exact state snapshot used for advisory repetition memory."""
        return self.state_action_memory.make_state_snapshot(
            location=location or "unknown",
            observation=observation or "",
            inventory=inventory or [],
            visible_objects=visible_objects or [],
            score=score,
        )

    def _parse_repetition_evaluation_response(self, response: str) -> tuple[dict, str]:
        body = str(response or "").strip()
        if not body:
            return {}, "empty response"

        candidates = [body]
        match = re.search(r"\{.*\}", body, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            remember = parsed.get("remember", False)
            if isinstance(remember, str):
                remember = remember.strip().lower() in {"true", "yes", "1"}
            return {
                "remember": bool(remember),
                "reason": re.sub(r"\s+", " ", str(parsed.get("reason", ""))).strip(),
            }, ""

        return {}, "could not parse repetition-evaluation JSON"

    def _snapshot_key(self, snapshot: dict) -> str:
        """Stable key for compact state snapshots used only for advisory matching."""
        try:
            return json.dumps(snapshot or {}, ensure_ascii=True, sort_keys=True)
        except Exception:
            return self._clean_text(snapshot)

    def _experience_kind_for_trigger(self, trigger: str) -> str:
        mapping = {
            "navigation": "route",
            "environmental": "state_change",
            "narrative": "clue",
            "error_correction": "syntax_lesson",
        }
        return mapping.get(str(trigger or "").strip().lower(), str(trigger or "memory"))

    def _retrieve_experiences_for_prompt(self, query: str) -> str:
        general_records = self.experience_lib.retrieve_relevant_structured(
            query,
            top_k=config.EXPERIENCE_FETCH_K,
            fetch_k=config.EXPERIENCE_FETCH_K,
        )
        achievement_records = self.experience_lib.retrieve_relevant_structured(
            query,
            top_k=3,
            fetch_k=max(6, config.EXPERIENCE_FETCH_K),
            where={"kind": "achievement"},
        )
        enabler_records = self.experience_lib.retrieve_relevant_structured(
            query,
            top_k=3,
            fetch_k=max(6, config.EXPERIENCE_FETCH_K),
            where={"kind": "enabler"},
        )
        death_records = self.experience_lib.retrieve_relevant_structured(
            query,
            top_k=3,
            fetch_k=max(6, config.EXPERIENCE_FETCH_K),
            where={"kind": "death_warning"},
        )
        records = self._merge_experience_records(
            death_records + enabler_records + achievement_records + general_records
        )
        if config.EXPERIENCE_RENDER_DIVERSITY:
            shown = self._select_diverse_experiences(records, config.EXPERIENCE_TOP_K)
        else:
            shown = [
                record for record in records
                if self._experience_location_relevant(record, include_neighbors=False)
            ][:config.EXPERIENCE_TOP_K]
        self.last_retrieval_debug = {
            "diversity_enabled": bool(config.EXPERIENCE_RENDER_DIVERSITY),
            "fetch_k": config.EXPERIENCE_FETCH_K,
            "top_k": config.EXPERIENCE_TOP_K,
            "selection_policy": "decision_relevance_cap_not_quota",
            "earned_score_event_keys_this_epoch": sorted(
                self.earned_score_event_keys_this_epoch
            ),
            "earned_score_location_reward_keys_this_epoch": sorted(
                self.earned_score_location_reward_keys_this_epoch
            ),
            "candidates": [self._experience_digest(record) for record in records],
            "shown": [self._experience_digest(record) for record in shown],
        }
        return self._format_experiences_for_prompt(shown, candidate_pool=records)

    def _select_diverse_experiences(self, records: list[dict],
                                    top_k: int) -> list[dict]:
        """Pick a compact, mixed retrieval set while preserving rank order."""
        if not records or top_k <= 0:
            return []

        selected = []
        selected_ids = set()

        def record_id(record: dict) -> str:
            return self._experience_record_id(record)

        def kind_of(record: dict) -> str:
            metadata = record.get("metadata") or {}
            return str(metadata.get("kind") or metadata.get("trigger") or "memory")

        def add(record: dict) -> bool:
            rid = record_id(record)
            if rid in selected_ids or len(selected) >= top_k:
                return False
            selected.append(record)
            selected_ids.add(rid)
            return True

        # Highest priority: relevant warning/death memory.
        for record in records:
            if len(selected) >= top_k:
                break
            if (kind_of(record) == "death_warning"
                    and self._warning_experience_relevant(record)):
                add(record)
                break

        # Then exact-room setup actions that enable an unearned reward.
        for record in records:
            if len(selected) >= top_k:
                break
            if (kind_of(record) == "enabler"
                    and not self._enabler_completed_this_epoch(record)
                    and self._experience_location_relevant(
                        record, include_neighbors=False
                    )):
                add(record)
                break

        # Then exact-room achievement not already earned this epoch.
        for record in records:
            if len(selected) >= top_k:
                break
            if (kind_of(record) == "achievement"
                    and not self._achievement_earned_this_epoch(record)
                    and self._experience_location_relevant(
                        record, include_neighbors=False
                    )):
                add(record)
                break

        # Local object-anchored mechanics, then object-anchored clues.
        for wanted_kind in ("state_change", "syntax_lesson", "clue"):
            if len(selected) >= top_k:
                break
            for record in records:
                if len(selected) >= top_k:
                    break
                if kind_of(record) != wanted_kind:
                    continue
                if self._object_anchored_experience(record):
                    add(record)
                    break

        # Filler: only novel routes. This is a cap, not a quota, so do not
        # pad with weak recent/non-route memories just to fill every slot.
        for record in records:
            if len(selected) >= top_k:
                break
            if (kind_of(record) == "route"
                    and self._route_for_current_room(record)
                    and not self._route_already_mapped(record)):
                add(record)
        return selected[:top_k]

    def _merge_experience_records(self, records: list[dict]) -> list[dict]:
        merged = []
        seen = set()
        for record in records or []:
            rid = self._experience_record_id(record)
            if rid in seen:
                continue
            merged.append(record)
            seen.add(rid)
        return merged

    def _experience_record_id(self, record: dict) -> str:
        metadata = record.get("metadata") or {}
        return str(metadata.get("event_key") or metadata.get("id")
                   or hashlib.sha1(str(record.get("text", "")).encode("utf-8")).hexdigest())

    def _achievement_earned_this_epoch(self, record: dict) -> bool:
        metadata = record.get("metadata") or {}
        key = str(metadata.get("event_key") or "")
        if key and key in self.earned_score_event_keys_this_epoch:
            return True
        return self._score_location_reward_from_metadata_earned(metadata)

    def _enabler_completed_this_epoch(self, record: dict) -> bool:
        metadata = record.get("metadata") or {}
        key = str(metadata.get("enables_event_key") or "")
        if key and key in self.earned_score_event_keys_this_epoch:
            return True
        reward = metadata.get("enables_reward")
        location = metadata.get("enables_location") or metadata.get("location")
        return self._score_location_reward_earned(
            location,
            reward,
            location_fingerprint=metadata.get("enables_location_fingerprint", ""),
        )

    def _score_location_reward_from_metadata_earned(self, metadata: dict) -> bool:
        reward = metadata.get("score_change")
        location = (
            metadata.get("location_issued")
            or metadata.get("location")
            or metadata.get("enables_location")
        )
        return self._score_location_reward_earned(
            location,
            reward,
            location_fingerprint=metadata.get("location_fingerprint", ""),
        )

    def _score_location_reward_earned(self, location, reward,
                                      location_fingerprint: str = "") -> bool:
        try:
            reward_int = int(reward)
        except Exception:
            return False
        key = self._score_location_reward_key(
            str(location or ""),
            reward_int,
            location_fingerprint=location_fingerprint,
        )
        return bool(key and key in self.earned_score_location_reward_keys_this_epoch)

    def _experience_location_relevant(self, record: dict,
                                      include_neighbors: bool = False) -> bool:
        """Match the physical issuing room; neighbors and destinations never qualify."""
        metadata = record.get("metadata") or {}
        current = self.kg_map.current_location or ""
        record_location = metadata.get("location_issued") or metadata.get("location") or ""
        if not current or not record_location:
            return False

        record_fingerprint = str(metadata.get("location_fingerprint") or "")
        record_registry_id = str(metadata.get("location_registry_id") or "")
        current_registry_id = self.kg_map.registry_id_for(current)
        if record_registry_id and current_registry_id:
            return record_registry_id == current_registry_id
        current_fingerprint = self._location_fingerprint_hash(current)
        if record_fingerprint and current_fingerprint:
            record_base = self._base_location_identity_key(record_location)
            current_base = self._base_location_identity_key(current)
            return (
                record_base == current_base
                and record_fingerprint == current_fingerprint
            )
        return normalize_location_key(record_location) == normalize_location_key(current)

    def _warning_experience_relevant(self, record: dict) -> bool:
        return self._experience_location_relevant(record, include_neighbors=False)

    def _object_anchored_experience(self, record: dict) -> bool:
        if not self._experience_location_relevant(record, include_neighbors=False):
            return False
        text = self._normalize_event_piece(record.get("text", ""))
        for obj in self._visible_objects_for_location(self.kg_map.current_location or "unknown"):
            obj_tokens = [
                tok for tok in self._normalize_event_piece(str(obj)).split()
                if len(tok) >= 3
            ]
            if obj_tokens and any(tok in text.split() for tok in obj_tokens):
                return True
        return False

    def _experience_digest(self, record: dict) -> dict:
        metadata = record.get("metadata") or {}
        return {
            "kind": metadata.get("kind") or metadata.get("trigger") or "memory",
            "trigger": metadata.get("trigger", ""),
            "location": metadata.get("location", ""),
            "location_issued": metadata.get("location_issued", ""),
            "location_fingerprint": metadata.get("location_fingerprint", ""),
            "prev_location": metadata.get("prev_location", ""),
            "action": metadata.get("action", ""),
            "score_change": metadata.get("score_change", ""),
            "enables_event_key": metadata.get("enables_event_key", ""),
            "enables_reward": metadata.get("enables_reward", ""),
            "enabler_action": metadata.get("enabler_action", ""),
            "epoch": metadata.get("epoch", ""),
            "step": metadata.get("step", ""),
            "distance": record.get("distance"),
            "already_in_current_map": self._route_already_mapped(record),
            "text_preview": re.sub(r"\s+", " ", str(record.get("text", ""))).strip()[:120],
        }

    def _format_experiences_for_prompt(self, records: list[dict],
                                       candidate_pool: list[dict] = None) -> str:
        if not records:
            return "No relevant experiences found yet."
        output = []
        for i, record in enumerate(records, 1):
            metadata = record.get("metadata") or {}
            kind = metadata.get("kind") or metadata.get("trigger") or "memory"
            trigger = metadata.get("trigger", "")
            location = metadata.get("location", "")
            action = metadata.get("action", "")
            score_change = metadata.get("score_change", "")
            stored_epoch = metadata.get("epoch", "")
            stored_step = metadata.get("step", "")
            header_bits = [f"kind={kind}"]
            if trigger:
                header_bits.append(f"trigger={trigger}")
            if location:
                header_bits.append(f"location={location}")
            if action:
                header_bits.append(f"action={action}")
            if score_change not in ("", None):
                header_bits.append(f"score_change={score_change}")
            if stored_epoch not in ("", None) and stored_step not in ("", None):
                try:
                    epoch_int = int(stored_epoch)
                    step_int = int(stored_step)
                    if epoch_int == int(self.current_epoch):
                        age = max(int(self.step_count) - step_int, 0)
                        header_bits.append(
                            f"stored=epoch {epoch_int}, step {step_int} ({age} steps ago)"
                        )
                    else:
                        header_bits.append(
                            f"stored=epoch {epoch_int}, step {step_int}"
                        )
                except Exception:
                    header_bits.append(
                        f"stored=epoch {stored_epoch}, step {stored_step}"
                    )
            elif stored_step not in ("", None):
                header_bits.append(f"stored=step {stored_step}")
            if kind == "achievement":
                if self._achievement_earned_this_epoch(record):
                    header_bits.append("already_earned_this_epoch=true")
                    header_bits.append("use_as=do_not_repeat_for_reward")
                else:
                    header_bits.append("not_earned_this_epoch=true")
                    header_bits.append("use_as=exact_room_reward_procedure")
                if metadata.get("current_score") not in ("", None):
                    header_bits.append(f"score_then={metadata.get('current_score')}")
                    header_bits.append(f"score_now={self.total_score}")
                if metadata.get("scoring_action"):
                    header_bits.append(f"exact_scoring_action={metadata.get('scoring_action')}")
                if metadata.get("location_after"):
                    header_bits.append(f"location_after={metadata.get('location_after')}")
            elif kind == "enabler":
                if self._enabler_completed_this_epoch(record):
                    header_bits.append("linked_reward_earned_this_epoch=true")
                    header_bits.append("use_as=already_completed_setup")
                else:
                    header_bits.append("linked_reward_not_earned_this_epoch=true")
                    header_bits.append("use_as=do_this_before_reward")
                if metadata.get("enabler_action"):
                    header_bits.append(f"exact_enabler_action={metadata.get('enabler_action')}")
                if metadata.get("enables_scoring_action"):
                    header_bits.append(f"enables_scoring_action={metadata.get('enables_scoring_action')}")
                if metadata.get("enables_reward") not in ("", None):
                    header_bits.append(f"enables_reward={metadata.get('enables_reward')}")
                if metadata.get("enables_location"):
                    header_bits.append(f"enables_location={metadata.get('enables_location')}")
            elif kind == "death_warning":
                header_bits.append("use_as=avoid_repeating_death")
                if metadata.get("fatal_action"):
                    header_bits.append(f"exact_fatal_action={metadata.get('fatal_action')}")
                room_details = self._death_warning_room_details(
                    record,
                    candidate_pool if candidate_pool is not None else records,
                )
                if room_details["also_fatal_here"]:
                    header_bits.append(
                        "also_fatal_here=" + ", ".join(room_details["also_fatal_here"])
                    )
                header_bits.append(
                    f"room_death_count={room_details['room_death_count']}"
                )
            elif kind == "terminal":
                header_bits.append("use_as=terminal_outcome_memory")
            elif kind == "route":
                header_bits.append("use_as=navigation_fact")
                if self._route_already_mapped(record):
                    header_bits.append("already_in_current_map=true")
                    header_bits.append("adds_nothing_beyond_map=true")
            elif kind == "syntax_lesson":
                header_bits.append("use_as=parser_syntax_lesson")
            elif kind == "state_change":
                header_bits.append("use_as=world_state_change")
            elif kind == "clue":
                header_bits.append("use_as=clue")
            output.append(
                f"Experience {i} [{'; '.join(str(bit) for bit in header_bits)}]:\n"
                f"{record.get('text', '')}"
            )
        return "\n\n".join(output)

    def _death_warning_room_details(self, record: dict,
                                    candidate_pool: list[dict]) -> dict:
        """Merge sibling fatal actions into one room-level retrieval header."""
        target_key = self._death_record_room_key(record)
        current_metadata = record.get("metadata") or {}
        current_action = normalize_command_key(
            current_metadata.get("fatal_action") or current_metadata.get("action")
        )
        sibling_actions: list[str] = []
        same_room_records = 0
        for candidate in candidate_pool or []:
            metadata = candidate.get("metadata") or {}
            kind = metadata.get("kind") or metadata.get("trigger")
            if kind != "death_warning" or self._death_record_room_key(candidate) != target_key:
                continue
            same_room_records += 1
            action = self._clean_text(
                metadata.get("fatal_action") or metadata.get("action")
            )
            action_key = normalize_command_key(action)
            if (
                action
                and action_key != current_action
                and action_key not in {normalize_command_key(item) for item in sibling_actions}
                and len(sibling_actions) < 5
            ):
                sibling_actions.append(action)
        death_counts = getattr(self, "_hazard_room_deaths", {})
        tracked_count = int(death_counts.get(target_key, 0))
        if not tracked_count:
            room_base = target_key.split("|", 1)[0]
            tracked_count = max(
                [
                    int(count)
                    for key, count in death_counts.items()
                    if str(key).split("|", 1)[0] == room_base
                ]
                or [0]
            )
        return {
            "also_fatal_here": sibling_actions,
            "room_death_count": max(tracked_count, same_room_records, 1),
        }

    def _death_record_room_key(self, record: dict) -> str:
        metadata = record.get("metadata") or {}
        issued = self._clean_text(
            metadata.get("location_issued") or metadata.get("location")
        )
        after = self._clean_text(metadata.get("location_after"))
        location = after if after and normalize_location_key(after) != normalize_location_key(issued) else issued
        fingerprint = ""
        if normalize_location_key(location) == normalize_location_key(
            metadata.get("location")
        ):
            fingerprint = self._clean_text(metadata.get("location_fingerprint"))
        if not fingerprint:
            fingerprint = self._location_fingerprint_hash(location)
        return self._hazard_room_identity_key(location, fingerprint)

    def _route_for_current_room(self, record: dict) -> bool:
        metadata = record.get("metadata") or {}
        if (metadata.get("kind") or metadata.get("trigger")) != "route":
            return False
        return self._experience_location_relevant(record, include_neighbors=False)

    def _route_already_mapped(self, record: dict) -> bool:
        metadata = record.get("metadata") or {}
        if (metadata.get("kind") or metadata.get("trigger")) != "route":
            return False
        prev_location = metadata.get("prev_location") or ""
        destination = metadata.get("location") or ""
        action = metadata.get("action") or ""
        if not prev_location or not destination or not action:
            return False

        prev_key = normalize_location_key(prev_location)
        dest_key = normalize_location_key(destination)
        action_key = normalize_command_key(action)
        for loc, data in self.kg_map.nodes.items():
            if normalize_location_key(loc) != prev_key:
                continue
            direction_edges = data.get("direction", {}) or {}
            for direction, mapped_dest in direction_edges.items():
                if (normalize_command_key(direction) == action_key
                        and normalize_location_key(mapped_dest) == dest_key):
                    return True
            action_edges = data.get("confirmed_actions", {}) or {}
            for mapped_action, mapped_dest in action_edges.items():
                if (normalize_command_key(mapped_action) == action_key
                        and normalize_location_key(mapped_dest) == dest_key):
                    return True
        return False

    def _visible_objects_for_location(self, location: str) -> list:
        if location and location in self.kg_map.nodes:
            return list(self.kg_map.nodes[location].get("have", []))
        room_info = self.kg_map.get_current_room_info()
        return list(room_info.get("objects", []))

    def _visible_objects_with_state_for_location(self, location: str) -> list:
        if location and location in self.kg_map.nodes:
            return self.kg_map._visible_objects_with_state(location)
        current_state = self.kg_map.to_clean_dict().get("current_room_state", {})
        return list(current_state.get("visible_objects", []))

    def _fallback_failure_reason(self, observation: str) -> str:
        obs = re.sub(r"\s+", " ", str(observation or "")).strip()
        if obs:
            return f"The game rejected this command: {obs[:180]}"
        return "The game rejected this command."

    def _clean_inventory_item(self, item: str) -> str:
        """Normalize simple inventory object text from successful player commands."""
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        return re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)

    def _detect_environmental_change_with_llm(self, action: str, observation: str,
                                              action_valid) -> dict:
        """Use the aux LLM to decide if a valid action changed world state."""
        result = {
            "status": "not_run",
            "location": self.kg_map.current_location or "unknown",
            "action": action,
            "observation": observation,
            "environmental_change": False,
            "evidence": "",
            "prompt": "",
            "llm_raw_response": "",
            "response_body": "",
            "error": "",
        }

        if action_valid is not True:
            result["status"] = "skipped_invalid_action"
            return result
        if self._is_movement_action(action):
            result["status"] = "skipped_movement_action"
            return result

        try:
            location = self.kg_map.current_location or "unknown"
            response_body = self.llm.detect_environmental_change(
                location=location,
                action=action,
                observation=observation,
                inventory=list(self.kg_map.inventory),
                visible_objects=self._visible_objects_for_location(location),
                active_situations=self.situation_memory.active_situations(),
            )
            result["prompt"] = self.llm.last_environmental_change_prompt or ""
            result["llm_raw_response"] = self.llm.last_environmental_change_raw_response or ""
            result["response_body"] = response_body
            parsed, parse_error = self._parse_environmental_change_response(response_body)
            if parse_error:
                result["status"] = "parse_error"
                result["error"] = parse_error
            else:
                result.update(parsed)
                result["status"] = "changed" if parsed["environmental_change"] else "none"
        except Exception as e:
            logger.warning(f"Environmental change detection failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["prompt"] = self.llm.last_environmental_change_prompt or ""
            result["llm_raw_response"] = self.llm.last_environmental_change_raw_response or ""

        return result

    def _parse_environmental_change_response(self, response_body: str) -> tuple[dict, str]:
        body = str(response_body or "").strip()
        if not body:
            return {"environmental_change": False, "evidence": ""}, "empty response"
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            normalized = re.sub(r"[^a-z]+", " ", body.lower())
            if "environmental change true" in normalized or normalized.strip() == "true":
                return {"environmental_change": True, "evidence": body[:180]}, ""
            if "environmental change false" in normalized or normalized.strip() == "false":
                return {"environmental_change": False, "evidence": ""}, ""
            return {"environmental_change": False, "evidence": ""}, "response was not valid JSON"

        if not isinstance(parsed, dict):
            return {"environmental_change": False, "evidence": ""}, "response did not contain an object"
        return {
            "environmental_change": bool(parsed.get("environmental_change")),
            "evidence": re.sub(r"\s+", " ", str(parsed.get("evidence", ""))).strip(),
        }, ""

    def _detect_neutral_triggers(self, observation: str, action_valid,
                                  visited_rooms_before: set,
                                  prev_location: str,
                                  environmental_change: dict = None,
                                  auxiliary_gate: dict = None) -> list:
        """Detect LPLH2 neutral-state experience triggers for this step."""
        if (auxiliary_gate
                and not auxiliary_gate.get("use_legacy_summary_trigger_detection")):
            return self._detect_neutral_triggers_from_gate(
                action_valid=action_valid,
                prev_location=prev_location,
                auxiliary_gate=auxiliary_gate,
            )

        return self._detect_neutral_triggers_legacy(
            observation=observation,
            action_valid=action_valid,
            visited_rooms_before=visited_rooms_before,
            prev_location=prev_location,
            environmental_change=environmental_change,
        )

    def _detect_neutral_triggers_from_gate(self, action_valid,
                                           prev_location: str,
                                           auxiliary_gate: dict) -> list:
        """Use the auxiliary gate's summary-trigger decisions."""
        triggers = []

        if action_valid is not True:
            return triggers
        command_outcome = (
            auxiliary_gate.get("decision", {})
            .get("command_outcome", {})
            .get("status", "unknown")
        )

        summary_triggers = (
            auxiliary_gate.get("decision", {}).get("summary_triggers", {})
        )
        if not isinstance(summary_triggers, dict):
            summary_triggers = {}

        for trigger_type in ("navigation", "environmental", "narrative"):
            trigger_decision = summary_triggers.get(trigger_type, {})
            if not isinstance(trigger_decision, dict):
                continue
            if trigger_decision.get("run"):
                if trigger_type in {"environmental"} and command_outcome != "accepted":
                    continue
                meta = {
                    "gate_reason": trigger_decision.get("reason", ""),
                    "gate_evidence": trigger_decision.get("evidence", ""),
                }
                if trigger_type == "navigation":
                    meta["prev_location"] = prev_location
                triggers.append((trigger_type, meta))

        # 4. Error correction: valid action after 2+ consecutive failures.
        if self.consecutive_failures >= 2 and command_outcome == "accepted":
            triggers.append(("error_correction", {
                "failed_attempts": list(self.recent_failed_actions),
            }))

        return triggers

    def _detect_neutral_triggers_legacy(self, observation: str, action_valid,
                                        visited_rooms_before: set,
                                        prev_location: str,
                                        environmental_change: dict = None) -> list:
        """Fallback neutral-trigger detection used only when the gate fails."""
        triggers = []

        new_location = self.kg_map.current_location
        if (action_valid is True
                and new_location
                and new_location not in visited_rooms_before
                and new_location != prev_location):
            triggers.append(("navigation", {"prev_location": prev_location}))

        if action_valid is not True:
            return triggers

        nav_fired = any(t == "navigation" for t, _ in triggers)
        environmental = (
            not nav_fired
            and bool((environmental_change or {}).get("environmental_change"))
        )
        narrative = self._is_narrative_action(self.prev_action) and self._is_informative(observation)
        if environmental:
            triggers.append(("environmental", {}))
        elif narrative:
            triggers.append(("narrative", {}))

        if self.consecutive_failures >= 2:
            triggers.append(("error_correction", {
                "failed_attempts": list(self.recent_failed_actions),
            }))

        return triggers

    def _detect_and_store_situation(self, action: str, observation: str,
                                    just_resolved: list = None) -> dict:
        """Run LPLH2 situation-memory detection for every completed step."""
        active_before = self.situation_memory.active_situations()
        just_resolved = just_resolved or []
        just_resolved_keys = {
            self.situation_memory.key_for(item)
            for item in just_resolved
        }
        result = {
            "status": "not_run",
            "location": self.kg_map.current_location or "unknown",
            "action": action,
            "inventory": list(self.kg_map.inventory),
            "active_situations_before": active_before,
            "just_resolved_situations": just_resolved,
            "prompt": "",
            "llm_raw_response": "",
            "finish_reason": "",
            "response_body": "",
            "parsed_situation": None,
            "new_stored_situation": None,
            "active_situations_after": active_before,
            "error": "",
        }

        try:
            response_body = self.llm.detect_stored_situation(
                location=result["location"],
                action=action,
                observation=observation,
                inventory=result["inventory"],
                stored_situations=active_before,
            )
            result["prompt"] = self.llm.last_situation_prompt or ""
            result["llm_raw_response"] = self.llm.last_situation_raw_response or ""
            result["finish_reason"] = self.llm.last_situation_finish_reason or ""
            result["response_body"] = response_body

            parsed, parse_error = self.situation_memory.parse_response(response_body)
            if parse_error:
                result["status"] = "parse_error"
                result["error"] = parse_error
            elif parsed is None:
                result["status"] = "none"
            else:
                result["parsed_situation"] = parsed
                if self.situation_memory.key_for(parsed) in just_resolved_keys:
                    result["status"] = "skipped_just_resolved"
                else:
                    stored, stored_situation = self.situation_memory.add(parsed)
                    if stored:
                        result["status"] = "stored"
                        result["new_stored_situation"] = stored_situation
                    else:
                        result["status"] = (
                            self.situation_memory.last_add_status or "duplicate"
                        )

        except Exception as e:
            logger.warning(f"Situation memory detection failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        result["active_situations_after"] = self.situation_memory.active_situations()
        return result

    def _resolve_stored_situations(self, action: str, observation: str,
                                   score: int, reward_change: int,
                                   action_valid, prev_location: str,
                                   environmental_change: dict = None) -> dict:
        """Ask the aux LLM whether any active stored situation is now solved."""
        active_before = self.situation_memory.active_situations()
        result = {
            "status": "not_run",
            "active_situations_before": active_before,
            "prompt": "",
            "llm_raw_response": "",
            "response_body": "",
            "parsed_resolved_situations": [],
            "removed_situations": [],
            "active_situations_after": active_before,
            "error": "",
        }

        if not active_before:
            result["status"] = "skipped_no_active_situations"
            return result

        location_changed = (
            bool(prev_location)
            and bool(self.kg_map.current_location)
            and self.kg_map.current_location != prev_location
        )
        worth_checking = (
            action_valid is True
            and (
                reward_change != 0
                or location_changed
                or bool((environmental_change or {}).get("environmental_change"))
            )
        )
        if not worth_checking:
            result["status"] = "skipped_no_resolution_signal"
            result["active_situations_after"] = self.situation_memory.active_situations()
            return result

        try:
            response_body = self.llm.resolve_stored_situations(
                location=self.kg_map.current_location or "unknown",
                action=action,
                observation=observation,
                inventory=list(self.kg_map.inventory),
                score=score,
                reward_change=reward_change,
                active_situations=active_before,
            )
            result["prompt"] = self.llm.last_situation_resolution_prompt or ""
            result["llm_raw_response"] = self.llm.last_situation_resolution_raw_response or ""
            result["response_body"] = response_body

            resolved, parse_error = self.situation_memory.parse_resolution_response(response_body)
            if parse_error:
                result["status"] = "parse_error"
                result["error"] = parse_error
            elif not resolved:
                result["status"] = "none"
            else:
                result["parsed_resolved_situations"] = resolved
                for situation in resolved:
                    if self.situation_memory.remove(situation):
                        result["removed_situations"].append(situation)
                result["status"] = "removed" if result["removed_situations"] else "none_removed"
        except Exception as e:
            logger.warning(f"Situation resolution failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["prompt"] = self.llm.last_situation_resolution_prompt or ""
            result["llm_raw_response"] = self.llm.last_situation_resolution_raw_response or ""

        result["active_situations_after"] = self.situation_memory.active_situations()
        return result

    def _ground_death_room_from_summary(self, summary: str,
                                        terminal_observation: str) -> tuple[str, dict]:
        """Accept a death-room title only when copied from terminal game text."""
        body = str(summary or "").strip()
        match = re.search(r"\{.*\}", body, re.DOTALL)
        try:
            parsed = json.loads(match.group(0) if match else body)
        except Exception:
            return "", {"status": "summary_not_json"}
        raw = self._clean_text(parsed.get("death_room_title", ""))
        if not raw:
            return "", {"status": "empty"}
        if raw not in str(terminal_observation or ""):
            return "", {"status": "ungrounded", "raw_title": raw}
        canonical = canonical_room_display(raw)
        return canonical, {
            "status": "grounded",
            "raw_title": raw,
            "canonical_title": canonical,
        }

    def _goal_hazard_context(self, action: str, action_valid,
                             location_issued: str,
                             location_after: str,
                             grounded_death_room: str = "") -> dict:
        """Resolve a repeated death's hazard room and its entry gateway."""
        issued = self._first_known_location(location_issued, "Starting Location")
        destination = ""
        hazard_registry_id = ""
        source = ""
        if grounded_death_room:
            candidates = self.kg_map.room_candidates(grounded_death_room)
            if len(candidates) == 1:
                destination = candidates[0]
                hazard_registry_id = self.kg_map.registry_id_for(destination)
                source = "grounded_death_room_registry_match"
            else:
                destination = canonical_room_display(grounded_death_room)
                source = (
                    "grounded_death_room_base_ambiguous"
                    if len(candidates) > 1 else "grounded_death_room_base_only"
                )
        destination = self._first_known_location(
            destination, location_after, issued
        )
        moved_to_destination = (
            bool(grounded_death_room)
            or self._is_location_change_action(action, True)
        ) and (
            normalize_location_key(destination)
            != normalize_location_key(issued)
        )
        if moved_to_destination:
            return {
                "hazard_location": destination,
                "hazard_fingerprint": (
                    hazard_registry_id
                    or self._location_fingerprint_hash(destination)
                    or f"base:{normalize_location_key(destination) or 'unknown'}"
                ),
                "hazard_registry_id": hazard_registry_id,
                "gateway": {
                    "room": issued,
                    "fingerprint": (
                        self.kg_map.registry_id_for(issued)
                        or self._location_fingerprint_hash(issued)
                    ),
                    "command": self._clean_text(action) or "unknown",
                },
                "source": source or "fatal_movement_destination",
            }
        return {
            "hazard_location": issued,
            "hazard_fingerprint": (
                self.kg_map.registry_id_for(issued)
                or self._location_fingerprint_hash(issued)
            ),
            "hazard_registry_id": self.kg_map.registry_id_for(issued),
            "gateway": None,
            "source": "fatal_action_issuing_room",
        }

    def _update_goal_visit_lifecycle(self, previous_location: str,
                                     current_location: str, done: bool,
                                     inventory: list) -> list[dict]:
        """Record goal-room entry or confirm survival after leaving it."""
        transitions: list[dict] = []
        previous = self._clean_text(previous_location)
        current = self._clean_text(current_location)
        if not previous or not current or previous == current:
            return transitions

        previous_goal = self.situation_memory.find_goal_for_room(
            previous,
            self._room_memory_identity(previous),
            open_only=True,
        )
        if previous_goal and not done:
            goal_id = previous_goal.get("goal_id", "")
            entry_inventory = self._goal_visit_inventory.pop(
                goal_id,
                list(inventory or []),
            )
            if self._inventory_matches_requirements(entry_inventory, previous_goal):
                result = self.situation_memory.confirm_goal(
                    goal_id,
                    entry_inventory,
                )
                transitions.append({
                    "type": "confirmed",
                    "location": previous,
                    "inventory_at_entry": list(entry_inventory),
                    **result,
                })
            else:
                survival = self.situation_memory.record_goal_survival(
                    goal_id,
                    entry_inventory,
                    epoch=self.current_epoch,
                    step=self.step_count,
                )
                transitions.append({
                    "type": "survived_unprepared",
                    "status": "goal_kept_open",
                    "goal_id": goal_id,
                    "location": previous,
                    "inventory_at_entry": list(entry_inventory),
                    "requires": list(previous_goal.get("requires", [])),
                    "survival_record": survival,
                })

        current_goal = self.situation_memory.find_goal_for_room(
            current,
            self._room_memory_identity(current),
            open_only=True,
        )
        if current_goal:
            goal_id = current_goal.get("goal_id", "")
            self._goal_visit_inventory[goal_id] = list(inventory or [])
            transitions.append({
                "type": "entered_goal_room",
                "goal_id": goal_id,
                "location": current,
                "inventory_at_entry": list(inventory or []),
            })
        return transitions

    def _handle_death_goal_lifecycle(self, event_key: str,
                                     hazard_location: str,
                                     hazard_fingerprint: str,
                                     fatal_action: str,
                                     death_observation: str,
                                     inventory_at_death: list,
                                     gateway: dict = None) -> tuple[dict, dict]:
        """Update one advisory preparation goal for every death in a room."""
        inventory = [self._clean_text(item) for item in inventory_at_death or []]
        inventory = [item for item in inventory if item]
        room_key = self._hazard_room_identity_key(
            hazard_location,
            hazard_fingerprint,
        )
        room_deaths = getattr(self, "_hazard_room_deaths", None)
        if room_deaths is None:
            self._hazard_room_deaths = {}
            room_deaths = self._hazard_room_deaths
        room_deaths[room_key] = int(room_deaths.get(room_key, 0)) + 1
        room_death_count = room_deaths[room_key]
        room_evidence = getattr(self, "_hazard_room_evidence", None)
        if room_evidence is None:
            self._hazard_room_evidence = {}
            room_evidence = self._hazard_room_evidence
        evidence_entries = room_evidence.setdefault(room_key, [])
        evidence = {
            "fatal_action": self._clean_text(fatal_action),
            "gateway": dict(gateway or {}),
            "hazard_text": self._clean_text(death_observation)[:320],
        }
        evidence_key = (
            normalize_command_key(evidence["fatal_action"]),
            normalize_location_key(evidence["gateway"].get("room", "")),
            normalize_command_key(evidence["gateway"].get("command", "")),
        )
        existing_evidence_keys = {
            (
                normalize_command_key(item.get("fatal_action", "")),
                normalize_location_key((item.get("gateway") or {}).get("room", "")),
                normalize_command_key((item.get("gateway") or {}).get("command", "")),
            )
            for item in evidence_entries
        }
        if evidence_key not in existing_evidence_keys:
            evidence_entries.append(evidence)
        existing = self.situation_memory.find_goal_for_room(
            hazard_location,
            hazard_fingerprint,
        )
        transition = {
            "type": "room_death_goal",
            "event_key": event_key,
            "hazard_location": hazard_location,
            "hazard_fingerprint": hazard_fingerprint,
            "hazard_room_key": room_key,
            "room_death_count": room_death_count,
            "fatal_action": fatal_action,
            "inventory_at_death": inventory,
            "gateway": gateway or {},
            "status": "not_run",
        }

        if existing and existing.get("declined"):
            transition["status"] = "skipped_declined"
            transition["goal_id"] = existing.get("goal_id", "")
            return transition, {}
        contradicted_confirmation = bool(
            existing and existing.get("status") == "confirmed"
        )
        if contradicted_confirmation:
            transition["confirmation_contradiction"] = (
                self.situation_memory.reopen_goal(existing.get("goal_id", ""))
            )

        previous_hypothesis = self._goal_hypothesis_snapshot(existing)
        previous_refutations = list((existing or {}).get("refutations", []))
        death_count = room_death_count
        new_evidence = True
        if existing:
            resolved_gateway = gateway or self._derive_goal_gateway(
                hazard_location, hazard_fingerprint, fatal_action
            )
            transition["evidence_merge"] = self.situation_memory.merge_goal_evidence(
                existing.get("goal_id", ""),
                fatal_action=fatal_action,
                gateway=resolved_gateway,
                hazard_text=self._clean_text(death_observation)[:320],
            )
            death_record = self.situation_memory.record_goal_death(
                existing.get("goal_id", ""),
                inventory,
                epoch=self.current_epoch,
                step=self.step_count,
            )
            transition["death_record"] = death_record
            death_count = max(
                room_death_count,
                int(death_record.get("deaths", existing.get("deaths", 1))),
            )
            new_evidence = bool(death_record.get("new_evidence"))
            if contradicted_confirmation:
                new_evidence = True
                transition["death_record"]["new_evidence"] = True
            if existing.get("status") == "avoid":
                transition["status"] = "avoid_recorded"
                return transition, {}

            if new_evidence and self._inventory_matches_requirements(inventory, existing):
                refutation = self.situation_memory.refute_goal(
                    existing.get("goal_id", ""),
                    inventory,
                    epoch=self.current_epoch,
                    step=self.step_count,
                )
                transition["refutation"] = refutation
                previous_refutations = list(existing.get("refutations", []))
            if not new_evidence:
                transition["status"] = "recorded_no_new_evidence"
                return transition, {}
        else:
            if room_death_count < 2:
                transition["status"] = "room_death_recorded_below_threshold"
                return transition, {}
            if self.situation_memory.open_goal_count() >= self.situation_memory.MAX_OPEN_GOALS:
                transition["status"] = "refused_cap"
                return transition, {}

        death_summary = self._death_summary_for_event(event_key)
        response_body = self.llm.hypothesize_precondition(
            death_count=death_count,
            fatal_action=fatal_action,
            hazard_location=hazard_location,
            death_observation=death_observation,
            death_summary=death_summary,
            inventory_at_death=inventory,
            previous_hypothesis=previous_hypothesis,
            previous_refutations=previous_refutations,
        )
        parsed, parse_error = self._parse_precondition_hypothesis(response_body)
        transition["hypothesis"] = parsed or {}
        transition["parse_error"] = parse_error or ""

        log_entry = {
            "state_type": "precondition_hypothesis",
            "prompt": getattr(self.llm, "last_precondition_prompt", "") or "",
            "summary": response_body,
            "raw_response": (
                getattr(self.llm, "last_precondition_raw_response", "") or ""
            ),
            "metadata": {
                "event_key": event_key,
                "hazard_location": hazard_location,
                "hazard_fingerprint": hazard_fingerprint,
                "fatal_action": fatal_action,
                "death_count": death_count,
                "inventory_at_death": inventory,
                "parse_error": parse_error or "",
            },
        }
        if parse_error or parsed is None:
            transition["status"] = "hypothesis_parse_error"
            return transition, log_entry

        hazard_text = self._clean_text(death_observation)[:320]
        if existing:
            goal_id = existing.get("goal_id", "")
            if parsed.get("preparable"):
                update = self.situation_memory.update_goal_hypothesis(
                    goal_id,
                    requires=parsed.get("requires", []),
                    item_keywords=parsed.get("item_keywords", []),
                    advice=parsed.get("advice", ""),
                    hazard_text=hazard_text,
                )
                transition["status"] = "updated"
                transition["goal_update"] = update
            else:
                decline = self.situation_memory.decline_goal(goal_id)
                transition["status"] = "declined"
                transition["goal_update"] = decline
            transition["goal_id"] = goal_id
            return transition, log_entry

        first_evidence = evidence_entries[0] if evidence_entries else evidence
        resolved_gateway = first_evidence.get("gateway") or gateway or self._derive_goal_gateway(
            hazard_location, hazard_fingerprint, fatal_action
        )
        created, goal, status = self.situation_memory.add_goal_situation(
            hazard_location=hazard_location,
            hazard_fingerprint=hazard_fingerprint,
            fatal_action=first_evidence.get("fatal_action") or fatal_action,
            gateway=resolved_gateway,
            hazard_text=hazard_text,
            requires=parsed.get("requires", []),
            item_keywords=parsed.get("item_keywords", []),
            advice=parsed.get("advice", ""),
            last_death_inventory=inventory,
            created_epoch=self.current_epoch,
            deaths=death_count,
        )
        if goal:
            for prior_evidence in evidence_entries[1:]:
                self.situation_memory.merge_goal_evidence(
                    goal.get("goal_id", ""),
                    fatal_action=prior_evidence.get("fatal_action", ""),
                    gateway=prior_evidence.get("gateway") or None,
                    hazard_text=prior_evidence.get("hazard_text", ""),
                )
            goal = self.situation_memory.find_goal_for_room(
                hazard_location,
                hazard_fingerprint,
            ) or goal
        transition.update({
            "status": status,
            "created": created,
            "goal": goal or {},
            "goal_id": (goal or {}).get("goal_id", ""),
            "gateway": resolved_gateway,
        })
        if goal and not parsed.get("preparable"):
            transition["goal_update"] = self.situation_memory.decline_goal(
                goal.get("goal_id", "")
            )
            transition["status"] = "declined"
        elif goal and self._inventory_matches_requirements(inventory, goal):
            transition["refutation"] = self.situation_memory.refute_goal(
                goal.get("goal_id", ""),
                inventory,
                epoch=self.current_epoch,
                step=self.step_count,
            )
        return transition, log_entry

    def _maybe_handle_repeated_death_goal(self, duplicate_death: bool,
                                          **kwargs) -> tuple[dict, dict]:
        """Compatibility wrapper; room-level counts now determine the threshold."""
        del duplicate_death
        return self._handle_death_goal_lifecycle(**kwargs)

    def _parse_precondition_hypothesis(self, response_body: str) -> tuple[dict, str]:
        text = str(response_body or "").strip()
        if not text:
            return None, "empty response"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"
        if not isinstance(parsed, dict) or not isinstance(parsed.get("preparable"), bool):
            return None, "response missing boolean preparable"
        requires = parsed.get("requires", [])
        if isinstance(requires, str):
            requires = [requires]
        if not isinstance(requires, list):
            return None, "requires must be a list"
        item_keywords = parsed.get("item_keywords", [])
        if isinstance(item_keywords, str):
            item_keywords = [item_keywords]
        if not isinstance(item_keywords, list):
            return None, "item_keywords must be a list"
        return {
            "preparable": parsed["preparable"],
            "requires": [
                self._clean_text(item) for item in requires
                if self._clean_text(item)
            ],
            "item_keywords": [
                self._clean_text(item) for item in item_keywords
                if self._clean_text(item)
            ],
            "reason": self._clean_text(parsed.get("reason", "")),
            "advice": self._clean_text(parsed.get("advice", "")),
        }, ""

    def _goal_hypothesis_snapshot(self, goal: dict) -> dict:
        if not goal:
            return {}
        return {
            "goal_id": goal.get("goal_id", ""),
            "requires": list(goal.get("requires", [])),
            "item_keywords": list(goal.get("item_keywords", [])),
            "advice": goal.get("advice", ""),
            "status": goal.get("status", "untested"),
            "deaths": int(goal.get("deaths", 0)),
        }

    def _inventory_matches_requirements(self, inventory: list, goal: dict) -> bool:
        inventory_tokens = {
            token
            for item in inventory or []
            for token in re.findall(r"[a-z0-9]+", str(item).lower())
            if len(token) > 2
        }
        anchors = list((goal or {}).get("item_keywords", []))
        if not anchors:
            anchors = list((goal or {}).get("requires", []))
        requirement_tokens = {
            token
            for item in anchors
            for token in re.findall(r"[a-z0-9]+", str(item).lower())
            if len(token) > 2 and token not in {"other", "item", "thing", "suitable"}
        }
        return bool(inventory_tokens & requirement_tokens)

    def _hazard_room_identity_key(self, location: str, fingerprint: str = "") -> str:
        base, resolved_fingerprint = self._event_location_identity(
            location,
            location_fingerprint=fingerprint,
        )
        return f"{base}|{resolved_fingerprint or 'no-fingerprint'}"

    def _derive_goal_gateway(self, hazard_location: str,
                             hazard_fingerprint: str,
                             fatal_action: str) -> dict:
        for outcome in reversed(self._recent_outcomes):
            destination = self._clean_text(outcome.get("destination", ""))
            if not destination:
                continue
            destination_fingerprint = self._location_fingerprint_hash(destination)
            same_room = (
                destination_fingerprint == hazard_fingerprint
                if destination_fingerprint and hazard_fingerprint
                else self._base_location_identity_key(destination)
                == self._base_location_identity_key(hazard_location)
            )
            if same_room and outcome.get("outcome_class") in {"moved", "scored"}:
                source = self._clean_text(outcome.get("location", "")) or "unknown"
                return {
                    "room": source,
                    "fingerprint": self._location_fingerprint_hash(source),
                    "command": self._clean_text(outcome.get("command", "")) or "unknown",
                }
        return {
            "room": hazard_location or "unknown",
            "fingerprint": hazard_fingerprint or "",
            "command": fatal_action or "unknown",
        }

    def _death_summary_for_event(self, event_key: str) -> str:
        try:
            records = self.experience_lib.retrieve_relevant_structured(
                query=f"death warning {event_key}",
                top_k=1,
                fetch_k=1,
                where={"event_key": event_key},
            )
            if records:
                return str(records[0].get("text", ""))
        except Exception as exc:
            logger.debug(f"Could not retrieve stored death summary: {exc}")
        return ""

    def _neutral_event_key(self, trigger: str, action: str, observation: str,
                           location: str, prev_location: str = None,
                           failed_attempts: list = None) -> str:
        """Build a stable key for neutral-summary dedup across epochs."""
        trigger_norm = self._normalize_event_piece(trigger)
        action_norm = normalize_command_key(action) or "unknown"
        location_base, location_identity = self._event_location_identity(location)
        location_norm = (
            f"{location_base}:{location_identity}"
            if location_identity else location_base
        )

        if trigger_norm == "navigation":
            prev_base, prev_identity = self._event_location_identity(
                prev_location or "unknown"
            )
            prev_norm = (
                f"{prev_base}:{prev_identity}" if prev_identity else prev_base
            )
            return f"neutral:v1:navigation:{prev_norm}:{action_norm}:{location_norm}"

        if trigger_norm in {"narrative", "environmental"}:
            obs_sig = self._event_text_signature(observation)
            return f"neutral:v1:{trigger_norm}:{location_norm}:{action_norm}:{obs_sig}"

        if trigger_norm == "error_correction":
            failed_norm = " > ".join(
                normalize_command_key(a) or "unknown" for a in (failed_attempts or [])
            )
            failed_sig = self._event_text_signature(failed_norm)
            return f"neutral:v1:error_correction:{location_norm}:{action_norm}:{failed_sig}"

        obs_sig = self._event_text_signature(observation)
        return f"neutral:v1:{trigger_norm}:{location_norm}:{action_norm}:{obs_sig}"

    def _location_fingerprint_hash(self, location: str) -> str:
        """Return the stable first-sentence fingerprint hash for a KG room."""
        kg_map = getattr(self, "kg_map", None)
        if kg_map is None:
            return ""
        location_text = self._clean_text(location)
        fingerprints = getattr(kg_map, "room_fingerprints", {}) or {}
        fingerprint = fingerprints.get(location_text, "")
        if not fingerprint:
            try:
                canonical = kg_map._canonicalize_known_location(location_text)
            except Exception:
                canonical = location_text
            fingerprint = fingerprints.get(canonical, "")
        if not fingerprint:
            wanted = canonical_room_display(location_text)
            fingerprint = next(
                (
                    value for key, value in fingerprints.items()
                    if canonical_room_display(key) == wanted and value
                ),
                "",
            )
        if not fingerprint:
            return ""
        return hashlib.sha1(str(fingerprint).encode("utf-8")).hexdigest()[:12]

    def _room_memory_identity(self, location: str) -> str:
        return (
            self.kg_map.registry_id_for(location)
            or self._location_fingerprint_hash(location)
        )

    def _base_location_identity_key(self, location: str) -> str:
        """Normalize a room title without its visit-order-dependent #N suffix."""
        base = re.sub(r"\s+#\d+\s*$", "", self._clean_text(location))
        return normalize_location_key(base) or "unknown"

    def _event_location_identity(self, location: str,
                                 location_fingerprint: str = None) -> tuple[str, str]:
        """Return a stable location key plus fingerprint for event identities."""
        kg_map = getattr(self, "kg_map", None)
        registry_id = kg_map.registry_id_for(location) if kg_map is not None else ""
        if registry_id:
            return self._base_location_identity_key(location), registry_id
        fingerprint = (
            self._location_fingerprint_hash(location)
            if location_fingerprint is None
            else str(location_fingerprint or "")
        )
        if fingerprint:
            return self._base_location_identity_key(location), fingerprint
        return normalize_location_key(location) or "unknown", ""

    def _score_event_key(self, trigger: str, action: str, location: str,
                         reward_change: int,
                         location_fingerprint: str = None) -> str:
        """Build a stable key for score-gain summary dedup across epochs."""
        trigger_norm = self._normalize_event_piece(trigger)
        action_norm = normalize_command_key(action) or "unknown"
        location_norm, fingerprint = self._event_location_identity(
            location,
            location_fingerprint,
        )
        return (
            f"score:v1:{trigger_norm}:{location_norm}:{fingerprint}:"
            f"{action_norm}:{int(reward_change or 0)}"
        )

    def _death_event_key(self, action: str, location: str,
                         location_fingerprint: str = None) -> str:
        """Build a stable key for fatal-action summary dedup across epochs."""
        action_norm = normalize_command_key(action) or "unknown"
        location_norm, fingerprint = self._event_location_identity(
            location,
            location_fingerprint,
        )
        return f"death:v1:{location_norm}:{fingerprint}:{action_norm}"

    def _classify_death(self, done: bool, reward_change: int,
                        terminal_status: str, observation: str) -> bool:
        """Return True when the latest step should become death-warning memory."""
        if reward_change < 0:
            return True
        if not done:
            return False
        terminal_norm = self._normalize_terminal_outcome(terminal_status)
        if terminal_norm == "defeat":
            return True
        if terminal_norm == "victory":
            return False
        text = str(observation or "")
        return bool(re.search(
            r"(\*{2,}\s*you have died\s*\*{2,}|you have died|you are dead|"
            r"you have lost|game over|you were killed|you are killed)",
            text,
            re.IGNORECASE,
        ))

    def _score_location_reward_key(self, location: str, reward_change: int,
                                   location_fingerprint: str = None) -> str:
        location_norm, fingerprint = self._event_location_identity(
            location,
            location_fingerprint,
        )
        return (
            f"score_location_reward:v1:{location_norm}:{fingerprint}:"
            f"{int(reward_change or 0)}"
        )

    def _store_reward_enabler_experiences(self, score_event_key: str,
                                          reward_change: int,
                                          scoring_action: str,
                                          scoring_location: str,
                                          location_after: str) -> list[dict]:
        """Store recent state-changing setup actions that preceded a reward."""
        if not score_event_key or reward_change <= 0:
            return []

        candidates = [
            item for item in self._recent_outcomes[-5:]
            if item.get("outcome_class") == "state_change"
        ][-2:]
        log_entries: list[dict] = []
        for item in candidates:
            enabler_action = self._clean_text(item.get("command", ""))
            enabler_location = self._clean_text(item.get("location", ""))
            if not enabler_action or not enabler_location:
                continue
            enabler_location_fingerprint = self._location_fingerprint_hash(
                enabler_location
            )
            scoring_location_fingerprint = self._location_fingerprint_hash(
                scoring_location
            )
            enabler_location_key, enabler_fingerprint_key = (
                self._event_location_identity(
                    enabler_location,
                    enabler_location_fingerprint,
                )
            )

            enabler_event_key = (
                "enabler:v1:"
                f"{enabler_location_key}:{enabler_fingerprint_key}:"
                f"{normalize_command_key(enabler_action)}:"
                f"{score_event_key}"
            )
            if self.experience_lib.event_seen(enabler_event_key):
                continue

            steps_before = max(
                int(self.step_count or 0) - int(item.get("step", self.step_count) or 0),
                1,
            )
            obs_excerpt = self._clean_text(item.get("observation", ""))[:240]
            experience_text = (
                f"In <loc>{enabler_location}</loc>, "
                f"<step>{enabler_action}</step> changed the game state"
            )
            if obs_excerpt:
                experience_text += f" (observation: \"{obs_excerpt}\")"
            experience_text += (
                f". {steps_before} step(s) later, "
                f"<step>{scoring_action}</step> at <loc>{scoring_location}</loc> "
                f"earned {int(reward_change):+d} point(s). Complete this setup "
                "before trying to earn that linked reward again."
            )

            metadata = {
                "kind": "enabler",
                "trigger": "score_enabler",
                "event_key": enabler_event_key,
                "location": enabler_location,
                "location_issued": enabler_location,
                "location_fingerprint": enabler_location_fingerprint,
                "location_registry_id": self.kg_map.registry_id_for(
                    enabler_location
                ),
                "action": enabler_action,
                "enabler_action": enabler_action,
                "enables_event_key": score_event_key,
                "enables_reward": int(reward_change),
                "enables_scoring_action": scoring_action,
                "enables_location": scoring_location,
                "enables_location_fingerprint": scoring_location_fingerprint,
                "enables_location_registry_id": self.kg_map.registry_id_for(
                    scoring_location
                ),
                "location_after": location_after,
                "steps_before_reward": int(steps_before),
                "epoch": int(self.current_epoch),
                "step": int(item.get("step", self.step_count) or self.step_count),
            }
            self.experience_lib.store_experience(
                experience_text=experience_text,
                metadata=metadata,
            )
            self.experience_lib.record_event(
                enabler_event_key,
                metadata={**metadata, "status": "stored"},
            )
            log_entries.append({
                "state_type": "score_enabler",
                "prompt": "template_from_recent_state_change_window",
                "summary": experience_text,
                "raw_response": "",
                "metadata": metadata,
            })
        return log_entries

    def _validate_score_summary(self, summary: str, history_text: str,
                                reward_change: int, current_score: int,
                                scoring_action: str, location_issued: str,
                                location_after: str) -> tuple[str, str]:
        """Ensure achievement summaries include the authoritative score facts."""
        if reward_change <= 0:
            return summary, "not_applicable"

        def contains_facts(text: str) -> bool:
            normalized = self._normalize_event_piece(text)
            action_norm = normalize_command_key(scoring_action) or "unknown"
            location_norm = normalize_location_key(location_issued) or "unknown"
            return action_norm in normalized and location_norm in normalized

        if contains_facts(summary):
            return summary, "ok"

        retry_note = (
            "Your previous summary omitted or contradicted the authoritative "
            f"scoring facts. Rewrite it so the exact scoring_action "
            f"'{scoring_action}' and scoring_location '{location_issued}' both "
            "appear clearly. Do not attribute the reward to another command."
        )
        retry_summary = self.llm.summarize_experience(
            history=history_text,
            reward_change=reward_change,
            current_score=current_score,
            scoring_action=scoring_action,
            location_issued=location_issued,
            location_after=location_after,
            retry_note=retry_note,
        )
        if contains_facts(retry_summary):
            return retry_summary, "retry_ok"

        fact = (
            f"<scoring_fact>{int(reward_change):+d} points were earned by "
            f"<step>{scoring_action}</step> at <loc>{location_issued}</loc>. "
            f"Location after the command: <loc>{location_after}</loc>."
            f"</scoring_fact>"
        )
        return f"{fact}\n{retry_summary}", "prefixed_authoritative_fact"

    def _event_text_signature(self, text: str) -> str:
        normalized = self._normalize_event_piece(text)
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def _normalize_event_piece(self, text: str) -> str:
        text = (text or "").lower()
        text = re.sub(
            r"</?(?:loc|step|obj|room|tag|dif|scoring_fact)\b[^>]*>",
            " ",
            text,
        )
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or "unknown"

    def _is_empty_experience_summary(self, summary: str) -> bool:
        """True when the summarizer intentionally declined to store memory."""
        text = (summary or "").strip().lower()
        text = re.sub(r"^\|start\|\s*|\s*\|end\|$", "", text).strip()
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text in {"", "none", "null", "noexperience"}

    def _is_narrative_action(self, action: str) -> bool:
        """True if the action is an examine / read / talk type."""
        a = action.lower().strip()
        return a.startswith((
            "examine ", "read ", "look at ", "inspect ", "describe ",
            "ask ", "talk to ", "x ",
        ))

    def _is_informative(self, observation: str) -> bool:
        """True if the observation contains meaningful content."""
        obs = observation.lower()
        noise = [
            "nothing special", "nothing unusual", "i don't understand",
            "i don't know the word", "you can't see any", "there is nothing",
            "how does one", "you must tell me", "that's not a verb",
            "you can't", "i don't see any", "what do you want to",
        ]
        if any(p in obs for p in noise):
            return False
        return len(observation.strip()) > 50

    def _is_movement_action(self, action: str) -> bool:
        """True if the action is a movement/navigation command."""
        directions = set(self._direction_aliases().values()) | {
            "n", "s", "e", "w", "ne", "nw", "se", "sw", "u", "d", "in", "out",
        }
        a = action.lower().strip()
        if a in directions:
            return True
        words = a.split()
        if len(words) >= 2 and words[0] in {"go", "walk", "run", "head", "travel", "move"}:
            if words[1] in directions:
                return True
            if len(words) >= 3 and words[1] == "to" and words[2] in directions:
                return True
        if len(words) >= 2 and words[0] == "climb" and words[1] in {"up", "down"}:
            return True
        return False

    def _blocked_direction_for_command(self, action: str) -> str:
        """Return canonical direction only for simple direction commands."""
        aliases = self._direction_aliases()
        a = str(action or "").strip().lower()
        if not a:
            return ""
        if a in aliases:
            return aliases[a]
        if a in aliases.values():
            return a
        words = a.split()
        if len(words) >= 2 and words[0] in {"go", "walk", "run", "head", "travel"}:
            candidate = words[1]
            if candidate == "to" and len(words) >= 3:
                candidate = words[2]
            return aliases.get(candidate, candidate if candidate in aliases.values() else "")
        return ""

    def _empty_navigation_guard_debug(self) -> dict:
        """Stable log schema for visit-scoped navigation enforcement."""
        return {
            "triggered": False,
            "layer": 0,
            "direction": "",
            "failed_this_visit": 0,
            "last_message": "",
            "last_failed_steps_ago": None,
            "safety_switch": "",
            "adjudication_raw_response": "",
            "adjudicated_insisted": False,
            "menu_offered": [],
            "menu_raw_response": "",
            "substituted_command": "",
            "final_command": "",
        }

    def _visit_direction_record(self, location: str, direction: str,
                                create: bool = False) -> dict:
        location_key = normalize_location_key(location) or "unknown"
        direction = self._blocked_direction_for_command(direction) or str(direction or "").strip().lower()
        if not direction:
            return {}
        location_records = self._visit_direction_failures.get(location_key)
        if location_records is None:
            if not create:
                return {}
            location_records = {}
            self._visit_direction_failures[location_key] = location_records
        record = location_records.get(direction)
        if record is None and create:
            record = {
                "count": 0,
                "last_message": "",
                "last_step": 0,
                "adjudicated": False,
                "menu_adjudicated": False,
                "menu_substituted": False,
            }
            location_records[direction] = record
        return record or {}

    def _record_visit_direction_failure(self, location: str, direction: str,
                                        observation: str, step: int) -> dict:
        """Record one rejected movement during the current contiguous room visit."""
        record = self._visit_direction_record(location, direction, create=True)
        record["count"] = int(record.get("count", 0)) + 1
        record["last_message"] = str(observation or "").strip()
        record["last_step"] = int(step or 0)
        return dict(record)

    def _reset_direction_visit_budget(self, previous_location: str,
                                      current_location: str):
        """End the source visit and start the destination with fresh budgets."""
        previous_key = normalize_location_key(previous_location)
        current_key = normalize_location_key(current_location)
        if previous_key:
            self._visit_direction_failures.pop(previous_key, None)
        if current_key:
            self._visit_direction_failures.pop(current_key, None)

    def _observation_mentions_direction(self, observation: str, direction: str) -> bool:
        direction = self._blocked_direction_for_command(direction) or str(direction or "").strip().lower()
        if not direction:
            return False
        text = str(observation or "").replace("’", "'").lower()
        segments = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", text)
        for segment in segments:
            normalized = re.sub(r"[^a-z0-9]+", " ", segment)
            if not re.search(rf"\b{re.escape(direction)}\b", normalized):
                continue
            if self._is_rejected_direction_mention(segment, direction):
                continue
            return True
        return False

    def _is_rejected_direction_mention(self, segment: str, direction: str) -> bool:
        """Distinguish a route hint from an error message echoing the command."""
        segment = re.sub(r"\s+", " ", str(segment or "").strip().lower())
        escaped = re.escape(direction)
        toward = r"(?:to\s+|towards?\s+)?(?:the\s+)?"
        blocked_state = (
            r"(?:blocked|closed|impassable|impenetrable|unavailable|"
            r"not\s+available|not\s+possible|dead\s+end)"
        )
        patterns = (
            rf"\b(?:can\s*not|can't|could\s*not|couldn't|won't|unable\s+to)\s+"
            rf"(?:go|walk|run|head|travel|move)\s+{toward}{escaped}\b",
            rf"\b(?:no|not)\s+(?:way|path|passage|exit|route|door)\s+"
            rf"{toward}{escaped}\b",
            rf"\b{blocked_state}\b[^.!?]{{0,48}}\b{escaped}\b",
            rf"\b{escaped}\b[^.!?]{{0,48}}"
            rf"\b(?:is\s+)?{blocked_state}\b",
            rf"\b(?:don't|do\s+not|doesn't|does\s+not)\s+"
            rf"(?:understand|recognize)[^.!?]*\b{escaped}\b",
            rf"\b{escaped}\b[^.!?]*\b(?:don't|do\s+not|doesn't|does\s+not)\s+"
            rf"(?:understand|recognize)\b",
        )
        return any(re.search(pattern, segment) for pattern in patterns)

    def _navigation_safety_switch(self, location: str, direction: str,
                                  observation: str, failed_this_visit: int) -> str:
        if self._observation_mentions_direction(observation, direction):
            return "observation_mention"
        if self.kg_map.was_direction_confirmed(location, direction):
            return "confirmed_history"
        if (
            self.kg_map.has_same_title_sibling(location)
            or self.kg_map.room_fingerprint_conflicts(location, observation)
        ):
            return "identity_risk"
        if failed_this_visit <= 0:
            return "first_probe"
        return ""

    def _navigation_alternative_menu(self, exhausted_direction: str,
                                     affordance_agenda: list) -> list[str]:
        """Return compact untried/open exits followed by pending local commands."""
        room_info = self.kg_map.get_current_room_info()
        candidates = []
        candidates.extend(room_info.get("may_direction", []) or [])
        candidates.extend((room_info.get("directions", {}) or {}).keys())
        for entry in affordance_agenda or []:
            candidates.extend(entry.get("pending_commands", []) or [])

        exhausted_direction = self._blocked_direction_for_command(exhausted_direction)
        output = []
        seen = set()
        for candidate in candidates:
            command = str(candidate or "").strip()
            if not command:
                continue
            if self._blocked_direction_for_command(command) == exhausted_direction:
                continue
            key = normalize_command_key(command)
            if key and key not in seen:
                output.append(command)
                seen.add(key)
            if len(output) >= 8:
                break
        return output

    def _apply_navigation_enforcement(self, command: str, raw_llm_response: str,
                                      repeat_check: dict, prompt: str,
                                      observation: str,
                                      affordance_agenda: list) -> tuple:
        """Rate-limit repeated rejected exits without permanently prohibiting them."""
        debug = self._empty_navigation_guard_debug()
        debug["final_command"] = command
        direction = self._blocked_direction_for_command(command)
        if not direction:
            return command, raw_llm_response, repeat_check, debug

        location = self.kg_map.current_location or "unknown"
        record = self._visit_direction_record(location, direction)
        failed_this_visit = int(record.get("count", 0)) if record else 0
        last_step = int(record.get("last_step", 0)) if record else 0
        last_message = str(record.get("last_message", "")) if record else ""
        debug.update({
            "direction": direction,
            "failed_this_visit": failed_this_visit,
            "last_message": last_message,
            "last_failed_steps_ago": (
                max(0, int(self.step_count) - last_step) if last_step else None
            ),
        })

        safety_switch = self._navigation_safety_switch(
            location=location,
            direction=direction,
            observation=observation,
            failed_this_visit=failed_this_visit,
        )
        if safety_switch:
            debug["safety_switch"] = safety_switch
            return command, raw_llm_response, repeat_check, debug

        if failed_this_visit >= 1 and not record.get("adjudicated"):
            debug["triggered"] = True
            debug["layer"] = 2
            record["adjudicated"] = True
            recency = debug["last_failed_steps_ago"]
            adjudication_prompt = (
                f"{prompt}\n\n"
                "NAVIGATION CHECK:\n"
                f"You chose '{command}'. That exact direction failed in this room "
                f"during this visit {failed_this_visit} time(s), most recently "
                f"{recency} step(s) ago, with: {json.dumps(last_message, ensure_ascii=False)}.\n"
                "If you have a concrete, currently-observable reason to retry it, "
                "answer with the same command and state that reason. Otherwise "
                "choose a different command."
            )
            adjudication_raw = self.llm.chat(
                system_prompt="You are an expert player of text-based interactive fiction games.",
                user_prompt=adjudication_prompt,
                think=True,
            )
            adjudicated_command = self._parse_command(adjudication_raw)
            insisted = self._blocked_direction_for_command(adjudicated_command) == direction
            debug["adjudication_raw_response"] = adjudication_raw
            debug["adjudicated_insisted"] = insisted
            command = adjudicated_command
            raw_llm_response = adjudication_raw
            repeat_check = self._parse_repeat_check(adjudication_raw)
            debug["final_command"] = command
            return command, raw_llm_response, repeat_check, debug

        if (
            failed_this_visit >= 2
            and record.get("adjudicated")
            and not record.get("menu_adjudicated")
        ):
            debug["triggered"] = True
            debug["layer"] = 3
            record["menu_adjudicated"] = True
            menu = self._navigation_alternative_menu(direction, affordance_agenda)
            debug["menu_offered"] = menu
            menu_text = json.dumps(menu, ensure_ascii=False) if menu else "[]"
            menu_prompt = (
                f"{prompt}\n\n"
                "NAVIGATION CHECK:\n"
                f"Direction '{direction}' is exhausted for this visit after "
                f"{failed_this_visit} fresh failure(s). Choose one of: {menu_text}. "
                "Return one executable command."
            )
            menu_raw = self.llm.chat(
                system_prompt="You are an expert player of text-based interactive fiction games.",
                user_prompt=menu_prompt,
                think=True,
            )
            menu_command = self._parse_command(menu_raw)
            insisted = self._blocked_direction_for_command(menu_command) == direction
            debug["menu_raw_response"] = menu_raw
            debug["adjudication_raw_response"] = menu_raw
            debug["adjudicated_insisted"] = insisted
            raw_llm_response = menu_raw
            if insisted:
                command = menu[0] if menu else "look"
                record["menu_substituted"] = True
                debug["substituted_command"] = command
                repeat_check = {
                    "is_repeat": False,
                    "reason": (
                        "Visit-scoped navigation rate limit selected an offered alternative "
                        "after repeated fresh failures."
                    ),
                }
            else:
                command = menu_command
                repeat_check = self._parse_repeat_check(menu_raw)
            debug["final_command"] = command
            return command, raw_llm_response, repeat_check, debug

        return command, raw_llm_response, repeat_check, debug

    def _prompt_kg_map_context(self) -> str:
        """Render prompt-facing KG JSON with evidence-rich blocked exits."""
        context = self.kg_map.to_clean_dict()
        room_state = context.get("current_room_state", {})
        location = self.kg_map.current_location or "unknown"
        room_info = self.kg_map.get_current_room_info()
        ledger_records = self.attempt_ledger.counts_for_location(location)
        enriched = []
        for direction in room_info.get("blocked_directions", []) or []:
            direction = self._blocked_direction_for_command(direction) or str(direction)
            visit_record = self._visit_direction_record(location, direction)
            ledger_record = {}
            for candidate in ledger_records.values():
                if self._blocked_direction_for_command(candidate.get("command", "")) != direction:
                    continue
                if int(candidate.get("last_step", 0)) >= int(ledger_record.get("last_step", 0)):
                    ledger_record = candidate
            last_step = int(visit_record.get("last_step", 0) or 0)
            last_message = str(visit_record.get("last_message", "") or "")
            if not last_message:
                last_message = str(ledger_record.get("last_observation", "") or "")
                last_step = int(ledger_record.get("last_step", 0) or 0)
            enriched.append({
                "direction": direction,
                "failed_this_visit": int(visit_record.get("count", 0) or 0),
                "last_message": last_message,
                "last_failed_steps_ago": (
                    max(0, int(self.step_count) - last_step) if last_step else None
                ),
            })
        room_state["blocked_exits"] = enriched
        context["current_room_state"] = room_state
        return json.dumps(context, indent=2, ensure_ascii=False)

    def _is_location_change_action(self, action: str, action_valid) -> bool:
        """True when a room title in the response may update current location."""
        if action_valid is not True:
            return False
        a = str(action or "").strip().lower()
        if not a:
            return False
        if a in self.kg_map._direction_set() or self._is_movement_action(a):
            return True
        first = a.split()[0]
        return first in {
            "enter", "exit", "leave", "climb", "descend", "ascend", "cross",
            "board", "disembark", "crawl", "jump",
        }

    def _filter_failed_movement_triples(self, triples: list) -> list:
        """Drop state-changing KG claims extracted from a failed movement.

        A blocked movement does not move the player or confirm an exit, even if
        the relation extractor hallucinates triples such as
        <Forest, north, to north>. Keep only requirement-style triples, which
        can encode useful obstacle information without changing location.
        """
        filtered = []
        direction_rels = self.kg_map._direction_set()
        for subj, rel, obj in triples:
            rel_lower = rel.strip().lower()
            if subj.strip().lower() == "you" and rel_lower == "in":
                continue
            if rel_lower in direction_rels:
                continue
            if rel_lower in {"need", "require"}:
                filtered.append((subj, rel, obj))
        return filtered

    def _without_location_triples(self, triples: list) -> list:
        """Drop only current-location claims while preserving object facts."""
        output = []
        for subj, rel, obj in triples or []:
            if str(subj).strip().lower() == "you" and str(rel).strip().lower() == "in":
                continue
            output.append((subj, rel, obj))
        return output

    def _parse_command(self, response: str) -> str:
        """Extract the game command from the LLM response.

        Prefer the paper's requested <com>...</com> field. Qwen sometimes emits
        malformed-but-obvious variants (for example <north>...</nort> or plain
        text inside |start|...|end|); recover those before using the safe fallback.
        """
        com_match = re.search(r"<com>\s*([^<\n]+?)\s*</com>", response, re.IGNORECASE)
        if com_match:
            cmd = self._canonicalize_command(self._clean_command(com_match.group(1)))
            if self._is_plausible_command(cmd):
                return cmd

        for candidate in self._command_candidates(response):
            cmd = self._canonicalize_command(self._clean_command(candidate))
            if self._is_plausible_command(cmd):
                return cmd

        return "look"  # fallback: safe and avoids sending prose to the game

    def _parse_repeat_check(self, response: str) -> dict:
        """Extract the advisory repeat self-check from the action response."""
        match = re.search(r"<repeat>\s*(.*?)\s*</repeat>", response or "", re.IGNORECASE | re.DOTALL)
        if not match:
            return {"present": False}
        body = match.group(1).strip()
        parsed = {}
        json_match = re.search(r"\{.*\}", body, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except Exception:
                parsed = {}
        is_repeat = parsed.get("is_repeat", False)
        if isinstance(is_repeat, str):
            is_repeat = is_repeat.strip().lower() in {"true", "yes", "1"}
        return {
            "present": True,
            "is_repeat": bool(is_repeat),
            "reason": self._clean_text(parsed.get("reason", body)),
            "raw": body,
        }

    def _command_candidates(self, response: str) -> list:
        """Return possible commands from common malformed LLM output shapes."""
        if not response:
            return []

        candidates = []

        # Examples: <north>Move north...</nort>, <northeast>to north</northeast>,
        # or <take>pile of leaves</take>. The tag name is often the intended command.
        for tag, body in re.findall(r"<([a-zA-Z][\w-]{0,24})>\s*([^<\n]*)", response):
            tagged = self._command_from_tag(tag, body)
            if tagged:
                candidates.append(tagged)

        # Examples: |start| south |end| or |start| take pile of leaves <rea>...
        for block in re.findall(r"\|start\|\s*(.*?)\s*\|end\|", response, re.IGNORECASE | re.DOTALL):
            plain = self._candidate_from_text(block)
            if plain:
                candidates.append(plain)

        return candidates

    def _command_from_tag(self, tag: str, body: str) -> str:
        """Recover a command when the model used the command itself as a tag."""
        tag_lower = tag.strip().lower()
        ignored_tags = {
            "com", "rea", "reason", "motivation", "start", "end", "act",
            "loc", "obj", "tag", "dif", "room", "repeat",
        }
        if tag_lower in ignored_tags:
            return ""

        directions = self._direction_aliases()
        body_cmd = self._canonicalize_command(
            self._clean_command(self._candidate_from_text(body))
        )
        if tag_lower in directions:
            tag_cmd = directions[tag_lower]
            # Malformed Qwen outputs sometimes use a stale direction as the tag
            # while the tag body contains the intended recovery command, e.g.
            # <north>Try moving east...</north>. Prefer a clear body command so
            # parser recovery does not turn strategy changes back into loops.
            if body_cmd and self._is_plausible_command(body_cmd):
                if body_cmd in set(directions.values()):
                    return body_cmd
                body_words = body_cmd.split()
                if len(body_words) >= 2 and body_words[0] in {"go", "walk", "head", "travel", "move"}:
                    if body_words[1] in directions:
                        return directions[body_words[1]]
            body_direction = self._direction_from_body_intent(body)
            if body_direction and body_direction != tag_cmd:
                return body_direction
            return tag_cmd

        one_arg_verbs = {
            "take", "get", "open", "close", "read", "examine", "x", "drop",
            "use", "climb", "enter", "move", "push", "pull", "turn", "unlock",
            "light", "eat", "drink", "throw", "attack", "kill",
        }
        if tag_lower in one_arg_verbs and body_cmd and self._is_short_object_phrase(body_cmd):
            return f"{tag_lower} {body_cmd}"

        # Some malformed tags are abbreviations like <nor>north</nor>. In that
        # case the body, not the tag, is the command.
        if body_cmd and self._looks_like_bare_command(body_cmd):
            return body_cmd

        return ""

    def _direction_from_body_intent(self, text: str) -> str:
        """Return a clear navigation command mentioned in malformed tag text."""
        if not text:
            return ""
        body = re.sub(r"<[^>]+>", " ", text.lower())
        body = body.replace("|", " ")
        body = re.sub(r"\s+", " ", body).strip()
        directions = self._direction_aliases()
        dir_pattern = (
            r"northwest|northeast|southwest|southeast|"
            r"north|south|east|west|up|down"
        )
        command_verbs = (
            r"try|trying|move|moving|go|going|head|heading|walk|walking|"
            r"travel|explore|check|test|follow"
        )
        m = re.search(
            rf"\b(?:{command_verbs})\s+(?:to\s+|towards?\s+|the\s+)?({dir_pattern})\b",
            body,
        )
        if m:
            return directions[m.group(1)]
        m = re.search(r"\b(northern|southern|eastern|western)\s+(?:path|part|direction|area|side)\b", body)
        if m:
            return {
                "northern": "north",
                "southern": "south",
                "eastern": "east",
                "western": "west",
            }[m.group(1)]
        return ""

    def _candidate_from_text(self, text: str) -> str:
        """Strip prompt scaffolding/prose and return a compact command candidate."""
        if not text:
            return ""
        text = re.sub(r"Your internal reasoning steps Here\.?", " ", text, flags=re.IGNORECASE)
        text = re.split(r"<(?:rea|reason|motivation)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("|", " ")
        text = re.sub(r"\bFinal Command\s*:", " ", text, flags=re.IGNORECASE)
        text = re.split(r"[\r\n]", text, maxsplit=1)[0]
        text = re.sub(r"\s+", " ", text).strip()
        return self._clean_command(text)

    def _direction_aliases(self) -> dict:
        return {
            "n": "north", "s": "south", "e": "east", "w": "west",
            "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
            "u": "up", "d": "down",
            "north": "north", "south": "south", "east": "east", "west": "west",
            "northeast": "northeast", "northwest": "northwest",
            "southeast": "southeast", "southwest": "southwest",
            "up": "up", "down": "down",
        }

    def _canonicalize_command(self, cmd: str) -> str:
        """Normalize obvious navigation prose back to IF direction commands."""
        cmd = re.sub(r"\s+", " ", cmd.strip()).lower()
        directions = self._direction_aliases()
        if cmd in directions:
            return directions[cmd]
        if cmd.startswith("to ") and cmd[3:] in directions:
            return directions[cmd[3:]]
        words = cmd.split()
        if len(words) >= 2 and words[0] in {"go", "walk", "head", "travel"} and words[1] in directions:
            return directions[words[1]]
        if len(words) >= 3 and words[0] == "move" and words[1] in directions:
            # "move north along the path" is navigation prose; "move rug" remains untouched.
            return directions[words[1]]
        return cmd

    def _clean_command(self, cmd: str) -> str:
        """Strip markdown formatting that the LLM sometimes wraps commands in."""
        cmd = cmd.strip()
        cmd = re.sub(r'^`+|`+$', '', cmd).strip()   # remove surrounding backticks
        cmd = re.sub(r'^\*+|\*+$', '', cmd).strip()  # remove surrounding asterisks
        cmd = re.sub(r'^\[+|\]+$', '', cmd).strip()  # remove surrounding square brackets
        cmd = re.sub(r'^"+|"+$', '', cmd).strip()    # remove surrounding quotes
        return cmd

    def _is_short_object_phrase(self, text: str) -> bool:
        if not text or any(p in text for p in ".?!:;<>|"):
            return False
        return 1 <= len(text.split()) <= 5

    def _looks_like_bare_command(self, cmd: str) -> bool:
        if not self._is_plausible_command(cmd):
            return False
        return len(cmd.split()) <= 6

    def _is_plausible_command(self, cmd: str) -> bool:
        """Return False if cmd looks like assistant prose rather than a game command."""
        if not cmd:
            return False
        if len(cmd) > 50 or any(ch in cmd for ch in "<>|\n\r"):
            return False
        if any(p in cmd for p in ".?!"):
            return False
        if len(cmd.split()) > 7:
            return False
        bad_phrases = [
            "would you", "i can help", "as an ai", "here's", "please ", "i'll ",
            "i will ", "should ", "could ", "might ", "let's ", "follow the",
            "explore the", "take a look", "based on", "given that", "since ",
        ]
        return not any(p in cmd.lower() for p in bad_phrases)

    def _format_history(self) -> str:
        """Format the recent history for prompt inclusion."""
        if not self.history:
            return "No history yet. This is the start of the game."

        output = []
        for i, (action, obs) in enumerate(self.history):
            output.append(f"Turn {i+1}:")
            output.append(f"  Action: {action}")
            # Truncate long observations
            obs_short = obs[:300] + "..." if len(obs) > 300 else obs
            output.append(f"  Observation: {obs_short}")
        return "\n".join(output)

    def get_stats(self) -> dict:
        """Get current agent statistics."""
        return {
            "step": self.step_count,
            "score": self.total_score,
            "rooms_visited": self.kg_map.num_rooms(),
            "actions_learned": self.action_space.num_actions(),
            "experiences_stored": self.experience_lib.num_experiences(),
            "situations_stored": len(self.situation_memory.active_situations()),
            "attempt_ledger": self.attempt_ledger.to_dict(),
            "failed_action_memory": self.failed_action_memory.to_dict(),
            "hazard_room_deaths": dict(getattr(self, "_hazard_room_deaths", {})),
            "hazard_room_evidence": dict(getattr(self, "_hazard_room_evidence", {})),
            "current_location": self.kg_map.current_location,
        }

    def get_step_details(self) -> list:
        """Return the detailed per-step log."""
        return self.step_details
