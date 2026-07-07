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
from .kg_map import KGMap
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
        self.kg_map = KGMap()
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

    def reset(self, keep_experiences: bool = True):
        """Reset the agent for a new epoch.

        Args:
            keep_experiences: If True, keep only the Experience Library across
                            epochs. All live world/action bookkeeping is reset
                            because a new epoch starts from a fresh game world.
                            If False, also clear the Experience Library.
        """
        self.kg_map.reset()
        self.action_space.reset()
        self.situation_memory.reset()
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
            self._initialize_kg_from_observation(observation)
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
            prev_lower = completed_action.lower().strip()
            if is_valid is False and prev_lower in self.kg_map._direction_set():
                self.kg_map.mark_direction_tried(prev_lower)
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

        extracted_triples = []
        applied_triples = []
        kg_location_resolution = {
            "raw_room_title": "",
            "fm_location": "",
            "title_fallback_used": False,
            "title_overrode_fm": False,
            "chosen_location_hint": "",
            "previous_location": prev_location,
            "location_after_update": "",
            "confirmed_transition_type": "",
            "confirmed_transition": {},
            "action_transition_candidate": {},
            "action_transition_gate_decision": {},
            "action_transition_applied": False,
            "action_transition_status": "",
        }
        timer = time.perf_counter()
        if pure_rejection:
            kg_location_resolution["location_after_update"] = self.kg_map.current_location
            kg_location_resolution["action_transition_status"] = "skipped_pure_rejection"
            logger.debug("KG relation extraction skipped for pure parser/world rejection")
        else:
            try:
                extracted_triples = self.fm.extract_relations(completed_action, observation)
                room_title = self._extract_observation_room_title(observation)
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
                    title_key = self._normalize_event_piece(room_title)
                    fm_key = self._normalize_event_piece(fm_location)
                    current_key = self._normalize_event_piece(self.kg_map.current_location or "")
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
                    fm_key = self._normalize_event_piece(fm_location)
                    current_key = self._normalize_event_piece(self.kg_map.current_location or "")
                    if self.kg_map.current_location and fm_key != current_key:
                        applied_triples = self._without_location_triples(applied_triples)
                        kg_location_resolution["remote_fm_location_ignored"] = fm_location
                self.kg_map.update(applied_triples, completed_action)
                kg_location_resolution["location_after_update"] = self.kg_map.current_location
                if (action_valid is True
                        and prev_location
                        and self.kg_map.current_location
                        and self.kg_map.current_location != prev_location
                        and prev_lower in self.kg_map._direction_set()):
                    self.kg_map.confirm_direction(
                        from_location=prev_location,
                        direction=prev_lower,
                        to_location=self.kg_map.current_location,
                    )
                    kg_location_resolution["confirmed_transition_type"] = "direction"
                    kg_location_resolution["confirmed_transition"] = {
                        "from": prev_location,
                        "command": prev_lower,
                        "to": self.kg_map.current_location,
                    }
                elif (action_valid is True
                        and prev_location
                        and self.kg_map.current_location
                        and self.kg_map.current_location != prev_location
                        and prev_lower not in self.kg_map._direction_set()):
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
            "room_info": self.kg_map.get_current_room_info(),
            "kg_map_context": self.kg_map.to_prompt_string(),
        }

        timer = time.perf_counter()
        auxiliary_gate = self._run_auxiliary_gate(
            action=completed_action,
            observation=observation,
            score=score,
            reward_change=reward_change,
            action_valid=action_valid,
            prev_location=prev_location,
            inventory_before=inventory_before,
            visited_rooms_before=visited_rooms_before,
        )
        record_timing("auxiliary_gate", timer)
        detail["modules"]["auxiliary_gate"] = auxiliary_gate
        command_outcome = auxiliary_gate.get("decision", {}).get("command_outcome", {})
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
        if action_transition_result.get("applied"):
            kg_location_resolution["confirmed_transition_type"] = "action"
            kg_location_resolution["confirmed_transition"] = (
                action_transition_result.get("candidate", {})
            )
            detail["modules"]["kg_map"]["room_info"] = self.kg_map.get_current_room_info()
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
            detail["modules"]["kg_map"]["room_info"] = self.kg_map.get_current_room_info()
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
            detail["modules"]["kg_map"]["room_info"] = self.kg_map.get_current_room_info()
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
                score_location = prev_location or self.kg_map.current_location or "unknown"
                location_after = self.kg_map.current_location or "unknown"
                score_experience_kind = (
                    "achievement" if reward_change > 0
                    else "death_warning" if reward_change < 0
                    else "terminal"
                )
                score_event_key = ""
                if reward_change > 0:
                    score_event_key = self._score_event_key(
                        trigger="gain",
                        action=completed_action,
                        location=score_location,
                        reward_change=reward_change,
                    )
                    self.earned_score_event_keys_this_epoch.add(score_event_key)
                    self.earned_score_location_reward_keys_this_epoch.add(
                        self._score_location_reward_key(score_location, reward_change)
                    )

                if score_event_key and self.experience_lib.event_seen(score_event_key):
                    score_summary_skipped = {
                        "trigger": "score_change",
                        "event_key": score_event_key,
                        "reason": "duplicate_score_gain",
                        "kind": score_experience_kind,
                        "epoch": self.current_epoch,
                        "step": self.step_count,
                        "score_change": reward_change,
                        "current_score": score,
                        "location": score_location,
                        "action": completed_action,
                    }
                    self.experience_lib.record_event(
                        score_event_key,
                        metadata=score_summary_skipped,
                    )
                    logger.info(
                        f"Experience skipped as duplicate score gain: {completed_action}"
                    )
                else:
                    if reward_change < 0:
                        ledger_block = self.attempt_ledger.format_room_block(
                            score_location
                        )
                        exp_summary = self.llm.summarize_loss_experience(
                            history=history_text,
                            reward_change=reward_change,
                            current_score=score,
                            fatal_action=completed_action,
                            location=location_after,
                            location_issued=score_location,
                            command_history_block=ledger_block,
                        )
                    else:
                        exp_summary = self.llm.summarize_experience(
                            history=history_text,
                            reward_change=reward_change,
                            current_score=score,
                            scoring_action=completed_action,
                            location_issued=score_location,
                            location_after=location_after,
                        )
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
                        "trigger": "score_change",
                        "score_change": reward_change,
                        "current_score": score,
                        "epoch": self.current_epoch,
                        "step": self.step_count,
                        "location": score_location,
                        "location_issued": score_location,
                        "location_after": location_after,
                        "action": completed_action,
                        "scoring_action": completed_action if reward_change > 0 else "",
                        "fatal_action": completed_action if reward_change < 0 else "",
                        "terminal": done,
                    }
                    if reward_change > 0:
                        score_metadata["summary_validation"] = validation
                    if score_event_key:
                        score_metadata["event_key"] = score_event_key
                    self.experience_lib.store_experience(
                        experience_text=exp_summary,
                        metadata=score_metadata,
                    )
                    if score_event_key:
                        self.experience_lib.record_event(
                            score_event_key,
                            metadata={
                                **score_metadata,
                                "status": "stored",
                            },
                        )
                    summary_log_entries.append({
                        "state_type": (
                            "terminal" if done and reward_change == 0
                            else "score_loss" if reward_change < 0
                            else "score_change"
                        ),
                        "prompt": score_summary_prompt,
                        "summary": exp_summary,
                        "raw_response": self.llm.last_summary_raw_response or "",
                        "metadata": score_metadata,
                    })
                    logger.info(f"Experience stored: score change {reward_change:+d}")

                if reward_change > 0 and score_event_key:
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
                    event_key = self._neutral_event_key(
                        trigger=trigger_type,
                        action=completed_action,
                        observation=observation,
                        location=self.kg_map.current_location or "unknown",
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
                                "location": self.kg_map.current_location or "unknown",
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
                                "location": self.kg_map.current_location or "unknown",
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
                                "location": self.kg_map.current_location or "unknown",
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
                                "location": self.kg_map.current_location or "unknown",
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

    def _initialize_kg_from_observation(self, observation: str):
        """Seed the KG-map from the game's initial observation before step 1."""
        if self.initial_observation_processed:
            return
        self.initial_observation_processed = True
        if not observation:
            return
        try:
            print("  Initial KG seed: extracting initial room/object facts...", flush=True)
            triples = self.fm.extract_relations("look", observation, max_new=192)
            room_title = self._extract_observation_room_title(observation)
            if room_title:
                triples = self._with_authoritative_location_triple(triples, room_title)
            self.kg_map.update(triples, "look")
            if room_title:
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

    def _location_from_triples(self, triples: list) -> str:
        for subj, rel, obj in triples or []:
            if str(subj).strip().lower() == "you" and str(rel).strip().lower() == "in":
                return self._clean_text(obj)
        return ""

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

    def _extract_observation_room_title(self, observation: str) -> str:
        """Conservative parser for IF room-title headers."""
        text = str(observation or "").strip()
        if not text:
            return ""
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
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
        return candidate

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
            action_space_context=action_space_context,
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

        prompt = LPLH_ACTION_GENERATION_PROMPT.format(
            kg_map=self.kg_map.to_prompt_string(),
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
        blocked_direction_guard = {
            "triggered": False,
            "blocked_command": "",
            "retry_raw_response": "",
            "retry_command": "",
            "fallback_used": False,
        }
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
            if self._is_blocked_direction_command(
                command,
                self.kg_map.current_location or "unknown",
            ):
                blocked_direction_guard["triggered"] = True
                blocked_direction_guard["blocked_command"] = command
                retry_prompt = (
                    f"{prompt}\n\n"
                    "BLOCKED EXIT CORRECTION:\n"
                    f"The command '{command}' is a confirmed blocked exit from "
                    f"the current location '{self.kg_map.current_location}'. "
                    "Choose one different executable command. Do not choose that "
                    "same blocked direction again unless the map explicitly changed."
                )
                retry_raw = self.llm.chat(
                    system_prompt="You are an expert player of text-based interactive fiction games.",
                    user_prompt=retry_prompt,
                    think=True,
                )
                retry_command = self._parse_command(retry_raw)
                blocked_direction_guard["retry_raw_response"] = retry_raw
                blocked_direction_guard["retry_command"] = retry_command
                if self._is_blocked_direction_command(
                    retry_command,
                    self.kg_map.current_location or "unknown",
                ):
                    command = "look"
                    repeat_check = {
                        "is_repeat": False,
                        "reason": "Blocked-exit guard replaced a repeated confirmed blocked direction with look.",
                    }
                    blocked_direction_guard["fallback_used"] = True
                else:
                    command = retry_command
                    raw_llm_response = retry_raw
                    repeat_check = self._parse_repeat_check(retry_raw)
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
            "kg_map_context": self.kg_map.to_prompt_string(),
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
                                action_space_context: str,
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
                action_space=action_space_context,
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

    def _run_auxiliary_gate(self, action: str, observation: str, score: int,
                            reward_change: int, action_valid,
                            prev_location: str,
                            inventory_before: set = None,
                            visited_rooms_before: set = None) -> dict:
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
        action_transition_candidate = {}
        if (action_valid is True
                and prev_location
                and location
                and location != prev_location
                and prev_lower not in self.kg_map._direction_set()):
            action_transition_candidate = {
                "from": prev_location,
                "command": action,
                "to": location,
            }

        result = {
            "status": "not_run",
            "location": location,
            "previous_location": prev_location or "unknown",
            "action": action,
            "action_valid": action_valid,
            "observation": observation,
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
        target = self._normalize_event_piece(location or "unknown")
        outcomes: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add(command: str, observation: str, source_location: str):
            if not command or not observation:
                return
            if self._normalize_event_piece(source_location) != target:
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
            shown = records[:config.EXPERIENCE_TOP_K]
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
        return self._format_experiences_for_prompt(shown)

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

        # Then local setup actions that enable an unearned reward.
        for record in records:
            if len(selected) >= top_k:
                break
            if (kind_of(record) == "enabler"
                    and not self._enabler_completed_this_epoch(record)
                    and self._experience_location_relevant(record, include_neighbors=True)):
                add(record)
                break

        # Then nearby achievement that has not already been earned this epoch.
        for record in records:
            if len(selected) >= top_k:
                break
            if (kind_of(record) == "achievement"
                    and not self._achievement_earned_this_epoch(record)
                    and self._experience_location_relevant(record, include_neighbors=True)):
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
            if kind_of(record) == "route" and not self._route_already_mapped(record):
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
        return self._score_location_reward_earned(location, reward)

    def _score_location_reward_from_metadata_earned(self, metadata: dict) -> bool:
        reward = metadata.get("score_change")
        location = (
            metadata.get("location_issued")
            or metadata.get("location")
            or metadata.get("enables_location")
        )
        return self._score_location_reward_earned(location, reward)

    def _score_location_reward_earned(self, location, reward) -> bool:
        try:
            reward_int = int(reward)
        except Exception:
            return False
        key = self._score_location_reward_key(str(location or ""), reward_int)
        return bool(key and key in self.earned_score_location_reward_keys_this_epoch)

    def _experience_location_relevant(self, record: dict,
                                      include_neighbors: bool = False) -> bool:
        metadata = record.get("metadata") or {}
        current_keys = self._current_location_relevance_keys(include_neighbors)
        for field in ("location", "location_issued", "location_after", "prev_location"):
            value = metadata.get(field)
            if value and normalize_location_key(value) in current_keys:
                return True
        return False

    def _current_location_relevance_keys(self, include_neighbors: bool) -> set[str]:
        current = self.kg_map.current_location or ""
        keys = {normalize_location_key(current)}
        if not include_neighbors or not current:
            return keys
        current_key = normalize_location_key(current)
        for loc, data in self.kg_map.nodes.items():
            loc_key = normalize_location_key(loc)
            direction_values = list((data.get("direction", {}) or {}).values())
            action_values = list((data.get("confirmed_actions", {}) or {}).values())
            if loc_key == current_key:
                keys.update(normalize_location_key(dest) for dest in direction_values)
                keys.update(normalize_location_key(dest) for dest in action_values)
            for dest in direction_values + action_values:
                if normalize_location_key(dest) == current_key:
                    keys.add(loc_key)
        keys.discard("")
        return keys

    def _warning_experience_relevant(self, record: dict) -> bool:
        if self._experience_location_relevant(record, include_neighbors=True):
            return True
        text = self._normalize_event_piece(record.get("text", ""))
        observation = ""
        if self.history:
            observation = self._normalize_event_piece(self.history[-1][1])
        current_objects = " ".join(self._visible_objects_for_location(
            self.kg_map.current_location or "unknown"
        ))
        context = self._normalize_event_piece(f"{observation} {current_objects}")
        if not text or not context:
            return False
        tokens = {tok for tok in context.split() if len(tok) >= 5}
        return bool(tokens and tokens.intersection(text.split()))

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

    def _format_experiences_for_prompt(self, records: list[dict]) -> str:
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
                    header_bits.append("use_as=nearby_reward_procedure")
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

    def _route_for_current_room(self, record: dict) -> bool:
        metadata = record.get("metadata") or {}
        if (metadata.get("kind") or metadata.get("trigger")) != "route":
            return False
        prev_location = metadata.get("prev_location", "")
        location = metadata.get("location", "")
        current_key = normalize_location_key(self.kg_map.current_location or "")
        return (
            normalize_location_key(prev_location) == current_key
            or normalize_location_key(location) == current_key
        )

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
                        result["status"] = "duplicate"

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

    def _neutral_event_key(self, trigger: str, action: str, observation: str,
                           location: str, prev_location: str = None,
                           failed_attempts: list = None) -> str:
        """Build a stable key for neutral-summary dedup across epochs."""
        trigger_norm = self._normalize_event_piece(trigger)
        action_norm = self._normalize_event_piece(action)
        location_norm = self._normalize_event_piece(location)

        if trigger_norm == "navigation":
            prev_norm = self._normalize_event_piece(prev_location or "unknown")
            return f"neutral:v1:navigation:{prev_norm}:{action_norm}:{location_norm}"

        if trigger_norm in {"narrative", "environmental"}:
            obs_sig = self._event_text_signature(observation)
            return f"neutral:v1:{trigger_norm}:{location_norm}:{action_norm}:{obs_sig}"

        if trigger_norm == "error_correction":
            failed_norm = " > ".join(
                self._normalize_event_piece(a) for a in (failed_attempts or [])
            )
            failed_sig = self._event_text_signature(failed_norm)
            return f"neutral:v1:error_correction:{location_norm}:{action_norm}:{failed_sig}"

        obs_sig = self._event_text_signature(observation)
        return f"neutral:v1:{trigger_norm}:{location_norm}:{action_norm}:{obs_sig}"

    def _score_event_key(self, trigger: str, action: str, location: str,
                         reward_change: int) -> str:
        """Build a stable key for score-gain summary dedup across epochs."""
        trigger_norm = self._normalize_event_piece(trigger)
        action_norm = self._normalize_event_piece(action)
        location_norm = self._normalize_event_piece(location)
        return (
            f"score:v1:{trigger_norm}:{location_norm}:"
            f"{action_norm}:{int(reward_change or 0)}"
        )

    def _score_location_reward_key(self, location: str, reward_change: int) -> str:
        location_norm = self._normalize_event_piece(location)
        return f"score_location_reward:v1:{location_norm}:{int(reward_change or 0)}"

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

            enabler_event_key = (
                "enabler:v1:"
                f"{normalize_location_key(enabler_location)}:"
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
                "action": enabler_action,
                "enabler_action": enabler_action,
                "enables_event_key": score_event_key,
                "enables_reward": int(reward_change),
                "enables_scoring_action": scoring_action,
                "enables_location": scoring_location,
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
            action_norm = self._normalize_event_piece(scoring_action)
            location_norm = self._normalize_event_piece(location_issued)
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
        text = re.sub(r"<[^>]+>", " ", text)
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

    def _is_blocked_direction_command(self, action: str, location: str) -> bool:
        direction = self._blocked_direction_for_command(action)
        return bool(direction and self.kg_map.is_direction_blocked(location, direction))

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
            "current_location": self.kg_map.current_location,
        }

    def get_step_details(self) -> list:
        """Return the detailed per-step log."""
        return self.step_details
