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
from collections import deque
from .kg_map import KGMap
from .action_space import ActionSpace
from .experience_lib import ExperienceLib
from .opportunity_module import SituationMemory
from .planning import ActivePlanMemory
from .affordance_brainstormer import AffordanceBrainstormer
from .action_memory import FailedActionMemory, StateScopedActionMemory
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
        self.active_plan_memory = ActivePlanMemory()
        self.affordance_brainstormer = AffordanceBrainstormer()
        self.failed_action_memory = FailedActionMemory()
        self.state_action_memory = StateScopedActionMemory()

        # History tracking
        self.history = []          # list of (action, observation) tuples
        self.prev_action = None
        self.prev_score = 0
        self.total_score = 0
        self.step_count = 0

        # Neutral-state tracking
        self.consecutive_failures = 0
        self.recent_failed_actions = []

        # Per-step detail log for tracking
        self.step_details = []
        self.pending_generation = None
        self.initial_observation_processed = False

    def reset(self, keep_experiences: bool = True):
        """Reset the agent for a new epoch.

        Args:
            keep_experiences: If True, keep learning state across epochs.
                            Experience Library always persists in this mode;
                            Action Space also persists when configured.
                            If False, perform a full reset.
        """
        self.kg_map.reset()
        if not keep_experiences or not config.PERSIST_ACTION_SPACE:
            self.action_space.reset()
        if not keep_experiences:
            self.experience_lib.reset()
            self.situation_memory.reset()
            self.failed_action_memory.reset()
            self.state_action_memory.reset()
        self.active_plan_memory.reset()
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

        detail["modules"]["action_space"] = {
            "prev_action_valid": action_valid,
            "action_split": action_split,
            "total_verbs": len(self.action_space.verbs),
            "total_actions_learned": self.action_space.num_actions(),
            "all_verbs": list(self.action_space.verbs.keys()),
        }

        extracted_triples = []
        applied_triples = []
        try:
            extracted_triples = self.fm.extract_relations(completed_action, observation)
            applied_triples = extracted_triples
            prev_lower = completed_action.lower().strip()
            if action_valid is False and prev_lower in self.kg_map._direction_set():
                applied_triples = self._filter_failed_movement_triples(extracted_triples)
            self.kg_map.update(applied_triples, completed_action)
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
            logger.debug(f"KG-map updated with {len(applied_triples)} triples")
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")
            extracted_triples = [("ERROR", str(e), "")]
            applied_triples = extracted_triples

        detail["modules"]["kg_map"] = {
            "extracted_triples": [(s, r, o) for s, r, o in extracted_triples],
            "applied_triples": [(s, r, o) for s, r, o in applied_triples],
            "current_location": self.kg_map.current_location,
            "rooms_visited": list(self.kg_map.visited_rooms),
            "inventory": list(self.kg_map.inventory),
            "room_info": self.kg_map.get_current_room_info(),
            "kg_map_context": self.kg_map.to_prompt_string(),
        }

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
        detail["modules"]["auxiliary_gate"] = auxiliary_gate
        inventory_reconciliation = self._apply_gate_inventory_update(
            auxiliary_gate=auxiliary_gate,
            inventory_before=inventory_before,
        )
        detail["modules"]["inventory_reconciliation"] = inventory_reconciliation
        if inventory_reconciliation.get("applied"):
            detail["modules"]["kg_map"]["inventory"] = list(self.kg_map.inventory)
            detail["modules"]["kg_map"]["room_info"] = self.kg_map.get_current_room_info()
            detail["modules"]["kg_map"]["kg_map_context"] = self.kg_map.to_prompt_string()

        environmental_change_detection = (
            auxiliary_gate.get("environmental_change_detection") or {}
        )
        if auxiliary_gate.get("use_legacy_environmental_detection"):
            environmental_change_detection = self._detect_environmental_change_with_llm(
                action=completed_action,
                observation=observation,
                action_valid=action_valid,
            )
            environmental_change_detection["source"] = "legacy_fallback_after_gate_failure"
        detail["modules"]["environmental_change_detection"] = environmental_change_detection

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

        active_plan_progress = self._update_active_plan_after_completed_action(
            completed_action=completed_action,
            source_state_snapshot=source_state_snapshot,
            situation_resolution=situation_resolution,
        )
        detail["modules"]["active_plan_progress"] = active_plan_progress
        situation_plan_update = self._apply_gate_situation_plan(
            auxiliary_gate=auxiliary_gate,
            active_situations_after=situation_detail["active_situations_after"],
            skipped=active_plan_progress.get("status") in {
                "cleared_attempted",
                "cleared_resolved_situation",
            },
        )
        detail["modules"]["situation_plan"] = situation_plan_update

        experience_summary = None
        summary_log_entries = []
        experience_triggered = bool(done or (reward_change != 0 and action_valid is True))
        if experience_triggered:
            try:
                history_text = self._format_history()
                exp_summary = self.llm.summarize_experience(
                    history=history_text,
                    reward_change=reward_change,
                    current_score=score,
                )
                experience_summary = exp_summary
                score_summary_prompt = self.llm.last_summary_prompt or ""
                self.experience_lib.store_experience(
                    experience_text=exp_summary,
                    metadata={
                        "trigger": "score_change",
                        "score_change": reward_change,
                        "current_score": score,
                        "step": self.step_count,
                        "location": self.kg_map.current_location or "unknown",
                    },
                )
                summary_log_entries.append({
                    "state_type": "terminal" if done and reward_change == 0 else "score_change",
                    "prompt": score_summary_prompt,
                    "summary": exp_summary,
                    "raw_response": self.llm.last_summary_raw_response or "",
                    "metadata": {
                        "score_change": reward_change,
                        "current_score": score,
                        "step": self.step_count,
                        "location": self.kg_map.current_location or "unknown",
                        "action": completed_action,
                    },
                })
                logger.info(f"Experience stored: score change {reward_change:+d}")
            except Exception as e:
                logger.warning(f"Experience summarization failed: {e}")
                experience_summary = f"ERROR: {e}"

        neutral_triggers = []
        neutral_summaries = []
        neutral_summaries_skipped = []
        neutral_event_keys = []
        if reward_change == 0 and not done:
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
                        self.experience_lib.store_experience(
                            experience_text=summary,
                            metadata={
                                "trigger": trigger_type,
                                "score_change": 0,
                                "current_score": score,
                                "step": self.step_count,
                                "location": self.kg_map.current_location or "unknown",
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
                                "score_change": 0,
                                "current_score": score,
                                "step": self.step_count,
                                "location": self.kg_map.current_location or "unknown",
                                "action": completed_action,
                                "event_key": event_key,
                                "gate_reason": trigger_meta.get("gate_reason", ""),
                                "gate_evidence": trigger_meta.get("gate_evidence", ""),
                            },
                        })
                        logger.info(f"Neutral experience stored: {trigger_type}")
                except Exception as e:
                    logger.warning(f"Neutral experience failed ({trigger_type}): {e}")

        action_failure_memory = {
            "status": "not_applicable",
            "source_location": prev_location or self.kg_map.current_location or "unknown",
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

        location_changed_for_affordance = (
            bool(prev_location)
            and bool(self.kg_map.current_location)
            and self.kg_map.current_location != prev_location
        )
        inventory_changed_for_repetition = set(self.kg_map.inventory) != inventory_before
        if not source_state_snapshot:
            source_location = action_failure_memory["source_location"]
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
            action_failure_memory["source_location"],
        )
        progress_was_useful = bool(
            reward_change != 0
            or location_changed_for_affordance
            or inventory_changed_for_repetition
            or environmental_change_detection.get("environmental_change")
            or done
        )

        if action_valid is False:
            failure_record = action_failure_memory.get("stored_failure") or {}
            failure_reason = (
                failure_record.get("failure_reason")
                or self._fallback_failure_reason(observation)
            )
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

        query = f"Location: {self.kg_map.current_location}. Observation: {observation[:200]}"
        experiences = self.experience_lib.retrieve_relevant(query)

        detail["modules"]["experience_lib"] = {
            "score_changed": reward_change != 0,
            "terminal": done,
            "experience_triggered": experience_triggered,
            "new_experience_summary": experience_summary,
            "neutral_triggers_fired": [t for t, _ in neutral_triggers],
            "neutral_summaries": neutral_summaries,
            "neutral_summaries_skipped": neutral_summaries_skipped,
            "neutral_event_keys": neutral_event_keys,
            "summary_log_entries": summary_log_entries,
            "retrieved_experiences": experiences,
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
            self.prev_action = None
            self.pending_generation = None
            self.prev_score = score
            self.total_score = score
            self.step_details.append(detail)
            logger.info(f"Completed step {self.step_count}: score={score} "
                        f"({reward_change:+d}) cmd='{completed_action}' "
                        f"loc='{self.kg_map.current_location}'")
            return ""

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
            self.kg_map.update(triples, "look")
            print(f"  Initial KG seed: {len(triples)} triple(s) extracted.", flush=True)
            logger.info(f"Initial KG-map seeded with {len(triples)} triples")
        except Exception as e:
            logger.warning(f"Initial KG-map extraction failed: {e}")
            print(f"  Initial KG seed skipped: {e}", flush=True)

    def _generate_command(self, observation: str, score: int,
                          experiences: str = None,
                          reset_affordance_cache: bool = False,
                          affordance_gate_decision: dict = None) -> tuple:
        """Generate the next command and return it with prompt metadata."""
        initial_generation = self.step_count == 0 and self.prev_action is None
        room_info = self.kg_map.get_current_room_info()
        current_objects = room_info.get("objects", [])
        current_state_snapshot = self._repetition_state_snapshot(
            location=self.kg_map.current_location or "unknown",
            observation=observation,
            visible_objects=current_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
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
        if experiences is None:
            if initial_generation:
                print("  Initial action generation: retrieving experiences...", flush=True)
            query = f"Location: {self.kg_map.current_location}. Observation: {observation[:200]}"
            experiences = self.experience_lib.retrieve_relevant(query)
            if initial_generation:
                print("  Initial action generation: experience retrieval done.", flush=True)
        stored_situations = self.situation_memory.format_for_prompt()
        action_space_context = self.action_space.to_prompt_string(current_objects)
        active_plan = self.active_plan_memory.active_plan()
        active_plan_navigation_hint = self._navigation_hint_for_plan(active_plan)
        active_plan_context = self.active_plan_memory.format_for_prompt(
            active_plan_navigation_hint
        )
        if initial_generation:
            print("  Initial action generation: running affordance brainstorm...", flush=True)
        affordance_result = self._brainstorm_affordances(
            observation=observation,
            current_objects=current_objects,
            stored_situations=self.situation_memory.active_situations(),
            action_space_context=action_space_context,
            known_failed_here=known_failed_here,
            same_state_tried_commands=same_state_tried_commands,
            experiences=experiences,
            score=score,
            reset_cache=reset_affordance_cache,
            gate_decision=affordance_gate_decision,
        )
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
            active_plan_context=active_plan_context,
            brainstormed_command_ideas=brainstormed_command_ideas,
            known_failed_commands_here=known_failed_here,
            same_state_tried_commands=same_state_tried_context,
            history=self._format_history(),
            history_length=config.HISTORY_LENGTH,
            score=score,
            observation=observation,
        )

        raw_llm_response = ""
        try:
            if initial_generation:
                print("  Initial action generation: asking main LLM for command...", flush=True)
            raw_llm_response = self.llm.chat(
                system_prompt="You are an expert player of text-based interactive fiction games.",
                user_prompt=prompt,
                think=True,
            )
            command = self._parse_command(raw_llm_response)
            if initial_generation:
                print("  Initial action generation: main LLM returned.", flush=True)
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["connect", "refused", "unreachable", "failed to connect"]):
                raise RuntimeError(f"Ollama server unreachable: {e}") from e
            logger.error(f"Action generation failed: {e}")
            raw_llm_response = f"ERROR: {e}"
            command = "look"

        generation = {
            "kg_map_context": self.kg_map.to_prompt_string(),
            "action_space_context": action_space_context,
            "retrieved_experiences": experiences,
            "stored_situations_context": stored_situations,
            "active_plan_context": active_plan_context,
            "active_plan": active_plan,
            "active_plan_navigation_hint": active_plan_navigation_hint,
            "brainstormed_command_ideas": brainstormed_command_ideas,
            "affordance_agenda": brainstormed_command_ideas,
            "known_failed_commands_here": known_failed_here,
            "same_state_tried_commands": same_state_tried_context,
            "same_state_tried_command_list": same_state_tried_commands,
            "state_snapshot_at_generation": current_state_snapshot,
            "observation_at_generation": observation,
            "affordance_brainstorming": affordance_result,
            "llm_raw_response": raw_llm_response,
            "parsed_command": command,
            "score_at_generation": score,
        }
        return command, generation

    def _navigation_hint_for_plan(self, plan: dict | None) -> dict:
        """Return an advisory BFS next-step hint for an active situation plan."""
        if not plan:
            return {"status": "no_active_plan"}
        start = self.kg_map.current_location or ""
        target_raw = plan.get("target_location", "")
        target = self._resolve_known_location(target_raw)
        if not start:
            return {
                "status": "missing_current_location",
                "target_location": target_raw,
            }
        if not target:
            return {
                "status": "target_not_in_known_map",
                "current_location": start,
                "target_location": target_raw,
            }
        if start == target:
            return {
                "status": "at_target",
                "current_location": start,
                "target_location": target,
                "next_command": "",
                "route_commands": [],
                "route_locations": [start],
            }
        route = self._bfs_route(start, target)
        if not route:
            return {
                "status": "no_known_route",
                "current_location": start,
                "target_location": target,
            }
        commands, locations = route
        return {
            "status": "route_found",
            "current_location": start,
            "target_location": target,
            "next_command": commands[0] if commands else "",
            "next_location": locations[1] if len(locations) > 1 else target,
            "route_commands": commands,
            "route_locations": locations,
        }

    def _resolve_known_location(self, target: str) -> str:
        """Resolve a plan target phrase to a known room name, if possible."""
        raw = self._clean_text(target)
        if not raw:
            return ""
        known = list(self.kg_map.nodes.keys())
        raw_norm = self._normalize_event_piece(raw)
        for loc in known:
            if self._normalize_event_piece(loc) == raw_norm:
                return loc
        parts = [
            part.strip()
            for part in re.split(r"[/,;|]+", raw)
            if part.strip()
        ]
        for part in parts:
            part_norm = self._normalize_event_piece(part)
            for loc in known:
                if self._normalize_event_piece(loc) == part_norm:
                    return loc
        for loc in known:
            loc_norm = self._normalize_event_piece(loc)
            if loc_norm and (loc_norm in raw_norm or raw_norm in loc_norm):
                return loc
        return ""

    def _bfs_route(self, start: str, target: str) -> tuple[list[str], list[str]] | None:
        """Shortest route over confirmed KG-map direction edges."""
        if start not in self.kg_map.nodes or target not in self.kg_map.nodes:
            return None
        queue = deque([(start, [], [start])])
        visited = {start}
        while queue:
            room, commands, locations = queue.popleft()
            node = self.kg_map.nodes.get(room, {})
            for direction, dest in (node.get("direction", {}) or {}).items():
                if not dest:
                    continue
                if dest in visited:
                    continue
                next_commands = commands + [direction]
                next_locations = locations + [dest]
                if dest == target:
                    return next_commands, next_locations
                if dest in self.kg_map.nodes:
                    visited.add(dest)
                    queue.append((dest, next_commands, next_locations))
        return None

    def _brainstorm_affordances(self, observation: str, current_objects: list,
                                stored_situations: list,
                                action_space_context: str,
                                known_failed_here: str,
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
        state_signature = self.affordance_brainstormer.state_signature(
            location=self.kg_map.current_location or "unknown",
            visible_objects=current_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
        )
        unproductive_commands = self.affordance_brainstormer.unproductive_commands(
            location=self.kg_map.current_location or "unknown",
            state_signature=state_signature,
        )
        filtered_commands = (
            failure_context["failed_commands"]
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
        result = {
            "status": "not_run",
            "location": self.kg_map.current_location or "unknown",
            "observation": observation,
            "visible_objects": list(current_objects or []),
            "inventory": list(self.kg_map.inventory),
            "recent_failed_commands": list(self.recent_failed_actions),
            "known_failed_commands_here": known_failed_here,
            "recent_command_outcomes": recent_command_outcomes,
            "failed_commands": filtered_commands,
            "unproductive_commands": unproductive_commands,
            "same_state_tried_commands": list(same_state_tried_commands or []),
            "same_state_tried_records": same_state_records,
            "failed_records_here": failed_records_here,
            "pending_carryover_commands": [],
            "failed_command_verbs": failure_context["failed_verbs"],
            "active_situations": list(stored_situations or []),
            "score": score,
            "state_signature": state_signature,
            "reset_cache": reset_cache,
            "gate_decision": dict(gate_decision or {}),
            "gate_reason": (gate_decision or {}).get("reason", ""),
            "gate_focus": list((gate_decision or {}).get("focus", []) or []),
            "cached_ideas_available": 0,
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
            )
        else:
            cached_ideas = self.affordance_brainstormer.cached_ideas_for_state(
                location=result["location"],
                state_signature=state_signature,
                failed_commands=filtered_commands,
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
                    failed_records=failed_records_here,
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
                    failed_records=failed_records_here,
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
        active_plan_before = self.active_plan_memory.active_plan()
        recent_command_outcomes = self._recent_same_location_outcomes(location)
        recent_failed_for_gate = list(self.recent_failed_actions)
        if action_valid is False and action:
            recent_failed_for_gate.append(action)
        affordance_signature = self.affordance_brainstormer.state_signature(
            location=location,
            visible_objects=visible_objects,
            inventory=list(self.kg_map.inventory),
            score=score,
        )
        failure_context = self.affordance_brainstormer.failure_context(
            recent_failed_commands=recent_failed_for_gate,
            known_failed_commands_here=known_failed_here,
        )
        cached_affordance_ideas = self.affordance_brainstormer.cached_ideas_for_state(
            location=location,
            state_signature=affordance_signature,
            failed_commands=(
                failure_context["failed_commands"]
                + list(same_state_tried_commands or [])
            ),
        )

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
            "active_plan_before": active_plan_before,
            "recent_failed_commands": recent_failed_for_gate,
            "known_failed_commands_here": known_failed_here,
            "recent_command_outcomes": recent_command_outcomes,
            "same_state_tried_commands": same_state_tried_commands,
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
                active_plan=active_plan_before,
                recent_failed_commands=recent_failed_for_gate,
                known_failed_commands_here=known_failed_here,
                recent_command_outcomes=recent_command_outcomes,
                same_state_tried_commands=same_state_tried_commands,
                kg_map=self.kg_map.to_prompt_string(),
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
        """Apply concrete inventory corrections from the auxiliary gate."""
        decision = (auxiliary_gate or {}).get("decision", {})
        update = decision.get("inventory_update", {})
        result = {
            "status": "noop",
            "gate_status": (auxiliary_gate or {}).get("status", "unknown"),
            "raw_update": update,
            "before": list(self.kg_map.inventory),
            "after": list(self.kg_map.inventory),
            "applied": False,
            "reason": "",
        }
        if not isinstance(update, dict) or not update.get("changed"):
            result["reason"] = self._clean_text(update.get("reason", "")) if isinstance(update, dict) else ""
            return result
        applied = self.kg_map.apply_inventory_update(
            update,
            inventory_before=sorted(str(item) for item in (inventory_before or [])),
            action=(auxiliary_gate or {}).get("action", ""),
        )
        result.update(applied)
        return result

    def _apply_gate_situation_plan(self, auxiliary_gate: dict,
                                   active_situations_after: list,
                                   skipped: bool = False) -> dict:
        """Accept an auxiliary-gate plan proposal when it matches active memory."""
        decision = (auxiliary_gate or {}).get("decision", {})
        proposal = decision.get("situation_plan", {})
        result = {
            "status": "noop",
            "gate_status": (auxiliary_gate or {}).get("status", "unknown"),
            "raw_plan": proposal,
            "active_plan_before": self.active_plan_memory.active_plan(),
            "active_plan_after": self.active_plan_memory.active_plan(),
            "reason": "",
        }
        if skipped:
            result["status"] = "skipped_after_plan_clear"
            result["reason"] = "A plan was just attempted or resolved this step."
            return result
        if not isinstance(proposal, dict) or not proposal.get("create"):
            result["reason"] = self._clean_text(proposal.get("reason", "")) if isinstance(proposal, dict) else ""
            return result

        related = proposal.get("related_situation") or {}
        active_keys = {
            self.situation_memory.key_for(s)
            for s in (active_situations_after or [])
        }
        if self.situation_memory.key_for(related) not in active_keys:
            result["status"] = "skipped_related_situation_not_active"
            result["reason"] = "The proposed plan does not match an active stored situation."
            return result

        current_plan = self.active_plan_memory.active_plan()
        if current_plan and self.active_plan_memory.matches_situation(related):
            result["status"] = "skipped_duplicate_active_plan"
            result["reason"] = "An active plan already targets this stored situation."
            result["active_plan_after"] = self.active_plan_memory.active_plan()
            return result

        applied = self.active_plan_memory.set_plan(
            proposal,
            step=self.step_count,
            source="auxiliary_gate",
        )
        result.update(applied)
        result["active_plan_after"] = self.active_plan_memory.active_plan()
        return result

    def _update_active_plan_after_completed_action(self, completed_action: str,
                                                   source_state_snapshot: dict,
                                                   situation_resolution: dict) -> dict:
        """Clear an active plan once the agent actually tries it."""
        plan = self.active_plan_memory.active_plan()
        result = {
            "status": "no_active_plan",
            "active_plan_before": plan,
            "active_plan_after": plan,
            "reason": "",
        }
        if not plan:
            return result

        for situation in (situation_resolution or {}).get("removed_situations", []) or []:
            if self.active_plan_memory.matches_situation(situation):
                cleared = self.active_plan_memory.clear(
                    reason="related stored situation was resolved",
                    step=self.step_count,
                )
                result.update(cleared)
                result["status"] = "cleared_resolved_situation"
                result["active_plan_after"] = self.active_plan_memory.active_plan()
                return result

        source_location = (
            (source_state_snapshot or {}).get("location")
            or self.kg_map.current_location
            or "unknown"
        )
        target = self._resolve_known_location(plan.get("target_location", ""))
        normalized_action = self._clean_text(completed_action).lower()
        target_commands = set(plan.get("commands_to_try_at_target", []) or [])
        utility_commands = {"look", "inventory", "i"}
        at_target_when_chosen = bool(target and source_location == target)
        tried_target_command = at_target_when_chosen and normalized_action in target_commands
        tried_non_nav_at_target = (
            at_target_when_chosen
            and normalized_action not in utility_commands
            and not self._is_movement_action(normalized_action)
        )

        if tried_target_command or tried_non_nav_at_target:
            reason = (
                "tried one of the plan target commands"
                if tried_target_command
                else "tried a non-navigation command at the plan target"
            )
            cleared = self.active_plan_memory.clear(reason=reason, step=self.step_count)
            result.update(cleared)
            result["status"] = "cleared_attempted"
            result["active_plan_after"] = self.active_plan_memory.active_plan()
            return result

        result["status"] = "kept_active"
        result["reason"] = "Plan remains advisory; latest action did not attempt it."
        result["active_plan_after"] = self.active_plan_memory.active_plan()
        return result

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
                                           action_valid) -> dict:
        command_outcome = self._normalize_command_outcome(
            parsed.get("command_outcome", {}),
            action_valid=action_valid,
        )
        summary_raw = parsed.get("summary_triggers", {})
        if not isinstance(summary_raw, dict):
            summary_raw = {}

        navigation_summary = self._normalize_gate_summary_trigger(
            summary_raw.get("navigation", parsed.get("navigation", {})),
            default_run=False,
        )
        env_summary = self._normalize_gate_summary_trigger(
            summary_raw.get(
                "environmental",
                summary_raw.get("environmental_change", parsed.get("environmental_change", {})),
            ),
            default_run=False,
        )
        narrative_summary = self._normalize_gate_summary_trigger(
            summary_raw.get("narrative", parsed.get("narrative", {})),
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
            parsed.get("situation_detection", {}),
        )
        affordance_raw = parsed.get("affordance_brainstorming", {})
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
            "inventory_update": self._normalize_gate_inventory_update(
                parsed.get("inventory_update", {}),
            ),
            "situation_plan": self._normalize_gate_situation_plan(
                parsed.get("situation_plan", {}),
            ),
        }

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

    def _normalize_gate_situation_plan(self, value) -> dict:
        if not isinstance(value, dict):
            value = {}
        related = value.get("related_situation", {})
        if not isinstance(related, dict):
            related = {}
        return {
            "create": self._coerce_gate_bool(value.get("create", False), False),
            "target_location": self._clean_text(value.get("target_location", "")),
            "related_situation": {
                "location": self._clean_text(related.get("location", "")),
                "situation": self._clean_text(related.get("situation", "")),
            },
            "reason": self._clean_text(value.get("reason", "")),
            "suggested_preparation": self._clean_gate_command_list(
                value.get("suggested_preparation", [])
            ),
            "commands_to_try_at_target": self._clean_gate_command_list(
                value.get("commands_to_try_at_target", [])
            ),
        }

    def _clean_gate_command_list(self, value) -> list[str]:
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, list):
            raw_items = value
        else:
            raw_items = []
        cleaned = []
        for item in raw_items:
            text = self._clean_text(item).lower()
            text = re.sub(r"^[`\"']+|[`\"']+$", "", text).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned[:8]

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
            "inventory_update": {
                "changed": False,
                "authoritative": False,
                "items_now_carried": [],
                "items_added": [],
                "items_removed": [],
                "reason": "Gate unavailable.",
            },
            "situation_plan": {
                "create": False,
                "target_location": "",
                "related_situation": {
                    "location": "",
                    "situation": "",
                },
                "reason": "Gate unavailable.",
                "suggested_preparation": [],
                "commands_to_try_at_target": [],
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

    def _visible_objects_for_location(self, location: str) -> list:
        if location and location in self.kg_map.nodes:
            return list(self.kg_map.nodes[location].get("have", []))
        room_info = self.kg_map.get_current_room_info()
        return list(room_info.get("objects", []))

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
            "loc", "obj", "tag", "dif", "room",
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
            "active_plan": self.active_plan_memory.active_plan(),
            "failed_action_memory": self.failed_action_memory.to_dict(),
            "current_location": self.kg_map.current_location,
        }

    def get_step_details(self) -> list:
        """Return the detailed per-step log."""
        return self.step_details
