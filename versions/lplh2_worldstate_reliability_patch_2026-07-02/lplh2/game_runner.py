"""Game Runner - Orchestrates the full game loop.

Manages epochs, steps, scoring, and logging for running
LPLH agents on IF games via the Jericho environment.
"""

import os
import re
import sys
import json
import time
import logging
from datetime import datetime
from . import config
from .agent import LPLHAgent
from .llm_client import LLMClient
from .fm_client import FmClient

logger = logging.getLogger(__name__)


def _trunc(text: str, length: int = 200) -> str:
    """Truncate text for display."""
    if not text:
        return ""
    t = text.replace("\n", " ").strip()
    return t[:length] + "..." if len(t) > length else t


class GameRunner:
    """Runs LPLH agent on an IF game for multiple epochs."""

    def __init__(self, game_path: str, num_epochs: int = None,
                 max_steps: int = None, verbose: bool = True):
        """
        Args:
            game_path: Path to the game ROM file (e.g., "games/zork1.z5")
            num_epochs: Number of epochs to run (default from config)
            max_steps: Max steps per epoch (default from config)
            verbose: Whether to print step-by-step output
        """
        self.game_path = game_path
        self.num_epochs = num_epochs or config.NUM_EPOCHS
        self.max_steps = max_steps or config.MAX_STEPS_PER_EPOCH
        self.verbose = verbose

        # Results tracking
        self.epoch_results = []
        self.all_scores = []
        self._log_file = None
        self._summary_log_file = None
        self._situation_log_file = None
        self._affordance_log_file = None
        self._action_failure_log_file = None
        self._action_generation_log_file = None
        self._auxiliary_gate_log_file = None
        self._kg_location_log_file = None
        self._timing_log_file = None
        self._run_timestamp = None
        self._experiment_log_dir = None
        self._run_log_path = None
        self._summary_log_path = None
        self._situation_log_path = None
        self._affordance_log_path = None
        self._action_failure_log_path = None
        self._action_generation_log_path = None
        self._auxiliary_gate_log_path = None
        self._kg_location_log_path = None
        self._timing_log_path = None
        self._results_path = None
        self._step_log_path = None
        self._all_step_logs = []
        self._current_epoch_log = None

    def run(self) -> dict:
        """Run the full game experiment.

        Returns:
            Dictionary with all results and statistics.
        """
        import jericho

        # Initialize Jericho environment
        env = jericho.FrotzEnv(self.game_path)
        game_name = os.path.splitext(os.path.basename(self.game_path))[0]

        # Open human-readable run log
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        self._experiment_log_dir = os.path.join(
            config.LOGS_DIR, f"{game_name}_{self._run_timestamp}"
        )
        os.makedirs(self._experiment_log_dir, exist_ok=True)
        log_path = os.path.join(self._experiment_log_dir, "run_log.txt")
        summary_log_path = os.path.join(self._experiment_log_dir, "summary_module_log.txt")
        situation_log_path = os.path.join(self._experiment_log_dir, "situation_memory_log.txt")
        affordance_log_path = os.path.join(self._experiment_log_dir, "affordance_brainstorm_log.txt")
        action_failure_log_path = os.path.join(self._experiment_log_dir, "action_failure_memory_log.txt")
        action_generation_log_path = os.path.join(self._experiment_log_dir, "action_generation_log.txt")
        auxiliary_gate_log_path = os.path.join(self._experiment_log_dir, "auxiliary_gate_log.txt")
        kg_location_log_path = os.path.join(self._experiment_log_dir, "kg_location_log.txt")
        timing_log_path = os.path.join(self._experiment_log_dir, "module_timing_log.txt")
        self._run_log_path = log_path
        self._summary_log_path = summary_log_path
        self._situation_log_path = situation_log_path
        self._affordance_log_path = affordance_log_path
        self._action_failure_log_path = action_failure_log_path
        self._action_generation_log_path = action_generation_log_path
        self._auxiliary_gate_log_path = auxiliary_gate_log_path
        self._kg_location_log_path = kg_location_log_path
        self._timing_log_path = timing_log_path
        self._step_log_path = os.path.join(self._experiment_log_dir, "steplog.json")
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._summary_log_file = open(summary_log_path, "w", encoding="utf-8", buffering=1)
        self._situation_log_file = open(situation_log_path, "w", encoding="utf-8", buffering=1)
        self._affordance_log_file = open(affordance_log_path, "w", encoding="utf-8", buffering=1)
        self._action_failure_log_file = open(
            action_failure_log_path, "w", encoding="utf-8", buffering=1
        )
        self._action_generation_log_file = open(
            action_generation_log_path, "w", encoding="utf-8", buffering=1
        )
        self._auxiliary_gate_log_file = open(
            auxiliary_gate_log_path, "w", encoding="utf-8", buffering=1
        )
        self._kg_location_log_file = open(
            kg_location_log_path, "w", encoding="utf-8", buffering=1
        )
        self._timing_log_file = open(timing_log_path, "w", encoding="utf-8", buffering=1)
        aux_model = config.LLM_ES_MODEL or "LLM_a fallback"
        brainstorm_model = config.LLM_BRAINSTORM_MODEL or aux_model
        brainstorm_model_lower = (config.LLM_BRAINSTORM_MODEL or "").lower()
        brainstorm_effort = (
            f" (reasoning_effort={config.LLM_BRAINSTORM_REASONING_EFFORT})"
            if (
                config.LLM_BRAINSTORM_MODEL
                and config.LLM_BRAINSTORM_REASONING_EFFORT
                and brainstorm_model_lower.startswith(("o1", "o3", "o4"))
            )
            else ""
        )
        self._log_file.write(f"LPLH Run Log — {game_name}\n")
        self._log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | "
            f"LLM_a: {config.LLM_PROVIDER}/{config.LLM_MODEL} | "
            f"LLM_aux/es: {aux_model} | "
            f"LLM_brainstorm: {brainstorm_model}{brainstorm_effort} | "
            f"fm: {config.FM_BASE_MODEL} + {config.FM_MODEL_PATH}\n"
        )
        self._log_file.write("=" * 70 + "\n")
        self._summary_log_file.write(f"LPLH Summary Module Log - {game_name}\n")
        self._summary_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | "
            f"LLM_es: {aux_model}\n"
        )
        self._summary_log_file.write("=" * 70 + "\n")
        self._situation_log_file.write(f"LPLH2 Situation Memory Log - {game_name}\n")
        self._situation_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | "
            f"LLM_es: {aux_model}\n"
        )
        self._situation_log_file.write("=" * 70 + "\n")
        self._affordance_log_file.write(f"LPLH2 Affordance Brainstorm Log - {game_name}\n")
        self._affordance_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | "
            f"LLM_brainstorm: {brainstorm_model}{brainstorm_effort}\n"
        )
        self._affordance_log_file.write("=" * 70 + "\n")
        self._action_failure_log_file.write(f"LPLH2 Action Failure Memory Log - {game_name}\n")
        self._action_failure_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | "
            f"LLM_es: {aux_model}\n"
        )
        self._action_failure_log_file.write("=" * 70 + "\n")
        self._action_generation_log_file.write(f"LPLH2 Action Generation Log - {game_name}\n")
        self._action_generation_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | "
            f"LLM_a: {config.LLM_PROVIDER}/{config.LLM_MODEL}\n"
        )
        self._action_generation_log_file.write("=" * 70 + "\n")
        self._auxiliary_gate_log_file.write(f"LPLH2 Auxiliary Gate Log - {game_name}\n")
        self._auxiliary_gate_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | "
            f"LLM_es: {aux_model}\n"
        )
        self._auxiliary_gate_log_file.write("=" * 70 + "\n")
        self._kg_location_log_file.write(f"LPLH2 KG Location Log - {game_name}\n")
        self._kg_location_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps}\n"
        )
        self._kg_location_log_file.write("=" * 70 + "\n")
        self._timing_log_file.write(f"LPLH2 Module Timing Log - {game_name}\n")
        self._timing_log_file.write(
            f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps}\n"
        )
        self._timing_log_file.write("=" * 70 + "\n")
        print(f"Experiment log dir: {self._experiment_log_dir}")
        print(f"Run log: {log_path}")
        print(f"Summary module log: {summary_log_path}")
        print(f"Situation memory log: {situation_log_path}")
        print(f"Affordance brainstorm log: {affordance_log_path}")
        print(f"Action failure memory log: {action_failure_log_path}")
        print(f"Action generation log: {action_generation_log_path}")
        print(f"Auxiliary gate log: {auxiliary_gate_log_path}")
        print(f"KG location log: {kg_location_log_path}")
        print(f"Module timing log: {timing_log_path}")

        # Initialize LPLH agent
        llm = LLMClient()
        fm = FmClient()
        agent = LPLHAgent(llm_client=llm, fm_client=fm)

        logger.info(f"Starting LPLH on '{game_name}': "
                     f"{self.num_epochs} epochs, {self.max_steps} steps/epoch")
        print(f"\n{'='*70}")
        print(f"  LPLH Framework - Playing: {game_name}")
        print(f"  Epochs: {self.num_epochs}, Steps/epoch: {self.max_steps}")
        print(f"  LLM_a: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
        print(f"  LLM_aux/es: {aux_model}")
        print(f"  LLM_brainstorm: {brainstorm_model}{brainstorm_effort}")
        print(f"  fm : {config.FM_BASE_MODEL} + {config.FM_MODEL_PATH}")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        # Integrity Check
        if os.path.getsize(self.game_path) == 0:
            logger.error("Game file is empty (0 bytes)! Upload failed?")
            print("\n❌ CRITICAL ERROR: Game file is 0 bytes. Please re-upload zork1.z5 correctly.\n")
            return {}


        start_time = time.time()
        all_step_logs = []   # detailed logs across epochs
        self._all_step_logs = all_step_logs
        self._current_epoch_log = None

        try:
            for epoch in range(1, self.num_epochs + 1):
                self._current_epoch_log = {
                    "epoch": epoch,
                    "status": "running",
                    "steps": [],
                }
                epoch_result, epoch_steps = self._run_epoch(env, agent, epoch, game_name)
                self.epoch_results.append(epoch_result)
                self.all_scores.append(epoch_result["final_score"])
                self._current_epoch_log.update({
                    "status": "completed",
                    "epoch_result": epoch_result,
                    "steps": epoch_steps,
                })
                all_step_logs.append(self._current_epoch_log)
                self._current_epoch_log = None
                self._save_step_log(self._logs_for_save(), game_name, quiet=True)

                print(f"\n{'─'*70}")
                print(f"  Epoch {epoch}/{self.num_epochs} Complete")
                print(f"  Final Score: {epoch_result['final_score']}  |  "
                      f"Max Score: {epoch_result['max_score']}  |  "
                      f"Steps Used: {epoch_result['steps_used']}")
                print(f"  Rooms: {epoch_result['rooms_visited']}  |  "
                      f"Experiences: {epoch_result['experiences_stored']}  |  "
                      f"Situations: {epoch_result.get('situations_stored', 0)}")

                # ── KG-Map rooms ──────────────────────────────
                rooms = list(agent.kg_map.visited_rooms)
                rooms_str = ", ".join(rooms) if rooms else "(none)"
                print(f"\n  KG-Map Rooms ({len(rooms)}):")
                print(f"    {rooms_str}")

                print(f"{'─'*70}\n")

                # ── Write same summary to the run log file ────
                if self._log_file:
                    self._log_file.write("\n" + "=" * 70 + "\n")
                    self._log_file.write(
                        f"EPOCH {epoch} SUMMARY\n"
                        f"  Final Score : {epoch_result['final_score']}\n"
                        f"  Steps Used  : {epoch_result['steps_used']}\n"
                        f"  Experiences : {epoch_result['experiences_stored']}\n"
                        f"  Situations  : {epoch_result.get('situations_stored', 0)}\n"
                    )
                    self._log_file.write(f"\n  KG-Map Rooms ({len(rooms)}):\n")
                    for r in rooms:
                        self._log_file.write(f"    - {r}\n")
                    self._log_file.write("=" * 70 + "\n")
        except KeyboardInterrupt:
            print("\n\n🛑 Run interrupted by user (Ctrl+C). Saving partial results...")
            logger.warning("Run interrupted by user.")
            if self._current_epoch_log is not None:
                self._current_epoch_log["status"] = "interrupted"
            if self._log_file:
                self._log_file.write("\nRUN INTERRUPTED BY USER - partial logs saved.\n")
        except RuntimeError as e:
            if "unreachable" in str(e).lower() or "connect" in str(e).lower():
                print(f"\n\n❌ OLLAMA SERVER LOST: {e}")
                print("The Ollama server crashed or became unreachable. Stopping run.")
                print("Partial results (completed epochs) will be saved.")
            else:
                raise
            logger.error(f"Run stopped: {e}")
            if self._current_epoch_log is not None:
                self._current_epoch_log["status"] = "runtime_error"
                self._current_epoch_log["error"] = str(e)
            if self._log_file:
                self._log_file.write(f"\nRUN STOPPED BY RUNTIME ERROR: {e}\n")
        finally:
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            if self._summary_log_file:
                self._summary_log_file.close()
                self._summary_log_file = None
            if self._situation_log_file:
                self._situation_log_file.close()
                self._situation_log_file = None
            if self._affordance_log_file:
                self._affordance_log_file.close()
                self._affordance_log_file = None
            if self._action_failure_log_file:
                self._action_failure_log_file.close()
                self._action_failure_log_file = None
            if self._action_generation_log_file:
                self._action_generation_log_file.close()
                self._action_generation_log_file = None
            if self._auxiliary_gate_log_file:
                self._auxiliary_gate_log_file.close()
                self._auxiliary_gate_log_file = None
            if self._kg_location_log_file:
                self._kg_location_log_file.close()
                self._kg_location_log_file = None
            if self._timing_log_file:
                self._timing_log_file.close()
                self._timing_log_file = None
            elapsed = time.time() - start_time
            logs_for_save = self._logs_for_save()

            if self.epoch_results:
                results = self._compute_summary(game_name, elapsed)
            else:
                results = self._compute_partial_summary(game_name, elapsed, logs_for_save)

            self._save_results(results, game_name)

            # Save detailed step log (whatever we have)
            if logs_for_save:
                self._save_step_log(logs_for_save, game_name)

            print(f"\n{'='*70}")
            if self.epoch_results:
                print(f"  LPLH Results for '{game_name}' (Partial)")
                print(f"  Average Score (all): {results['avg_score_all']:.1f}")
                print(f"  Max Score: {results['max_score']}")
            else:
                print(f"  LPLH Results: No full epochs completed.")
                print(f"  Partial steps saved: {results.get('partial_steps_saved', 0)}")
            print(f"  Log dir: {self._experiment_log_dir}")
            print(f"  Run log: {self._run_log_path}")
            print(f"  Summary log: {self._summary_log_path}")
            print(f"  Situation log: {self._situation_log_path}")
            print(f"  Affordance log: {self._affordance_log_path}")
            print(f"  Action failure log: {self._action_failure_log_path}")
            print(f"  Action generation log: {self._action_generation_log_path}")
            print(f"  Auxiliary gate log: {self._auxiliary_gate_log_path}")
            print(f"  KG location log: {self._kg_location_log_path}")
            print(f"  Module timing log: {self._timing_log_path}")
            print(f"  Step log: {self._step_log_path}")
            print(f"  Results: {self._results_path}")
            print(f"  Total Time: {elapsed:.1f}s")
            print(f"{'='*70}\n")

            return results


    def _logs_for_save(self) -> list:
        """Return completed epoch logs plus the in-progress epoch, if any."""
        logs = list(self._all_step_logs)
        if self._current_epoch_log is not None:
            logs.append(dict(self._current_epoch_log))
        return logs

    def _compute_partial_summary(self, game_name: str, elapsed: float,
                                 step_logs: list) -> dict:
        """Create a result payload even when no epoch completed."""
        partial_steps = sum(len(epoch.get("steps", [])) for epoch in step_logs)
        return {
            "game": game_name,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.LLM_MODEL,
            "avg_score_all": 0,
            "avg_score_last3": 0,
            "max_score": 0,
            "all_epoch_results": list(self.epoch_results),
            "partial": True,
            "partial_steps_saved": partial_steps,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            "log_dir_path": self._experiment_log_dir,
            "run_log_path": self._run_log_path,
            "summary_log_path": self._summary_log_path,
            "situation_log_path": self._situation_log_path,
            "affordance_log_path": self._affordance_log_path,
            "action_failure_log_path": self._action_failure_log_path,
            "action_generation_log_path": self._action_generation_log_path,
            "auxiliary_gate_log_path": self._auxiliary_gate_log_path,
            "kg_location_log_path": self._kg_location_log_path,
            "timing_log_path": self._timing_log_path,
            "step_log_path": self._step_log_path,
        }

    def _run_epoch(self, env, agent: LPLHAgent, epoch: int, game_name: str) -> tuple:
        """Run a single epoch of gameplay.

        Returns:
            Tuple of (epoch_result_dict, step_details_list)
        """
        # Reset environment and agent (keep experiences across epochs!)
        observation, info = env.reset()
        agent.reset(keep_experiences=True)

        # Set to verbose mode (paper specifies this).
        # We discard the verbose confirmation ("Maximum verbosity.") and keep
        # the real room description from env.reset() as the starting observation.
        try:
            env.step("verbose")
        except Exception:
            pass

        score = info.get("score", 0)
        max_score = score

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  EPOCH {epoch}")
            print(f"{'='*70}")
            print(f"\n  📍 Initial Observation:")
            print(f"  {_trunc(observation, 300)}")
            print()

        action = agent.act(observation, score, False, info)
        for step in range(1, self.max_steps + 1):
            if not action:
                logger.warning(f"No action available at step {step}; ending epoch.")
                break

            # Execute action in the game
            try:
                # print(f"DEBUG: Executing step {step} with action '{action}'...", flush=True)
                observation, reward, done, info = env.step(action)
                # print(f"DEBUG: Step returned done={done}", flush=True)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                logger.error(f"CRITICAL: Jericho step failed/crashed: {e} (Type: {type(e)})")
                print(f"\n❌ CRITICAL GAME CRASH at Step {step}: {e}")
                import traceback
                traceback.print_exc()
                done = True
                observation = "Game crashed."
                reward = 0
                info = {}
                break  # Stop the loop on crash


            score = info.get("score", score + reward)
            max_score = max(max_score, score)

            next_action = agent.act(
                observation,
                score,
                done,
                info,
                generate_next=(not done and step < self.max_steps),
            )

            # ── Live console output ───────────────────────────
            if self.verbose:
                self._print_step_detail(agent, step, action, observation, score, reward)
            self._log_step(epoch, step, action, observation, score, reward, agent)
            if self._current_epoch_log is not None:
                self._current_epoch_log.update({
                    "status": "running",
                    "steps_completed": step,
                    "last_score": score,
                    "last_action": action,
                    "steps": agent.get_step_details(),
                })
                self._save_step_log(self._logs_for_save(), game_name, quiet=True)

            if done:
                logger.info(f"Epoch {epoch} ended at step {step}")
                if self.verbose:
                    print(f"\n  🏁 GAME OVER at step {step}")
                break
            action = next_action

        stats = agent.get_stats()
        step_details = agent.get_step_details()

        epoch_result = {
            "epoch": epoch,
            "final_score": score,
            "max_score": max_score,
            "steps_used": step,
            "rooms_visited": stats["rooms_visited"],
            "actions_learned": stats["actions_learned"],
            "experiences_stored": stats["experiences_stored"],
            "situations_stored": stats.get("situations_stored", 0),
            "game": game_name,
        }

        return epoch_result, step_details

    def _print_step_detail(self, agent: LPLHAgent, step: int, action: str,
                            observation: str, score: int, reward: int):
        """Print detailed per-step info to console during the run."""
        # Get the latest step detail from the agent
        if not agent.step_details:
            return
        d = agent.step_details[-1]
        modules = d.get("modules", {})

        # ── Header ────────────────────────────────────────────
        reward_str = f" ({'+' if reward >= 0 else ''}{reward})" if reward != 0 else ""
        print(f"  ┌─ Step {step}  │  Score: {score}{reward_str}  │  "
              f"Location: {modules.get('kg_map', {}).get('current_location', '?')}", flush=True)
        print(f"  │", flush=True)

        # ── Command & Observation ─────────────────────────────
        print(f"  │  🎮 Command: {action}", flush=True)
        print(f"  │  📜 Observation: {_trunc(observation, 250)}", flush=True)

        # ── Module 1: KG-Map ──────────────────────────────────
        kg = modules.get("kg_map", {})
        triples = kg.get("applied_triples", kg.get("extracted_triples", []))
        if triples:
            print(f"  │", flush=True)
            print(f"  │  🗺️  KG-Map ({len(kg.get('rooms_visited', []))} rooms)", flush=True)
            for s, r, o in triples[:5]:
                print(f"  │     Triple: ({s}, {r}, {o})", flush=True)
            if len(triples) > 5:
                print(f"  │     ... +{len(triples)-5} more", flush=True)
        inv = kg.get("inventory", [])
        if inv:
            print(f"  │     Inventory: {', '.join(inv)}", flush=True)

        # ── Previous action validation ────────────────────────
        act_mod = modules.get("action_validation", {})
        valid = act_mod.get("prev_action_valid")
        if valid is not None:
            status = "✅ Valid" if valid is True else ("❌ Invalid" if valid is False else f"⚠️ {valid}")
            print(f"  │  Prev action: {status}", flush=True)

        fail_mem = modules.get("action_failure_memory", {})
        if fail_mem.get("stored_failure"):
            rec = fail_mem["stored_failure"]
            print(
                f"  │  Failed-command memory: {rec.get('command', '?')} -> "
                f"{_trunc(str(rec.get('failure_reason', '')), 120)}",
                flush=True,
            )
        if fail_mem.get("removed_failure"):
            rec = fail_mem["removed_failure"]
            print(
                f"  │  Removed failed-command memory after success: "
                f"{rec.get('command', '?')}",
                flush=True,
            )

        # ── Module 3: Experience Library ──────────────────────
        repeat_mem = modules.get("state_repetition_memory", {})
        if repeat_mem.get("stored_record"):
            rec = repeat_mem["stored_record"]
            print(
                f"  â”‚  Same-state repetition memory: {rec.get('command', '?')} -> "
                f"{_trunc(str(rec.get('reason', '')), 120)}",
                flush=True,
            )

        exp = modules.get("experience_lib", {})
        if exp.get("experience_triggered", exp.get("score_changed")):
            summary = exp.get("new_experience_summary", "")
            if summary and not str(summary).startswith("ERROR"):
                print(f"  │  💡 New Experience (score): {_trunc(str(summary), 150)}", flush=True)
        # Neutral-state experiences
        for trigger_type, summary in exp.get("neutral_summaries", []):
            label = {"navigation": "🧭", "narrative": "📖",
                     "environmental": "🔧", "error_correction": "✔️"}.get(trigger_type, "💡")
            print(f"  │  {label} New Experience ({trigger_type}): {_trunc(str(summary), 150)}", flush=True)
        for skipped in exp.get("neutral_summaries_skipped", []):
            trigger_type = skipped.get("trigger", "neutral")
            reason = skipped.get("reason", "duplicate_neutral_event")
            reason_text = "summary returned none" if reason == "summary_none" else "duplicate neutral event"
            print(f"  │  ⏭️ Skipped Experience ({trigger_type}): {reason_text}", flush=True)
        retrieved = exp.get("retrieved_experiences", "")
        if retrieved and retrieved != "No relevant experiences found yet.":
            print(f"  │  📚 Retrieved: {_trunc(str(retrieved), 150)}", flush=True)

        # ── LLM Response ──────────────────────────────────────
        situation = modules.get("situation_memory", {})
        stored_situation = situation.get("new_stored_situation")
        if stored_situation:
            print(
                f"  │  Situation stored: "
                f"{stored_situation.get('location', '?')} - "
                f"{_trunc(str(stored_situation.get('situation', '')), 120)}",
                flush=True,
            )
        resolution = situation.get("resolution", {})
        if resolution.get("removed_situations"):
            first = resolution["removed_situations"][0]
            print(
                f"  │  Situation resolved: "
                f"{first.get('location', '?')} - "
                f"{_trunc(str(first.get('situation', '')), 120)}",
                flush=True,
            )

        gen = modules.get("action_generation", {})
        aff = gen.get("affordance_brainstorming", {})
        aff_ideas = aff.get("ideas", [])
        if aff_ideas:
            first = aff_ideas[0]
            commands = ", ".join(first.get("commands_to_try", [])[:4])
            suffix = f" -> {commands}" if commands else ""
            print(
                f"  │  Brainstorm ({len(aff_ideas)}): "
                f"{_trunc(str(first.get('situation', '')), 100)}{suffix}",
                flush=True,
            )
        raw = gen.get("llm_raw_response", "")
        if raw and not str(raw).startswith("ERROR"):
            print(f"  │  🤖 LLM Response: {_trunc(str(raw), 200)}", flush=True)

        print(f"  └{'─'*69}", flush=True)
        print(flush=True)

    def _log_step(self, epoch: int, step: int, action: str, observation: str,
                  score: int, reward: int, agent: LPLHAgent):
        """Write a comprehensive per-step log to the run log file (real-time)."""
        if not self._log_file or not agent.step_details:
            return
        d = agent.step_details[-1]
        modules = d.get("modules", {})
        kg = modules.get("kg_map", {})
        act_mod = modules.get("action_validation", {})
        gen = modules.get("action_generation", {})
        exp = modules.get("experience_lib", {})

        reward_str = f" ({'+' if reward >= 0 else ''}{reward})" if reward != 0 else ""
        location = kg.get("current_location") or "?"
        W = 80

        def section(title):
            return f"\n--- {title} {'─' * max(0, W - len(title) - 6)}\n"

        lines = []
        lines.append("\n" + "=" * W)
        lines.append(f"[EPOCH {epoch} | STEP {step:03d}]  Score: {score}{reward_str}  |  Location: {location}")
        lines.append("=" * W)

        # ── Command ───────────────────────────────────────────
        lines.append(section("COMMAND"))
        lines.append(action)

        # ── Game Response ─────────────────────────────────────
        lines.append(section("GAME RESPONSE"))
        lines.append(observation)

        # ── KG Map (full JSON) ────────────────────────────────
        lines.append(section("KG MAP (JSON)"))
        lines.append("[LOCATION RESOLUTION]")
        lines.append(json.dumps(kg.get("location_resolution", {}), indent=2, ensure_ascii=False))
        lines.append("")
        lines.append(kg.get("kg_map_context", gen.get("kg_map_context", kg.get("room_info", "N/A"))))
        inv_rec = modules.get("inventory_reconciliation", {})
        lines.append(section("INVENTORY RECONCILIATION"))
        lines.append(f"Status: {inv_rec.get('status', 'missing')}")
        lines.append(f"Applied: {inv_rec.get('applied', False)}")
        lines.append(f"Before: {inv_rec.get('before', [])}")
        lines.append(f"After: {inv_rec.get('after', [])}")
        lines.append(f"Reason: {inv_rec.get('reason', '')}")
        lines.append("\n[INVENTORY RECONCILER STRUCTURED UPDATE]")
        lines.append(json.dumps(inv_rec.get("raw_update", {}), indent=2, ensure_ascii=False))

        # ── Previous action validation ────────────────────────
        lines.append(section("ACTION VALIDATION"))
        lines.append(f"Previous action valid: {act_mod.get('prev_action_valid')}")
        lines.append(f"Status: {act_mod.get('status', 'validation only')}")

        # ── Experience Library ────────────────────────────────
        lines.append(section("EXPERIENCE LIBRARY"))
        lines.append(f"Total experiences in DB: {exp.get('total_experiences', 0)}")
        lines.append(f"Score changed this step: {exp.get('score_changed', False)}")
        if exp.get("score_changed") and exp.get("new_experience_summary"):
            summary = str(exp["new_experience_summary"])
            if not summary.startswith("ERROR"):
                lines.append("\n[NEW EXPERIENCE STORED (score change)]")
                lines.append(summary)
        neutral_triggers_fired = exp.get("neutral_triggers_fired", [])
        neutral_summaries = exp.get("neutral_summaries", [])
        neutral_summaries_skipped = exp.get("neutral_summaries_skipped", [])
        if neutral_triggers_fired:
            lines.append(f"\nNeutral triggers fired: {', '.join(neutral_triggers_fired)}")
        for trigger_type, summary in neutral_summaries:
            lines.append(f"\n[NEUTRAL EXPERIENCE: {trigger_type.upper()}]")
            lines.append(str(summary))
        for skipped in neutral_summaries_skipped:
            lines.append(f"\n[NEUTRAL EXPERIENCE SKIPPED: {str(skipped.get('trigger', 'neutral')).upper()}]")
            lines.append(f"Reason: {skipped.get('reason', 'duplicate_neutral_event')}")
            lines.append(f"Event key: {skipped.get('event_key', '')}")
        lines.append("\n[RETRIEVED EXPERIENCES]")
        lines.append(str(exp.get("retrieved_experiences", "None")))

        # ── Full Prompt Sent to LLM ───────────────────────────
        # ── LLM Raw Response ──────────────────────────────────
        sit = modules.get("situation_memory", {})
        lines.append(section("STORED SITUATIONS"))
        lines.append(str(gen.get("stored_situations_context")
                         or json.dumps(sit.get("active_situations_after", []),
                                       indent=2, ensure_ascii=False)))

        env = modules.get("environmental_change_detection", {})
        lines.append(section("ENVIRONMENTAL CHANGE DETECTION"))
        lines.append(f"Status: {env.get('status', 'missing')}")
        lines.append(f"Environmental change: {env.get('environmental_change', False)}")
        if env.get("evidence"):
            lines.append(f"Evidence: {env.get('evidence')}")
        if env.get("error"):
            lines.append(f"Error: {env.get('error')}")
        if env.get("response_body"):
            lines.append("\n[DETECTOR RESPONSE BODY]")
            lines.append(str(env.get("response_body")))

        aff = gen.get("affordance_brainstorming", {})
        lines.append(section("AFFORDANCE BRAINSTORMING"))
        lines.append(f"Status: {aff.get('status', 'missing')}")
        if aff.get("error"):
            lines.append(f"Error: {aff.get('error')}")
        lines.append(str(gen.get("affordance_agenda")
                         or gen.get("brainstormed_command_ideas")
                         or aff.get("ideas_for_prompt")
                         or json.dumps(aff.get("ideas", []), indent=2, ensure_ascii=False)))

        fail_mem = modules.get("action_failure_memory", {})
        lines.append(section("ACTION FAILURE MEMORY"))
        lines.append(f"Status: {fail_mem.get('status', 'missing')}")
        lines.append(f"Source location: {fail_mem.get('source_location', '')}")
        lines.append(f"Command: {fail_mem.get('command', '')}")
        if fail_mem.get("error"):
            lines.append(f"Error: {fail_mem.get('error')}")
        if fail_mem.get("stored_failure"):
            lines.append("\n[STORED OR DUPLICATE FAILURE RECORD]")
            lines.append(json.dumps(fail_mem.get("stored_failure"), indent=2, ensure_ascii=False))
        if fail_mem.get("removed_failure"):
            lines.append("\n[REMOVED AFTER LATER SUCCESS]")
            lines.append(json.dumps(fail_mem.get("removed_failure"), indent=2, ensure_ascii=False))
        lines.append("\n[KNOWN FAILURES AT SOURCE LOCATION AFTER STEP]")
        lines.append(json.dumps(
            fail_mem.get("known_failures_here_after", []),
            indent=2,
            ensure_ascii=False,
        ))

        lines.append(section("LLM RAW RESPONSE"))
        lines.append(str(gen.get("llm_raw_response", "N/A")))

        # ── Reasoning (extracted <rea> tag) ───────────────────
        raw = str(gen.get("llm_raw_response", ""))
        rea_match = re.search(r"<rea>(.*?)</rea>", raw, re.DOTALL)
        if rea_match:
            lines.append(section("REASONING (extracted)"))
            lines.append(rea_match.group(1).strip())

        # ── Prev Action Validation ────────────────────────────
        valid = act_mod.get("prev_action_valid")
        split = act_mod.get("action_split")
        if valid is not None:
            lines.append(section("PREV ACTION VALIDATION"))
            lines.append(f"Valid: {valid}")
            if isinstance(split, dict):
                lines.append(f"Verb: {split.get('verb')}  |  Objects: {split.get('objects')}")

        lines.append("\n" + "─" * W + "\n")
        self._log_file.write("\n".join(lines) + "\n")
        self._log_summary_module_step(epoch, step, d)
        self._log_situation_memory_step(epoch, step, d)
        self._log_kg_location_step(epoch, step, d)
        self._log_auxiliary_gate_step(epoch, step, d)
        self._log_affordance_brainstorm_step(epoch, step, d)
        self._log_action_failure_memory_step(epoch, step, d)
        self._log_action_generation_step(epoch, step, d)
        self._log_module_timing_step(epoch, step, d)

    def _log_kg_location_step(self, epoch: int, step: int, detail: dict):
        """Write per-step KG location resolution decisions."""
        if not self._kg_location_log_file:
            return
        modules = detail.get("modules", {})
        kg = modules.get("kg_map", {}) or {}
        metadata = {
            "step": detail.get("step"),
            "action": detail.get("executed_action") or detail.get("prev_action"),
            "current_location": kg.get("current_location"),
            "rooms_visited": kg.get("rooms_visited"),
            "location_resolution": kg.get("location_resolution", {}),
            "extracted_you_in": [
                triple for triple in kg.get("extracted_triples", [])
                if len(triple) >= 3
                and str(triple[0]).strip().lower() == "you"
                and str(triple[1]).strip().lower() == "in"
            ],
            "applied_you_in": [
                triple for triple in kg.get("applied_triples", [])
                if len(triple) >= 3
                and str(triple[0]).strip().lower() == "you"
                and str(triple[1]).strip().lower() == "in"
            ],
        }
        self._kg_location_log_file.write("\n" + "=" * 90 + "\n")
        self._kg_location_log_file.write(f"[EPOCH {epoch} | STEP {step:03d}]\n")
        self._kg_location_log_file.write("-" * 90 + "\n")
        self._kg_location_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        self._kg_location_log_file.write("\n\nobservation:\n")
        self._kg_location_log_file.write(str(detail.get("observation", "")))
        self._kg_location_log_file.write("\n" + "=" * 90 + "\n")

    def _log_summary_module_step(self, epoch: int, step: int, detail: dict):
        """Write prompt+summary records for newly stored summaries."""
        if not self._summary_log_file:
            return

        modules = detail.get("modules", {})
        exp = modules.get("experience_lib", {})
        entries = exp.get("summary_log_entries", [])
        if not entries:
            return

        for idx, entry in enumerate(entries, 1):
            state_type = entry.get("state_type", "unknown")
            metadata = entry.get("metadata", {})
            self._summary_log_file.write("\n" + "=" * 90 + "\n")
            self._summary_log_file.write(
                f"[EPOCH {epoch} | STEP {step:03d} | SUMMARY {idx}] "
                f"state_type: {state_type}\n"
            )
            self._summary_log_file.write("-" * 90 + "\n")
            self._summary_log_file.write("metadata:\n")
            self._summary_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
            self._summary_log_file.write("\n\nstate type:\n")
            self._summary_log_file.write(str(state_type))
            self._summary_log_file.write("\n\nprompt:\n")
            self._summary_log_file.write(str(entry.get("prompt", "")))
            self._summary_log_file.write("\n\nsummary:\n")
            self._summary_log_file.write(str(entry.get("summary", "")))
            self._summary_log_file.write("\n" + "=" * 90 + "\n")

    def _log_situation_memory_step(self, epoch: int, step: int, detail: dict):
        """Write the situation detector prompt/response for every step."""
        if not self._situation_log_file:
            return

        modules = detail.get("modules", {})
        entry = modules.get("situation_memory", {})
        metadata = {
            "step": detail.get("step"),
            "action": detail.get("executed_action") or detail.get("prev_action"),
            "location": entry.get("location"),
            "status": entry.get("status"),
            "inventory": entry.get("inventory", []),
        }

        self._situation_log_file.write("\n" + "=" * 90 + "\n")
        self._situation_log_file.write(
            f"[EPOCH {epoch} | STEP {step:03d}] "
            f"status: {entry.get('status', 'missing')}\n"
        )
        self._situation_log_file.write("-" * 90 + "\n")
        self._situation_log_file.write("metadata:\n")
        self._situation_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        self._situation_log_file.write("\n\naux gate situation_manager decision:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("gate_decision", {}), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\naux gate stored_situation_detection decision:\n")
        self._situation_log_file.write(
            json.dumps(
                entry.get("stored_situation_detection_gate", {}),
                indent=2,
                ensure_ascii=False,
            )
        )
        self._situation_log_file.write("\n\nprompt:\n")
        self._situation_log_file.write(str(entry.get("prompt", "")))
        self._situation_log_file.write("\n\nllm response:\n")
        self._situation_log_file.write(str(entry.get("llm_raw_response", "")))
        self._situation_log_file.write("\n\nfinish reason:\n")
        self._situation_log_file.write(str(entry.get("finish_reason", "")))
        self._situation_log_file.write("\n\nparsed situation:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("parsed_situation"), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\nparsed manager decision:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("parsed_manager_decision", {}), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\nadded situations:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("added_situations", []), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\nupdated situations:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("updated_situations", []), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\nskipped situation updates:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("skipped_updates", []), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\nskipped duplicate adds marked by LLM:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("skipped_adds", []), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\nsame-state tried commands supplied to manager:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("same_state_tried_commands", []), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n\nnew stored situation:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("new_stored_situation"), indent=2, ensure_ascii=False)
        )
        resolution = entry.get("resolution", {})
        self._situation_log_file.write("\n\nresolution status:\n")
        self._situation_log_file.write(str(resolution.get("status", "missing")))
        self._situation_log_file.write("\n\nresolution prompt:\n")
        self._situation_log_file.write(str(resolution.get("prompt", "")))
        self._situation_log_file.write("\n\nresolution llm response:\n")
        self._situation_log_file.write(str(resolution.get("llm_raw_response", "")))
        self._situation_log_file.write("\n\nresolved situations parsed:\n")
        self._situation_log_file.write(
            json.dumps(
                resolution.get("parsed_resolved_situations", []),
                indent=2,
                ensure_ascii=False,
            )
        )
        self._situation_log_file.write("\n\nremoved situations:\n")
        self._situation_log_file.write(
            json.dumps(
                resolution.get("removed_situations", []),
                indent=2,
                ensure_ascii=False,
            )
        )
        if entry.get("error"):
            self._situation_log_file.write("\n\nerror:\n")
            self._situation_log_file.write(str(entry.get("error")))
        self._situation_log_file.write("\n\nall stored situations after step:\n")
        self._situation_log_file.write(
            json.dumps(entry.get("active_situations_after", []), indent=2, ensure_ascii=False)
        )
        self._situation_log_file.write("\n" + "=" * 90 + "\n")

    def _log_auxiliary_gate_step(self, epoch: int, step: int, detail: dict):
        """Write the auxiliary gate prompt/response and routing decision."""
        if not self._auxiliary_gate_log_file:
            return

        modules = detail.get("modules", {})
        entry = modules.get("auxiliary_gate", {})
        if not entry:
            return

        decision = entry.get("decision", {})
        metadata = {
            "step": detail.get("step"),
            "action": detail.get("executed_action") or detail.get("prev_action"),
            "location": entry.get("location"),
            "previous_location": entry.get("previous_location"),
            "status": entry.get("status"),
            "action_valid": entry.get("action_valid"),
            "reward_change": entry.get("reward_change"),
            "rooms_visited_before": entry.get("rooms_visited_before"),
            "inventory_before": entry.get("inventory_before"),
            "current_inventory": entry.get("inventory"),
            "active_situations": entry.get("active_situations"),
            "recent_failed_commands": entry.get("recent_failed_commands"),
            "known_failed_commands_here": entry.get("known_failed_commands_here"),
            "recent_command_outcomes": entry.get("recent_command_outcomes"),
            "same_state_tried_commands": entry.get("same_state_tried_commands"),
            "cached_affordance_ideas_available": entry.get("cached_affordance_ideas_available"),
            "use_legacy_environmental_detection": entry.get("use_legacy_environmental_detection"),
            "use_legacy_summary_trigger_detection": entry.get(
                "use_legacy_summary_trigger_detection"
            ),
        }

        self._auxiliary_gate_log_file.write("\n" + "=" * 90 + "\n")
        self._auxiliary_gate_log_file.write(
            f"[EPOCH {epoch} | STEP {step:03d}] "
            f"status: {entry.get('status', 'missing')}\n"
        )
        self._auxiliary_gate_log_file.write("-" * 90 + "\n")
        self._auxiliary_gate_log_file.write("metadata:\n")
        self._auxiliary_gate_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        self._auxiliary_gate_log_file.write("\n\nobservation:\n")
        self._auxiliary_gate_log_file.write(str(entry.get("observation", "")))
        self._auxiliary_gate_log_file.write("\n\nprompt:\n")
        self._auxiliary_gate_log_file.write(str(entry.get("prompt", "")))
        self._auxiliary_gate_log_file.write("\n\nllm response:\n")
        self._auxiliary_gate_log_file.write(str(entry.get("llm_raw_response", "")))
        self._auxiliary_gate_log_file.write("\n\nfinish reason:\n")
        self._auxiliary_gate_log_file.write(str(entry.get("finish_reason", "")))
        self._auxiliary_gate_log_file.write("\n\nresponse body:\n")
        self._auxiliary_gate_log_file.write(str(entry.get("response_body", "")))
        self._auxiliary_gate_log_file.write("\n\ndecision:\n")
        self._auxiliary_gate_log_file.write(json.dumps(decision, indent=2, ensure_ascii=False))
        self._auxiliary_gate_log_file.write("\n\ninventory reconciliation:\n")
        self._auxiliary_gate_log_file.write(
            json.dumps(
                modules.get("inventory_reconciliation", {}),
                indent=2,
                ensure_ascii=False,
            )
        )
        self._auxiliary_gate_log_file.write("\n\nenvironmental change detail:\n")
        self._auxiliary_gate_log_file.write(
            json.dumps(
                entry.get("environmental_change_detection", {}),
                indent=2,
                ensure_ascii=False,
            )
        )
        if entry.get("error"):
            self._auxiliary_gate_log_file.write("\n\nerror:\n")
            self._auxiliary_gate_log_file.write(str(entry.get("error")))
        self._auxiliary_gate_log_file.write("\n" + "=" * 90 + "\n")

    def _log_module_timing_step(self, epoch: int, step: int, detail: dict):
        """Write per-step module durations for optimization analysis."""
        if not self._timing_log_file:
            return

        modules = detail.get("modules", {})
        timings = modules.get("module_timings", {}) or {}
        if not timings:
            return

        total = timings.get("completed_step_total", 0.0) or 0.0
        sorted_timings = sorted(
            timings.items(),
            key=lambda item: float(item[1] or 0.0),
            reverse=True,
        )
        metadata = {
            "step": detail.get("step"),
            "action": detail.get("executed_action") or detail.get("prev_action"),
            "next_command": detail.get("next_command"),
            "score": detail.get("score"),
            "reward_change": detail.get("reward_change"),
            "location": (modules.get("kg_map", {}) or {}).get("current_location"),
            "total_seconds": total,
        }

        self._timing_log_file.write("\n" + "=" * 90 + "\n")
        self._timing_log_file.write(
            f"[EPOCH {epoch} | STEP {step:03d}] total={total:.4f}s "
            f"action={metadata.get('action')}\n"
        )
        self._timing_log_file.write("-" * 90 + "\n")
        self._timing_log_file.write("metadata:\n")
        self._timing_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        self._timing_log_file.write("\n\nmodule timings, slowest first:\n")
        for name, seconds in sorted_timings:
            try:
                seconds_float = float(seconds)
            except (TypeError, ValueError):
                seconds_float = 0.0
            pct = (seconds_float / total * 100.0) if total else 0.0
            self._timing_log_file.write(f"- {name}: {seconds_float:.4f}s ({pct:.1f}%)\n")
        self._timing_log_file.write("\nraw timings:\n")
        self._timing_log_file.write(json.dumps(timings, indent=2, ensure_ascii=False))
        self._timing_log_file.write("\n" + "=" * 90 + "\n")

    def _log_affordance_brainstorm_step(self, epoch: int, step: int, detail: dict):
        """Write the affordance brainstorm prompt/response for every generated action."""
        if not self._affordance_log_file:
            return

        modules = detail.get("modules", {})
        gen = modules.get("action_generation", {})
        entry = gen.get("affordance_brainstorming", {})
        metadata = {
            "step": detail.get("step"),
            "command_generated": detail.get("final_command") or detail.get("executed_action"),
            "location": entry.get("location"),
            "status": entry.get("status", "missing"),
            "visible_objects": entry.get("visible_objects", []),
            "inventory": entry.get("inventory", []),
            "recent_failed_commands": entry.get("recent_failed_commands", []),
            "failed_commands": entry.get("failed_commands", []),
            "unproductive_commands": entry.get("unproductive_commands", []),
            "failed_command_verbs": entry.get("failed_command_verbs", []),
            "reset_cache": entry.get("reset_cache", False),
            "pending_carryover_commands": entry.get("pending_carryover_commands", []),
        }
        ideas = entry.get("ideas", [])

        self._affordance_log_file.write("\n" + "=" * 90 + "\n")
        self._affordance_log_file.write(
            f"[EPOCH {epoch} | STEP {step:03d}] "
            f"status: {entry.get('status', 'missing')}\n"
        )
        self._affordance_log_file.write("-" * 90 + "\n")
        self._affordance_log_file.write("metadata:\n")
        self._affordance_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        self._affordance_log_file.write("\n\ncurrent observation:\n")
        self._affordance_log_file.write(str(entry.get("observation") or detail.get("observation", "")))
        self._affordance_log_file.write("\n\nactive stored situations supplied to brainstorm:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("active_situations", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nbrainstormed commands:\n")
        if ideas:
            for idx, idea in enumerate(ideas, 1):
                self._affordance_log_file.write(f"{idx}. {idea.get('situation', '')}\n")
                if idea.get("reason"):
                    self._affordance_log_file.write(f"   reason: {idea.get('reason')}\n")
                commands = idea.get("commands_to_try", [])
                self._affordance_log_file.write(
                    "   commands: " + (", ".join(commands) if commands else "none") + "\n"
                )
        else:
            self._affordance_log_file.write("none\n")
        self._affordance_log_file.write("\n\nprompt:\n")
        self._affordance_log_file.write(str(entry.get("prompt", "")))
        self._affordance_log_file.write("\n\nllm response:\n")
        self._affordance_log_file.write(str(entry.get("llm_raw_response", "")))
        self._affordance_log_file.write("\n\nfinish reason:\n")
        self._affordance_log_file.write(str(entry.get("finish_reason", "")))
        self._affordance_log_file.write("\n\nparsed ideas:\n")
        self._affordance_log_file.write(
            json.dumps(ideas, indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nfresh ideas from LLM:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("fresh_ideas", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\ncarried ideas before merge:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("carried_ideas_before", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nfiltered failed commands:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("filtered_failed_commands", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nunproductive commands in this state:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("unproductive_commands", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nsame-state tried commands:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("same_state_tried_commands", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nsame-state tried records:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("same_state_tried_records", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nfailed records here:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("failed_records_here", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\npending carryover commands supplied to brainstorm:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("pending_carryover_commands", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\ncarried ideas after merge:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("carried_ideas_after", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\naffordance agenda sent to action selector:\n")
        self._affordance_log_file.write(
            json.dumps(entry.get("affordance_agenda", []), indent=2, ensure_ascii=False)
        )
        self._affordance_log_file.write("\n\nagenda text sent to action selector:\n")
        self._affordance_log_file.write(
            str(gen.get("affordance_agenda")
                or gen.get("brainstormed_command_ideas")
                or entry.get("ideas_for_prompt", "[]"))
        )
        if entry.get("error"):
            self._affordance_log_file.write("\n\nerror:\n")
            self._affordance_log_file.write(str(entry.get("error")))
        self._affordance_log_file.write("\n" + "=" * 90 + "\n")

    def _log_action_failure_memory_step(self, epoch: int, step: int, detail: dict):
        """Write action-failure memory decisions for every step."""
        if not self._action_failure_log_file:
            return

        modules = detail.get("modules", {})
        entry = modules.get("action_failure_memory", {})
        if not entry:
            return

        metadata = {
            "step": detail.get("step"),
            "action": detail.get("executed_action") or detail.get("prev_action"),
            "status": entry.get("status"),
            "source_location": entry.get("source_location"),
            "command": entry.get("command"),
        }

        self._action_failure_log_file.write("\n" + "=" * 90 + "\n")
        self._action_failure_log_file.write(
            f"[EPOCH {epoch} | STEP {step:03d}] "
            f"status: {entry.get('status', 'missing')}\n"
        )
        self._action_failure_log_file.write("-" * 90 + "\n")
        self._action_failure_log_file.write("metadata:\n")
        self._action_failure_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        self._action_failure_log_file.write("\n\nworld signature:\n")
        self._action_failure_log_file.write(
            json.dumps(entry.get("world_signature", {}), indent=2, ensure_ascii=False)
        )
        self._action_failure_log_file.write("\n\nfailure reason prompt:\n")
        self._action_failure_log_file.write(str(entry.get("failure_reason_prompt", "")))
        self._action_failure_log_file.write("\n\nfailure reason llm response:\n")
        self._action_failure_log_file.write(str(entry.get("failure_reason_raw_response", "")))
        self._action_failure_log_file.write("\n\nstored failure:\n")
        self._action_failure_log_file.write(
            json.dumps(entry.get("stored_failure"), indent=2, ensure_ascii=False)
        )
        self._action_failure_log_file.write("\n\nremoved failure:\n")
        self._action_failure_log_file.write(
            json.dumps(entry.get("removed_failure"), indent=2, ensure_ascii=False)
        )
        if entry.get("error"):
            self._action_failure_log_file.write("\n\nerror:\n")
            self._action_failure_log_file.write(str(entry.get("error")))
        self._action_failure_log_file.write("\n\nknown failures at source location after step:\n")
        self._action_failure_log_file.write(
            json.dumps(
                entry.get("known_failures_here_after", []),
                indent=2,
                ensure_ascii=False,
            )
        )
        self._action_failure_log_file.write("\n" + "=" * 90 + "\n")

    def _log_action_generation_step(self, epoch: int, step: int, detail: dict):
        """Write main action-LLM reasoning and selected commands for every step."""
        if not self._action_generation_log_file:
            return

        modules = detail.get("modules", {})
        self._write_action_generation_record(
            epoch=epoch,
            step=step,
            label="executed_action_generation",
            command=detail.get("final_command") or detail.get("executed_action"),
            generation=modules.get("action_generation", {}),
            observation=detail.get("observation", ""),
            score=detail.get("score"),
        )

        next_generation = modules.get("next_action_generation")
        if next_generation:
            self._write_action_generation_record(
                epoch=epoch,
                step=step,
                label="next_action_generation",
                command=detail.get("next_command"),
                generation=next_generation,
                observation=detail.get("observation", ""),
                score=detail.get("score"),
            )

    def _write_action_generation_record(self, epoch: int, step: int, label: str,
                                        command: str, generation: dict,
                                        observation: str, score):
        if not generation:
            return

        raw = str(generation.get("llm_raw_response", ""))
        reasoning = self._extract_action_reasoning(raw)
        aff = generation.get("affordance_brainstorming", {})
        metadata = {
            "step": step,
            "label": label,
            "command": command or generation.get("parsed_command"),
            "parsed_command": generation.get("parsed_command"),
            "score": score,
            "affordance_status": aff.get("status"),
            "affordance_ideas": len(aff.get("ideas", [])),
            "generation_timings": generation.get("timings", {}),
        }

        self._action_generation_log_file.write("\n" + "=" * 90 + "\n")
        self._action_generation_log_file.write(
            f"[EPOCH {epoch} | STEP {step:03d} | {label}] "
            f"command: {metadata.get('command')}\n"
        )
        self._action_generation_log_file.write("-" * 90 + "\n")
        self._action_generation_log_file.write("metadata:\n")
        self._action_generation_log_file.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        self._action_generation_log_file.write("\n\ncurrent observation:\n")
        self._action_generation_log_file.write(str(observation or ""))
        self._action_generation_log_file.write("\n\nretrieved experiences:\n")
        self._action_generation_log_file.write(str(generation.get("retrieved_experiences", "")))
        self._action_generation_log_file.write("\n\nstored situations context:\n")
        self._action_generation_log_file.write(str(generation.get("stored_situations_context", "[]")))
        self._action_generation_log_file.write("\n\naffordance agenda sent to main LLM:\n")
        self._action_generation_log_file.write(
            str(generation.get("affordance_agenda")
                or generation.get("brainstormed_command_ideas", "[]"))
        )
        self._action_generation_log_file.write("\n\nknown failed commands here:\n")
        self._action_generation_log_file.write(str(generation.get("known_failed_commands_here", "[]")))
        self._action_generation_log_file.write("\n\nsame-state tried commands:\n")
        self._action_generation_log_file.write(str(generation.get("same_state_tried_commands", "[]")))
        self._action_generation_log_file.write("\n\naction generation timings:\n")
        self._action_generation_log_file.write(
            json.dumps(generation.get("timings", {}), indent=2, ensure_ascii=False)
        )
        self._action_generation_log_file.write("\n\nmain LLM extracted reasoning:\n")
        self._action_generation_log_file.write(reasoning or "(none extracted)")
        self._action_generation_log_file.write("\n\nmain LLM raw response:\n")
        self._action_generation_log_file.write(raw or "(empty)")
        self._action_generation_log_file.write("\n" + "=" * 90 + "\n")

    def _extract_action_reasoning(self, raw: str) -> str:
        if not raw:
            return ""
        tag_match = re.search(
            r"<(?:rea|reason|motivation)\b[^>]*>(.*?)</(?:rea|reason|motivation)>",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if tag_match:
            return re.sub(r"\s+", " ", tag_match.group(1)).strip()

        before_start = re.split(r"\|start\|", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        before_start = before_start.replace("Your internal reasoning steps Here.", "")
        before_start = re.sub(r"\s+", " ", before_start).strip()
        return before_start

    def _compute_summary(self, game_name: str, elapsed: float) -> dict:
        """Compute summary statistics across all epochs."""
        all_scores = [r["final_score"] for r in self.epoch_results]
        all_max = [r["max_score"] for r in self.epoch_results]

        # Paper uses last 3 epochs as "learning outcomes"
        last3_scores = all_scores[-3:] if len(all_scores) >= 3 else all_scores

        return {
            "game": game_name,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.LLM_MODEL,
            "avg_score_all": sum(all_scores) / len(all_scores),
            "avg_score_last3": sum(last3_scores) / len(last3_scores),
            "max_score": max(all_max),
            "all_epoch_results": self.epoch_results,
            "partial": False,
            "log_dir_path": self._experiment_log_dir,
            "run_log_path": self._run_log_path,
            "summary_log_path": self._summary_log_path,
            "situation_log_path": self._situation_log_path,
            "affordance_log_path": self._affordance_log_path,
            "action_failure_log_path": self._action_failure_log_path,
            "action_generation_log_path": self._action_generation_log_path,
            "auxiliary_gate_log_path": self._auxiliary_gate_log_path,
            "kg_location_log_path": self._kg_location_log_path,
            "timing_log_path": self._timing_log_path,
            "step_log_path": self._step_log_path,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_results(self, results: dict, game_name: str):
        """Save results to a JSON file."""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        timestamp = self._run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{game_name}_{timestamp}.json"
        filepath = self._results_path or os.path.join(config.DATA_DIR, filename)
        self._results_path = filepath
        results["log_dir_path"] = self._experiment_log_dir
        results["run_log_path"] = self._run_log_path
        results["summary_log_path"] = self._summary_log_path
        results["situation_log_path"] = self._situation_log_path
        results["affordance_log_path"] = self._affordance_log_path
        results["action_failure_log_path"] = self._action_failure_log_path
        results["action_generation_log_path"] = self._action_generation_log_path
        results["auxiliary_gate_log_path"] = self._auxiliary_gate_log_path
        results["kg_location_log_path"] = self._kg_location_log_path
        results["timing_log_path"] = self._timing_log_path
        results["step_log_path"] = self._step_log_path
        results["results_path"] = self._results_path

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        print(f"Results saved to: {filepath}")

    def _save_step_log(self, all_step_logs: list, game_name: str, quiet: bool = False):
        """Save detailed per-step log to a JSON file for post-run analysis."""
        os.makedirs(self._experiment_log_dir or config.LOGS_DIR, exist_ok=True)
        timestamp = self._run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"steplog_{game_name}_{timestamp}.json"
        filepath = self._step_log_path or os.path.join(
            self._experiment_log_dir or config.LOGS_DIR, filename
        )
        self._step_log_path = filepath
        payload = {
            "game": game_name,
            "run_timestamp": timestamp,
            "log_dir_path": self._experiment_log_dir,
            "run_log_path": self._run_log_path,
            "summary_log_path": self._summary_log_path,
            "situation_log_path": self._situation_log_path,
            "affordance_log_path": self._affordance_log_path,
            "action_failure_log_path": self._action_failure_log_path,
            "action_generation_log_path": self._action_generation_log_path,
            "auxiliary_gate_log_path": self._auxiliary_gate_log_path,
            "kg_location_log_path": self._kg_location_log_path,
            "timing_log_path": self._timing_log_path,
            "results_path": self._results_path,
            "epochs": all_step_logs,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        if not quiet:
            print(f"Detailed step log saved to: {filepath}")
