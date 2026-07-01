"""LLM Client — unified wrapper for action generation and experience summarization.

Providers:
  - "ollama"      : local, free
  - "huggingface" : in-process (Colab / GPU box), for paper-faithful Qwen2.5-7B
  - "openai"      : API, for GPT-o3-mini etc.

The 3 fm tasks (validate_action, extract_relations, split_action) live in
FmClient. Summaries and LPLH2 auxiliary reasoning passes use LLM_es
(paper: GPT-o3-mini via OpenAI) when configured, otherwise they fall back to
LLM_a doing double-duty.
"""

import re
import json
import logging

from . import config
from .prompts import (
    EXPERIENCE_SUMMARIZATION_PROMPT,
    NAVIGATION_EXPERIENCE_PROMPT,
    NARRATIVE_EXPERIENCE_PROMPT,
    ENVIRONMENTAL_CHANGE_PROMPT,
    ENVIRONMENTAL_CHANGE_DETECTION_PROMPT,
    AUXILIARY_MODULE_GATE_PROMPT,
    SITUATION_MANAGER_PROMPT,
    ERROR_CORRECTION_PROMPT,
    STORED_SITUATION_DETECTION_PROMPT,
    STORED_SITUATION_RESOLUTION_PROMPT,
    AFFORDANCE_BRAINSTORMING_PROMPT,
    ACTION_FAILURE_REASON_PROMPT,
    ACTION_REPETITION_EVALUATION_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client for action generation + experience summarization."""

    def __init__(self, provider=None, model=None, temperature=None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

        if self.provider == "ollama":
            import ollama
            self._ollama = ollama.Client(host=config.OLLAMA_BASE_URL)
            self._hf_model = None
            self._hf_tokenizer = None
        elif self.provider == "huggingface":
            self._setup_huggingface()
            self._ollama = None
        elif self.provider == "openai":
            from openai import OpenAI
            self._openai = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=config.OPENAI_TIMEOUT_SECONDS,
                max_retries=config.OPENAI_MAX_RETRIES,
            )
            self._ollama = None
            self._hf_model = None
            self._hf_tokenizer = None
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        # Optional separate client for summaries and LPLH2 auxiliary passes.
        # Gated on BOTH model name and API key so local users without an
        # OpenAI key don't trip up on LLMClient init.
        self._es_client = None
        if config.LLM_ES_MODEL and config.OPENAI_API_KEY:
            from openai import OpenAI
            self._es_client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=config.OPENAI_TIMEOUT_SECONDS,
                max_retries=config.OPENAI_MAX_RETRIES,
            )
            logger.info(f"Experience summarizer: openai/{config.LLM_ES_MODEL}")
        elif config.LLM_ES_MODEL:
            logger.warning(
                f"LLM_ES_MODEL is set ({config.LLM_ES_MODEL}) but OPENAI_API_KEY is empty — "
                f"experience summarization will fall back to LLM_a."
            )

        # Optional dedicated OpenAI client for affordance brainstorming only.
        self._brainstorm_client = None
        if config.LLM_BRAINSTORM_MODEL and config.OPENAI_API_KEY:
            from openai import OpenAI
            self._brainstorm_client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=config.OPENAI_TIMEOUT_SECONDS,
                max_retries=config.OPENAI_MAX_RETRIES,
            )
            effort_note = self._brainstorm_reasoning_effort_note()
            logger.info(
                f"Affordance brainstormer: openai/{config.LLM_BRAINSTORM_MODEL}"
                f"{effort_note}"
            )
        elif config.LLM_BRAINSTORM_MODEL:
            logger.warning(
                f"LLM_BRAINSTORM_MODEL is set ({config.LLM_BRAINSTORM_MODEL}) but "
                f"OPENAI_API_KEY is empty - affordance brainstorming will fall back."
            )

        self.last_summary_prompt = None
        self.last_summary_raw_response = None
        self.last_summary_kind = None
        self.last_situation_prompt = None
        self.last_situation_raw_response = None
        self.last_situation_finish_reason = None
        self.last_situation_resolution_prompt = None
        self.last_situation_resolution_raw_response = None
        self.last_situation_resolution_finish_reason = None
        self.last_situation_manager_prompt = None
        self.last_situation_manager_raw_response = None
        self.last_situation_manager_finish_reason = None
        self.last_environmental_change_prompt = None
        self.last_environmental_change_raw_response = None
        self.last_environmental_change_finish_reason = None
        self.last_affordance_prompt = None
        self.last_affordance_raw_response = None
        self.last_affordance_finish_reason = None
        self.last_failure_reason_prompt = None
        self.last_failure_reason_raw_response = None
        self.last_failure_reason_finish_reason = None
        self.last_repetition_eval_prompt = None
        self.last_repetition_eval_raw_response = None
        self.last_repetition_eval_finish_reason = None
        self.last_auxiliary_gate_prompt = None
        self.last_auxiliary_gate_raw_response = None
        self.last_auxiliary_gate_finish_reason = None

        # qwen3 thinking-mode probe (Ollama only — qwen2.5 has no such mode)
        self._thinking_supported = self._probe_thinking_support()

        logger.info(
            f"LLM Client initialized: provider={self.provider}, model={self.model}, "
            f"thinking={'yes' if self._thinking_supported else 'no'}"
        )

    def _setup_huggingface(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._torch = torch
        self._hf_device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (torch.bfloat16
                 if (self._hf_device == "cuda" and torch.cuda.is_bf16_supported())
                 else torch.float16)
        logger.info(f"Loading HF model {self.model} on {self._hf_device} ({dtype})")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model, dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        self._hf_model.eval()

    def _probe_thinking_support(self) -> bool:
        if self.provider != "ollama":
            return False
        try:
            self._ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                options={"temperature": 0.0, "num_predict": 1},
                think=True,
            )
            return True
        except Exception:
            return False

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = None,
             think: bool = False, max_new_tokens: int = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        if self.provider == "ollama":
            return self._chat_ollama(
                system_prompt, user_prompt, temp, think=think,
                max_new_tokens=max_new_tokens,
            )
        if self.provider == "huggingface":
            return self._chat_hf(system_prompt, user_prompt, temp,
                                 max_new_tokens=max_new_tokens)
        if self.provider == "openai":
            return self._chat_openai(system_prompt, user_prompt, temp,
                                     max_new_tokens=max_new_tokens)

    def _chat_ollama(self, system_prompt, user_prompt, temperature, think=False,
                     max_new_tokens=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        use_think = think and self._thinking_supported
        options = {"temperature": temperature}
        if max_new_tokens:
            options["num_predict"] = int(max_new_tokens)
        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
            think=use_think,
        )
        return response["message"]["content"]

    def _chat_hf(self, system_prompt, user_prompt, temperature,
                 max_new_tokens=None):
        torch = self._torch
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        text = self._hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._hf_tokenizer(text, return_tensors="pt").to(self._hf_device)
        # Only pass sampling kwargs when we're actually sampling — otherwise
        # transformers emits a "temperature is ignored under greedy decoding"
        # warning every call.
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens or 1024),
            "do_sample": temperature > 0,
            "pad_token_id": self._hf_tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        with torch.no_grad():
            out = self._hf_model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        return self._hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _chat_openai(self, system_prompt, user_prompt, temperature,
                     max_new_tokens=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_new_tokens:
            kwargs["max_completion_tokens"] = int(max_new_tokens)
        response = self._openai.chat.completions.create(
            **kwargs,
        )
        return response.choices[0].message.content

    def _chat_aux_fallback(self, prompt: str, max_new_tokens: int) -> str:
        """Run auxiliary work on LLM_a with deterministic decoding."""
        return self.chat(
            "",
            prompt,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

    def _chat_es_once(self, prompt: str, max_completion_tokens: int) -> tuple[str, str]:
        """Run one OpenAI aux/summarization call and return visible text + finish reason."""
        resp = self._es_client.chat.completions.create(
            model=config.LLM_ES_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_completion_tokens,
        )
        choice = resp.choices[0]
        content = getattr(choice.message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", "") or ""
        return content, finish_reason

    def _chat_es_json(self, prompt: str, max_completion_tokens: int,
                      retry_instruction: str = "", retry_tokens: int = None) -> tuple[str, str]:
        """Run an aux JSON call, retrying once when the visible response is empty/truncated."""
        response, finish_reason = self._chat_es_once(prompt, max_completion_tokens)
        should_retry = (not str(response or "").strip()) or finish_reason == "length"
        if should_retry and retry_instruction:
            retry_prompt = f"{prompt}\n\n{retry_instruction}"
            retry_response, retry_finish_reason = self._chat_es_once(
                retry_prompt,
                retry_tokens or max_completion_tokens,
            )
            if str(retry_response or "").strip():
                return retry_response, f"{finish_reason or 'empty'} -> retry:{retry_finish_reason or 'unknown'}"
            return response, f"{finish_reason or 'empty'} -> retry_empty:{retry_finish_reason or 'unknown'}"
        return response, finish_reason

    # ── Experience summarization (LLM_es) ──────────────────────

    def _brainstorm_supports_reasoning_effort(self) -> bool:
        model_name = (config.LLM_BRAINSTORM_MODEL or "").lower()
        return model_name.startswith(("o1", "o3", "o4"))

    def _brainstorm_reasoning_effort_note(self) -> str:
        effort = config.LLM_BRAINSTORM_REASONING_EFFORT
        if effort and self._brainstorm_supports_reasoning_effort():
            return f" (reasoning_effort={effort})"
        return ""

    def _chat_brainstorm_once(self, prompt: str,
                              max_completion_tokens: int) -> tuple[str, str]:
        """Run one dedicated OpenAI affordance-brainstorm call."""
        kwargs = {
            "model": config.LLM_BRAINSTORM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
        }
        effort = config.LLM_BRAINSTORM_REASONING_EFFORT
        if effort and self._brainstorm_supports_reasoning_effort():
            kwargs["reasoning_effort"] = effort
        resp = self._brainstorm_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = getattr(choice.message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", "") or ""
        return content, finish_reason

    def _chat_brainstorm_json(self, prompt: str, max_completion_tokens: int,
                              retry_instruction: str = "",
                              retry_tokens: int = None) -> tuple[str, str]:
        """Run a dedicated brainstorm JSON call, retrying empty/truncated output."""
        response, finish_reason = self._chat_brainstorm_once(
            prompt, max_completion_tokens
        )
        should_retry = (not str(response or "").strip()) or finish_reason == "length"
        if should_retry and retry_instruction:
            retry_prompt = f"{prompt}\n\n{retry_instruction}"
            retry_response, retry_finish_reason = self._chat_brainstorm_once(
                retry_prompt,
                retry_tokens or max_completion_tokens,
            )
            if str(retry_response or "").strip():
                return retry_response, f"{finish_reason or 'empty'} -> retry:{retry_finish_reason or 'unknown'}"
            return response, f"{finish_reason or 'empty'} -> retry_empty:{retry_finish_reason or 'unknown'}"
        return response, finish_reason

    def summarize_experience(self, history: str, reward_change: int,
                              current_score: int) -> str:
        """Paper Table 8 — LLM_es summarizes the recent history into a
        structured experience for ChromaDB storage. Uses GPT-o3-mini if
        config.LLM_ES_MODEL is set; otherwise falls back to LLM_a.
        """
        prompt = EXPERIENCE_SUMMARIZATION_PROMPT.format(
            history=history, reward_change=reward_change, current_score=current_score,
        )
        self.last_summary_kind = "score_change"
        self.last_summary_prompt = prompt
        self.last_summary_raw_response = None
        if self._es_client and config.LLM_ES_MODEL:
            # o3-mini is a reasoning model: no `temperature` kwarg, uses
            # `max_completion_tokens` instead of `max_tokens`.
            resp = self._es_client.chat.completions.create(
                model=config.LLM_ES_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2048,
            )
            response = resp.choices[0].message.content
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=2048)

        self.last_summary_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def detect_stored_situation(self, location: str, action: str,
                                observation: str, inventory: list,
                                stored_situations: list) -> str:
        """Detect one unresolved future-return situation for LPLH2 memory."""
        prompt = STORED_SITUATION_DETECTION_PROMPT.format(
            location=location or "unknown",
            action=action or "none",
            observation=observation or "",
            inventory=json.dumps(inventory or [], ensure_ascii=False),
            stored_situations=json.dumps(stored_situations or [], ensure_ascii=False),
        )

        self.last_situation_prompt = prompt
        self.last_situation_raw_response = None
        self.last_situation_finish_reason = None
        if self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=1536,
                retry_instruction=(
                    "The previous response was empty or truncated. Re-evaluate the exact "
                    "observation and return only the required |start|...|end| JSON object "
                    "or |start| none |end|. Do not explain."
                ),
                retry_tokens=1536,
            )
            self.last_situation_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=1536)
            self.last_situation_finish_reason = "llm_a_qwen14b"

        self.last_situation_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def resolve_stored_situations(self, location: str, action: str,
                                  observation: str, inventory: list,
                                  score: int, reward_change: int,
                                  active_situations: list) -> str:
        """Decide which active stored situations are directly solved now."""
        prompt = STORED_SITUATION_RESOLUTION_PROMPT.format(
            location=location or "unknown",
            action=action or "none",
            observation=observation or "",
            inventory=json.dumps(inventory or [], ensure_ascii=False),
            score=score,
            reward_change=reward_change,
            active_situations=json.dumps(active_situations or [], ensure_ascii=False),
        )

        self.last_situation_resolution_prompt = prompt
        self.last_situation_resolution_raw_response = None
        self.last_situation_resolution_finish_reason = None
        if self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=1536,
                retry_instruction=(
                    "The previous response was empty or truncated. Return only the "
                    "required |start|...|end| JSON list, or |start| [] |end|. Do not explain."
                ),
                retry_tokens=1536,
            )
            self.last_situation_resolution_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=1536)
            self.last_situation_resolution_finish_reason = "llm_a_qwen14b"

        self.last_situation_resolution_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def manage_situations(self, location: str, previous_location: str,
                          action: str, action_valid, command_outcome: dict,
                          observation: str, score: int, reward_change: int,
                          inventory_before: list, inventory: list,
                          visible_objects: list, active_situations: list,
                          active_plan: dict | None,
                          recent_failed_commands: list,
                          known_failed_commands_here: str,
                          recent_command_outcomes: list,
                          same_state_tried_commands: list) -> str:
        """Manage stored situations and one advisory active plan in one call."""
        prompt = SITUATION_MANAGER_PROMPT.format(
            location=location or "unknown",
            previous_location=previous_location or "unknown",
            action=action or "none",
            action_valid=str(action_valid),
            command_outcome=json.dumps(command_outcome or {}, ensure_ascii=False),
            observation=observation or "",
            score=score,
            reward_change=reward_change,
            inventory_before=json.dumps(inventory_before or [], ensure_ascii=False),
            inventory=json.dumps(inventory or [], ensure_ascii=False),
            visible_objects=json.dumps(visible_objects or [], ensure_ascii=False),
            active_situations=json.dumps(active_situations or [], ensure_ascii=False),
            active_plan=json.dumps(active_plan or None, ensure_ascii=False),
            recent_failed_commands=json.dumps(recent_failed_commands or [], ensure_ascii=False),
            known_failed_commands_here=known_failed_commands_here or "[]",
            recent_command_outcomes=json.dumps(recent_command_outcomes or [], ensure_ascii=False),
            same_state_tried_commands=json.dumps(same_state_tried_commands or [], ensure_ascii=False),
        )

        self.last_situation_manager_prompt = prompt
        self.last_situation_manager_raw_response = None
        self.last_situation_manager_finish_reason = None
        if self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=2048,
                retry_instruction=(
                    "The previous response was empty or truncated. Return only the "
                    "required |start|...|end| JSON object. Do not explain."
                ),
                retry_tokens=2048,
            )
            self.last_situation_manager_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=2048)
            self.last_situation_manager_finish_reason = "llm_a_qwen14b"

        self.last_situation_manager_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def detect_environmental_change(self, location: str, action: str,
                                    observation: str, inventory: list,
                                    visible_objects: list,
                                    active_situations: list) -> str:
        """Decide whether a valid action directly changed world state."""
        prompt = ENVIRONMENTAL_CHANGE_DETECTION_PROMPT.format(
            location=location or "unknown",
            action=action or "none",
            observation=observation or "",
            inventory=json.dumps(inventory or [], ensure_ascii=False),
            visible_objects=json.dumps(visible_objects or [], ensure_ascii=False),
            active_situations=json.dumps(active_situations or [], ensure_ascii=False),
        )

        self.last_environmental_change_prompt = prompt
        self.last_environmental_change_raw_response = None
        self.last_environmental_change_finish_reason = None
        if self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=768,
                retry_instruction=(
                    "The previous response was empty or truncated. Return only the "
                    "required |start|...|end| JSON object. Do not explain."
                ),
                retry_tokens=768,
            )
            self.last_environmental_change_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=768)
            self.last_environmental_change_finish_reason = "llm_a_qwen14b"

        self.last_environmental_change_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def gate_auxiliary_modules(self, location: str, previous_location: str,
                               action: str, action_valid, observation: str,
                               score: int, reward_change: int,
                               rooms_visited_before: list,
                               inventory_before: list,
                               inventory: list, visible_objects: list,
                               active_situations: list,
                               active_plan: dict | None,
                               recent_failed_commands: list,
                               known_failed_commands_here: str,
                               recent_command_outcomes: list,
                               same_state_tried_commands: list,
                               cached_affordance_ideas_available: int) -> str:
        """Route selected auxiliary modules for the latest completed step."""
        prompt = AUXILIARY_MODULE_GATE_PROMPT.format(
            location=location or "unknown",
            previous_location=previous_location or "unknown",
            action=action or "none",
            action_valid=str(action_valid),
            observation=observation or "",
            score=score,
            reward_change=reward_change,
            rooms_visited_before=json.dumps(rooms_visited_before or [], ensure_ascii=False),
            inventory_before=json.dumps(inventory_before or [], ensure_ascii=False),
            inventory=json.dumps(inventory or [], ensure_ascii=False),
            visible_objects=json.dumps(visible_objects or [], ensure_ascii=False),
            active_situations=json.dumps(active_situations or [], ensure_ascii=False),
            active_plan=json.dumps(active_plan or None, ensure_ascii=False),
            recent_failed_commands=json.dumps(recent_failed_commands or [], ensure_ascii=False),
            known_failed_commands_here=known_failed_commands_here or "[]",
            recent_command_outcomes=json.dumps(recent_command_outcomes or [], ensure_ascii=False),
            same_state_tried_commands=json.dumps(same_state_tried_commands or [], ensure_ascii=False),
            cached_affordance_ideas_available=int(cached_affordance_ideas_available or 0),
        )

        self.last_auxiliary_gate_prompt = prompt
        self.last_auxiliary_gate_raw_response = None
        self.last_auxiliary_gate_finish_reason = None
        if self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=1536,
                retry_instruction=(
                    "The previous response was empty or truncated. Return only the "
                    "required |start|...|end| JSON object. Do not explain."
                ),
                retry_tokens=1536,
            )
            self.last_auxiliary_gate_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=1536)
            self.last_auxiliary_gate_finish_reason = "llm_a_qwen14b"

        self.last_auxiliary_gate_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def brainstorm_affordances(self, location: str, observation: str,
                               visible_objects: list, inventory: list,
                               recent_failed_commands: list,
                               known_failed_commands_here: str,
                               recent_command_outcomes: list,
                               failed_command_verbs: list,
                               unproductive_commands_here: list,
                               same_state_tried_commands: list,
                               pending_carryover_commands: list,
                               stored_situations: list,
                               active_plan: dict | None,
                               action_space: str,
                               experiences: str,
                               score: int = 0) -> str:
        """Suggest concrete commands for visible objects, inventory, and stored situations."""
        prompt = AFFORDANCE_BRAINSTORMING_PROMPT.format(
            location=location or "unknown",
            observation=observation or "",
            visible_objects=json.dumps(visible_objects or [], ensure_ascii=False),
            inventory=json.dumps(inventory or [], ensure_ascii=False),
            recent_failed_commands=json.dumps(recent_failed_commands or [], ensure_ascii=False),
            known_failed_commands_here=known_failed_commands_here or "[]",
            recent_command_outcomes=json.dumps(recent_command_outcomes or [], ensure_ascii=False),
            failed_command_verbs=json.dumps(failed_command_verbs or [], ensure_ascii=False),
            unproductive_commands_here=json.dumps(unproductive_commands_here or [], ensure_ascii=False),
            same_state_tried_commands=json.dumps(same_state_tried_commands or [], ensure_ascii=False),
            pending_carryover_commands=json.dumps(pending_carryover_commands or [], ensure_ascii=False),
            stored_situations=json.dumps(stored_situations or [], ensure_ascii=False),
            active_plan=json.dumps(active_plan or None, ensure_ascii=False),
            action_space=action_space or "No learned action-space context available.",
            experiences=experiences or "No relevant experiences found yet.",
            score=score,
        )

        self.last_affordance_prompt = prompt
        self.last_affordance_raw_response = None
        self.last_affordance_finish_reason = None
        if self._brainstorm_client and config.LLM_BRAINSTORM_MODEL:
            response, finish_reason = self._chat_brainstorm_json(
                prompt,
                max_completion_tokens=4096,
                retry_instruction=(
                    "The previous response was empty or truncated. Return compact valid JSON only: "
                    "at most 3 objects, at most 3 commands per object, no extra prose. "
                    "If there are no useful ideas, return exactly |start| [] |end|."
                ),
                retry_tokens=3072,
            )
            self.last_affordance_finish_reason = (
                f"{finish_reason}; model=openai/{config.LLM_BRAINSTORM_MODEL}"
                f"{self._brainstorm_reasoning_effort_note()}"
            )
        elif self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=4096,
                retry_instruction=(
                    "The previous response was empty or truncated. Return compact valid JSON only: "
                    "at most 3 objects, at most 3 commands per object, no extra prose. "
                    "If there are no useful ideas, return exactly |start| [] |end|."
                ),
                retry_tokens=3072,
            )
            self.last_affordance_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=3072)
            self.last_affordance_finish_reason = "llm_a_qwen14b"

        self.last_affordance_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def explain_action_failure(self, location: str, command: str,
                               observation: str, world_signature: dict) -> str:
        """Write a short free-text reason for a failed command."""
        prompt = ACTION_FAILURE_REASON_PROMPT.format(
            location=location or "unknown",
            command=command or "",
            observation=observation or "",
            world_signature=json.dumps(world_signature or {}, ensure_ascii=False),
        )

        self.last_failure_reason_prompt = prompt
        self.last_failure_reason_raw_response = None
        self.last_failure_reason_finish_reason = None
        if self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=768,
                retry_instruction=(
                    "The previous response was empty or truncated. Return only the required "
                    "|start|...|end| JSON object. Do not explain."
                ),
                retry_tokens=768,
            )
            self.last_failure_reason_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=768)
            self.last_failure_reason_finish_reason = "llm_a_qwen14b"

        self.last_failure_reason_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        body = m.group(1).strip() if m else response.strip()
        try:
            parsed = json.loads(body)
            reason = parsed.get("failure_reason", "")
            if reason:
                return re.sub(r"\s+", " ", str(reason)).strip()
        except Exception:
            pass
        return re.sub(r"\s+", " ", body).strip()

    def evaluate_action_repetition(self, state_snapshot: dict, command: str,
                                   observation: str, progress_signals: dict) -> str:
        """Judge whether a no-progress action should become same-state memory."""
        prompt = ACTION_REPETITION_EVALUATION_PROMPT.format(
            state_snapshot=json.dumps(state_snapshot or {}, ensure_ascii=False),
            command=command or "",
            observation=observation or "",
            progress_signals=json.dumps(progress_signals or {}, ensure_ascii=False),
        )

        self.last_repetition_eval_prompt = prompt
        self.last_repetition_eval_raw_response = None
        self.last_repetition_eval_finish_reason = None
        if self._es_client and config.LLM_ES_MODEL:
            response, finish_reason = self._chat_es_json(
                prompt,
                max_completion_tokens=768,
                retry_instruction=(
                    "The previous response was empty or truncated. Return only the required "
                    "|start|...|end| JSON object. Do not explain."
                ),
                retry_tokens=768,
            )
            self.last_repetition_eval_finish_reason = finish_reason
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=768)
            self.last_repetition_eval_finish_reason = "llm_a_qwen14b"

        self.last_repetition_eval_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()

    def summarize_neutral_experience(self, trigger: str, action: str,
                                      observation: str, location: str,
                                      prev_location: str = None,
                                      failed_attempts: list = None) -> str:
        """Generate an experience summary for an LPLH2 neutral-state trigger."""
        if trigger == "navigation":
            prompt = NAVIGATION_EXPERIENCE_PROMPT.format(
                location=location,
                prev_location=prev_location or "unknown",
                action=action,
                observation=observation,
            )
        elif trigger == "narrative":
            prompt = NARRATIVE_EXPERIENCE_PROMPT.format(
                location=location,
                action=action,
                observation=observation,
            )
        elif trigger == "environmental":
            prompt = ENVIRONMENTAL_CHANGE_PROMPT.format(
                location=location,
                action=action,
                observation=observation,
            )
        elif trigger == "error_correction":
            failed_str = ", ".join(failed_attempts) if failed_attempts else "none recorded"
            prompt = ERROR_CORRECTION_PROMPT.format(
                location=location,
                action=action,
                observation=observation,
                failed_attempts=failed_str,
            )
        else:
            logger.warning(f"Unknown neutral trigger type: {trigger}")
            return ""

        self.last_summary_kind = trigger
        self.last_summary_prompt = prompt
        self.last_summary_raw_response = None
        if self._es_client and config.LLM_ES_MODEL:
            resp = self._es_client.chat.completions.create(
                model=config.LLM_ES_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2048,
            )
            response = resp.choices[0].message.content
        else:
            response = self._chat_aux_fallback(prompt, max_new_tokens=2048)

        self.last_summary_raw_response = response
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        return m.group(1).strip() if m else response.strip()
