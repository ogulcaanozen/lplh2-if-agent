"""fm client - fine-tuned Qwen2.5-1.5B + LoRA for paper's 3 structured tasks.

Used for action validation, relation extraction, and action splitting. The
adapter is loaded once on instantiation. Generation uses the configured fm
temperature, defaulting to the paper's 0.1.
"""

import re
import logging
from pathlib import Path

from . import config
from .prompts import (
    ACTION_VALIDATION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    ACTION_SPLITTING_PROMPT,
)

logger = logging.getLogger(__name__)


class FmClient:
    """Loads base + LoRA adapter once; exposes the 3 fm-task methods.

    Mirrors LLMClient's interface for validate_action / extract_relations /
    split_action so the agent can route to either without code changes.
    """

    def __init__(self, adapter_path: str = None, base_model: str = None,
                 device: str = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self._torch = torch
        self.adapter_path = Path(adapter_path or config.FM_MODEL_PATH)
        self.base_model_name = base_model or config.FM_BASE_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"fm adapter not found at {self.adapter_path.resolve()}. "
                f"Train via fm_training/train_fm.ipynb or set LPLH_FM_PATH."
            )

        logger.info(f"Loading fm base: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        base.eval()
        logger.info(f"Loading fm LoRA adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self.model.eval()
        logger.info(f"fm ready on {self.device} ({dtype})")

    def _generate(self, prompt: str, max_new: int = 256) -> str:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        temperature = config.FM_TEMPERATURE
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ── fm tasks ──────────────────────────────────────────────

    def validate_action(self, action: str, observation: str) -> bool:
        """Table 4 - returns True if action succeeded, False otherwise."""
        prompt = ACTION_VALIDATION_PROMPT.format(action=action, observation=observation)
        response = self._generate(prompt, max_new=64)
        m = re.search(r"<ais>\s*(True|False)\s*</ais>", response, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
        # Fallback: heuristic on observation. "i don't know" added for Zork's
        # canonical unknown-verb response ("I don't know the word 'X'.").
        fail_phrases = ["you can't", "you cannot", "that's not",
                        "i don't understand", "i don't know",
                        "doesn't seem", "nothing happens", "you don't see"]
        return not any(p in observation.lower() for p in fail_phrases)

    def extract_relations(self, action: str, observation: str, max_new: int = 512) -> list:
        """Table 5 - returns list of (subject, relation, object) triples.

        Deliberately LENIENT (returns [] on malformed), unlike
        eval_fm.ipynb which returns None on malformed for strict scoring.
        For live gameplay the agent passes this list to kg_map.update() which
        iterates it, so returning None would crash. We log a warning instead
        so malformed outputs are visible if v3 starts misbehaving at runtime.
        """
        prompt = RELATION_EXTRACTION_PROMPT.format(action=action, observation=observation)
        response = self._generate(prompt, max_new=max_new)
        m = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        if not m:
            logger.warning(f"fm extract: no |start|...|end| block; raw={response[:120]!r}")
            return []
        body = m.group(1).strip()
        if not body:
            logger.warning(f"fm extract: empty body; raw={response[:120]!r}")
            return []
        if body.lower() == "none":
            return []
        triples = re.findall(r"<([^<>]+)>", body)
        if not triples:
            logger.warning(f"fm extract: body has no <...> triples; body={body[:120]!r}")
            return []
        out = []
        malformed = 0
        for t in triples:
            parts = [p.strip() for p in t.split(",")]
            if len(parts) == 3:
                out.append((parts[0], parts[1], parts[2]))
            else:
                malformed += 1
        if malformed:
            logger.warning(f"fm extract: {malformed} of {len(triples)} triples had wrong field count")
        return out

    def split_action(self, action: str) -> dict:
        """Table 6 - returns {'verb': ..., 'objects': [...]}."""
        prompt = ACTION_SPLITTING_PROMPT.format(action=action)
        response = self._generate(prompt, max_new=64)
        m = re.search(r"<act>\s*<([^;<>]+);\s*\[([^\]]*)\]>\s*</act>", response)
        if m:
            verb = m.group(1).strip()
            objs_str = m.group(2).strip()
            objs = [o.strip().strip("'\"") for o in objs_str.split(",") if o.strip()] if objs_str else []
            return {"verb": verb, "objects": objs}
        # Fallback heuristic
        parts = action.strip().split()
        if len(parts) == 1:
            return {"verb": parts[0], "objects": []}
        return {"verb": parts[0] + " &", "objects": [" ".join(parts[1:])]}
