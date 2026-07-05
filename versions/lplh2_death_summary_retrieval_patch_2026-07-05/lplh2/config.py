"""Configuration for the LPLH2 framework."""

import os

# Game settings
NUM_EPOCHS = 2
MAX_STEPS_PER_EPOCH = 250
HISTORY_LENGTH = 10  # last N turns kept in context

# LLM_a settings: action generation
# Provider: "ollama" (local), "huggingface" (in-process), "openai" (API)
LLM_PROVIDER = os.getenv("LPLH_LLM_PROVIDER", "ollama")
# lplh2 keeps an Ollama-friendly default, but Colab/HF experiments can
# override this with LPLH_LLM_PROVIDER=huggingface and a Qwen2.5 model name.
LLM_MODEL = os.getenv("LPLH_LLM_MODEL", "qwen3:8b")
LLM_TEMPERATURE = 0.6  # paper
LLM_REASONING_EFFORT = os.getenv("LPLH_LLM_REASONING_EFFORT", "").strip()

# fm settings: action validation / relation extraction / action splitting
FM_MODEL_PATH = os.getenv("LPLH_FM_PATH", "fm_adapter_v3_round3/")
FM_BASE_MODEL = os.getenv("LPLH_FM_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
FM_TEMPERATURE = float(os.getenv("LPLH_FM_TEMPERATURE", "0.1"))

# Keep learned valid actions across epochs unless explicitly disabled.
PERSIST_ACTION_SPACE = os.getenv("LPLH_PERSIST_ACTION_SPACE", "true").lower() in (
    "1", "true", "yes", "on"
)

# LLM_es settings: summaries and LPLH2 auxiliary reasoning modules.
# Set a model name such as "o3-mini" or "gpt-4.1" to use OpenAI for aux calls. Set ""
# to run aux/summarization calls on LLM_a.
LLM_ES_MODEL = os.getenv("LPLH_LLM_ES_MODEL", "")
LLM_AUX_FALLBACK_LABEL = os.getenv("LPLH_LLM_AUX_FALLBACK_LABEL", "LLM_a fallback")

# Optional dedicated model for affordance brainstorming. When set, only the
# brainstormer uses this OpenAI model; all other auxiliary modules keep using
# LLM_ES_MODEL or the LLM_a fallback above.
LLM_BRAINSTORM_MODEL = os.getenv("LPLH_BRAINSTORM_MODEL", "")
LLM_BRAINSTORM_FALLBACK_LABEL = os.getenv(
    "LPLH_BRAINSTORM_FALLBACK_LABEL", LLM_AUX_FALLBACK_LABEL
)
LLM_BRAINSTORM_REASONING_EFFORT = os.getenv(
    "LPLH_BRAINSTORM_REASONING_EFFORT", "low"
).strip()
LLM_BRAINSTORM_FALLBACK_MAX_NEW_TOKENS = int(
    os.getenv("LPLH_BRAINSTORM_FALLBACK_MAX_NEW_TOKENS", "768")
)

# Experience library settings
EXPERIENCE_TOP_K = 3
EXPERIENCE_FETCH_K = int(os.getenv("LPLH_EXPERIENCE_FETCH_K", "8"))
EXPERIENCE_RENDER_DIVERSITY = os.getenv(
    "LPLH_EXPERIENCE_RENDER_DIVERSITY", "true"
).lower() in ("1", "true", "yes", "on")
CHROMA_COLLECTION = "lplh_experiences"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
EXPERIENCE_INDEX_DIR = os.getenv(
    "LPLH_EXPERIENCE_INDEX_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "experience_index"),
)

# Paths
GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "games")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("LPLH_OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_RETRIES = int(os.getenv("LPLH_OPENAI_MAX_RETRIES", "1"))
