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

# fm settings: action validation / relation extraction / action splitting
FM_MODEL_PATH = os.getenv("LPLH_FM_PATH", "fm_adapter_v3_round3/")
FM_BASE_MODEL = os.getenv("LPLH_FM_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
FM_TEMPERATURE = float(os.getenv("LPLH_FM_TEMPERATURE", "0.1"))

# Compatibility flag for old notebooks/logs. New epoch resets keep only the
# Experience Library; action space is learned fresh inside each epoch.
PERSIST_ACTION_SPACE = os.getenv("LPLH_PERSIST_ACTION_SPACE", "false").lower() in (
    "1", "true", "yes", "on"
)

# LLM_es settings: summaries and LPLH2 auxiliary reasoning modules.
# Paper uses GPT-o3-mini for experience summarization. Set "" to fall back
# to LLM_a doing double-duty for all aux/summarization calls.
LLM_ES_MODEL = os.getenv("LPLH_LLM_ES_MODEL", "o3-mini")

# Experience library settings
EXPERIENCE_TOP_K = 3
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
