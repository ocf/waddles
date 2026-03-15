"""Configuration and environment variables for the OCF bot."""

import os

# --- Discord Settings ---
TOKEN = os.getenv("DISCORD_TOKEN")
PREFIX = "?"

OWNER_IDS = set([
    446290930723717120,
    1023113941624295434,
    330550318305640458,
])

# Provide a fallback just in case it's missing in .env
ADMIN_ROLE_ID = 735620451295821906

# --- LLM Settings ---
OLLAMA_URL = "http://127.0.0.1:11434"
SGLANG_URL = "http://127.0.0.1:30000/v1"
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
EMBEDDING_NAME = "qwen3-embedding:8b"

# --- Paths ---
DOCS_DIR = "/app/docs"
STORAGE_DIR = "/app/storage"
DATA_DIR = "/app/data"
PERSONA_DIR = f"{DATA_DIR}/persona"
SETTINGS_DIR = f"{DATA_DIR}/settings"
MEMORY_DIR = f"{DATA_DIR}/memory"
SYNC_SCRIPT = "/app/sync.sh"

# Ensure directories exist
os.makedirs(PERSONA_DIR, exist_ok=True)
os.makedirs(SETTINGS_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# --- LLM Configuration ---
LLM_CONTEXT_WINDOW = 32768
LLM_TIMEOUT = 360.0
LLM_TEMPERATURE = 0.1
LLM_FREQUENCY_PENALTY = 0.1
LLM_PRESENCE_PENALTY = 0.3
LLM_REPETITION_PENALTY = 1.05
LLM_MIN_P = 0.05

# --- Embedding Configuration ---
EMBED_BATCH_SIZE = 128
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100

# --- Tool Configuration ---
DOCS_RETRIEVE_CHUNKS = 20
DOCS_TOP_N = 6
DOCS_RECENCY_WEIGHT = 0.3
WEB_SEARCH_MAX_RESULTS = 3
WEB_SCRAPE_MAX_LINKS = 20
WEB_SCRAPE_MAX_LEN = 10000

# --- Memory Configuration ---
MEMORY_MAX_FACTS = 20
MEMORY_TOKEN_LIMIT = 4000
MEMORY_CHAT_HISTORY_TOKEN_RATIO = 0.7
MEMORY_DB_PATH = os.path.join(MEMORY_DIR, "memory.db")
MEMORY_DB_URI = f"sqlite+aiosqlite:///{MEMORY_DB_PATH}"
