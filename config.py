"""Configuration and environment variables for the OCF bot."""

import os

# --- Discord Settings ---
TOKEN = os.getenv("DISCORD_TOKEN")
PREFIX = "?"

OWNER_USERS_STR = os.getenv("OWNER_USERS", "")
OWNER_IDS = {int(x.strip()) for x in OWNER_USERS_STR.split(",") if x.strip().isdigit()}

# Provide a fallback just in case it's missing in .env
ADMIN_ROLE_ID = int(os.getenv("ADMIN_ROLE_ID", "735620451295821906"))

# --- LLM Settings ---
OLLAMA_URL = "http://127.0.0.1:11434"
SGLANG_URL = "http://127.0.0.1:30000/v1"
MODEL_NAME = "qwen3.5:35b"
EMBEDDING_NAME = "qwen3-embedding:8b"

# --- Paths ---
DOCS_DIR = "/app/docs"
STORAGE_DIR = "/app/storage"
DATA_DIR = "/app/data"
PERSONA_DIR = f"{DATA_DIR}/persona"
SETTINGS_DIR = f"{DATA_DIR}/settings"
SYNC_SCRIPT = "/app/sync.sh"

# Ensure directories exist
os.makedirs(PERSONA_DIR, exist_ok=True)
os.makedirs(SETTINGS_DIR, exist_ok=True)

# --- LLM Configuration ---
LLM_CONTEXT_WINDOW = 32768
LLM_TIMEOUT = 360.0
LLM_TEMPERATURE = 0.1
LLM_FREQUENCY_PENALTY = 0.8
LLM_PRESENCE_PENALTY = 0.5

# --- Embedding Configuration ---
EMBED_BATCH_SIZE = 128
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
