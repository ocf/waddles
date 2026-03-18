# GEMINI.md - Waddles Project Context

## Project Overview
**Waddles** is a sophisticated, RAG-powered Discord bot designed for the Open Computing Facility (OCF) at UC Berkeley. It serves as a community assistant, providing quick access to OCF documentation, server status, and general help through an agentic AI interface.

### Key Technologies
- **Language:** Python (>= 3.11)
- **Discord API:** `discord.py`
- **AI Framework:** `LlamaIndex` (utilizing the Workflows system for agentic reasoning)
- **Vector Database:** `ChromaDB` (for document indexing and persistent user memory)
- **LLM Backend:** Supports local hosting via `Ollama` and `SGLang`
- **Package Management:** `uv`
- **Deployment:** Docker & Docker Compose

## Architecture
The bot employs a **Research -> Reason -> Respond** workflow:
1.  **Retrieval-Augmented Generation (RAG):** Waddles prioritizes internal OCF documentation (synced via `sync.sh`) to provide accurate, policy-compliant answers.
2.  **Agentic Workflows:** Uses `OCFAgentWorkflow` (in `workflow.py`) to manage a cyclic tool-calling loop. It can decide to search the web, scrape URLs, or run Python code to fulfill requests.
3.  **Thinking Mode:** Supports a "thinking" persona that performs step-by-step reasoning before acting.
4.  **Persistent Memory:** Utilizes `FactExtractionMemoryBlock` and `VectorMemoryBlock` to remember user-specific facts and conversation history across sessions.
5.  **Multi-Modal:** Capable of processing image attachments using vision-capable LLMs.

## Key Files & Directories
- `bot.py`: The main Discord bot implementation, handling commands and events.
- `workflow.py`: Core logic for the agentic reasoning loop and tool integration.
- `database.py`: Manages document embedding, vector store indexing, and user memory.
- `tools/`: A suite of function tools (web search, documentation search, Python execution, etc.).
- `prompts.py`: Management of bot "personas" and system prompt templates.
- `config.py`: Centralized configuration for LLM endpoints, storage paths, and bot settings.
- `sync.sh`: Shell script to synchronize external OCF documentation for indexing.

## Building and Running

### Prerequisites
- **Python 3.11+**
- **uv** (recommended for dependency management)
- **Local LLM Services:** `Ollama` and `SGLang` should be running as configured in `config.py`.
- **Environment Variables:** `DISCORD_TOKEN` must be set.

### Development Commands
- **Install Dependencies:**
  ```bash
  uv sync
  ```
- **Run the Bot:**
  ```bash
  python bot.py
  ```
- **Update Documentation Index:**
  The bot automatically updates its index hourly. You can manually trigger updates via Discord commands:
  - `?reload`: Smart-update (only changed files).
  - `?reloadfull`: Full sync and update.

## Development Conventions
- **Tool Creation:** New tools should be added to the `tools/` directory and registered in `tools/tools.py`.
- **Persona Management:** Custom personas can be added via the `?persona set` command or by modifying `prompts.py`.
- **Async First:** All bot interactions and AI workflows are asynchronous to prevent blocking the Discord gateway.
- **RAG-First Policy:** Always prioritize `search_docs` for OCF-related queries to ensure accuracy.
