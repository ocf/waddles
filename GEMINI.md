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

## Architecture & Features

### 1. Agentic Workflows
The bot uses `OCFAgentWorkflow` (in `workflow.py`) to manage a cyclic tool-calling loop (Research -> Reason -> Respond). It can search the web, scrape URLs, run Python code, or search OCF documentation to fulfill requests.

### 2. Continuous Conversation Threads
Waddles maintains context via Discord threads rather than just a persistent database.
- **Auto-Threading:** After an initial command (e.g., `?ask`), Waddles automatically creates a Discord thread on its response message.
- **Dynamic Thread Naming:** The bot uses the LLM to generate a concise, 3-5 word title based on the user's initial question.
- **History Re-reading:** When a user replies in a Waddles-owned thread, the bot fetches up to 20 previous messages to reconstruct the chat history.
- **Message Merging:** To keep the LLM's context clean, back-to-back chunks from Waddles (split by the 2000-char limit) are merged into single assistant messages during history reconstruction.

### 3. Multi-User Support
To handle threads where multiple users are active:
- **Transcript Format:** All user messages are prefixed as `[Display Name (username) @ Timestamp]: Content`. This includes both their server-specific name and their global account handle.
- **System Instruction:** The system prompt explicitly informs the LLM that it is in a multi-user environment and explains this transcript format.
- **Consistency:** This metadata is applied both to the current query and all historical messages in the thread.

### 4. Multi-Message Streaming
Waddles handles long responses (exceeding Discord's 2000-character limit) by:
- **Chunking:** Splitting the text into 2000-character segments.
- **Streaming integration:** Subsequent chunks are sent immediately to the thread as they are generated, ensuring a smooth reading experience.

### 5. RAG & Multi-Modal
- **Documentation Sync:** Internal OCF docs are synced via `sync.sh` and indexed hourly.
- **Vision:** Supports image attachments by converting them to base64 data URLs for vision-capable LLMs.

## Key Files & Directories
- `bot.py`: Main Discord bot implementation, handling commands, threading, and transcript reconstruction.
- `workflow.py`: Core logic for the agentic reasoning loop and tool integration.
- `database.py`: Manages document embedding and vector store indexing.
- `tools/`: A suite of function tools (web search, docs search, Python execution, etc.).
- `prompts.py`: Persona management and system prompt templates.
- `config.py`: Centralized configuration for LLM endpoints and bot settings.

## Building and Running

### Prerequisites
- **Python 3.11+**
- **uv** (recommended for dependency management)
- **Local LLM Services:** `Ollama` and `SGLang` should be running as configured in `config.py`.
- **Environment Variables:** `DISCORD_TOKEN` must be set.

### Development Commands
- **Install Dependencies:** `uv sync`
- **Run the Bot:** `python bot.py`
- **Manual Index Update:** `?reload` (smart) or `?reloadfull` (full sync).

## Development Conventions
- **Tool Creation:** New tools must be added to `tools/` and registered in `tools/tools.py`.
- **Async First:** All bot interactions and workflows are asynchronous.
- **RAG-First Policy:** Always prioritize `search_docs` for OCF-related queries.
- **Transcript Consistency:** When modifying message handling, ensure the `[Username @ Timestamp]` format is preserved for user attribution.
