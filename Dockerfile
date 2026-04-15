FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install git, basic utilities, and gnupg (required for the Node.js setup script)
RUN apt-get update && apt-get install -y git bash wget curl vim gnupg && rm -rf /var/lib/apt/lists/*

# Install Node.js (v20 LTS) directly from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the Pyodide package right in the /app directory.
# This ensures /app/node_modules/pyodide exists exactly where the Python script expects it.
RUN npm install pyodide

# Copy Python dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Sync Python dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Add the virtual environment to the PATH so 'python' defaults to it
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application files
COPY bot.py .
COPY sync.sh .
COPY workflow.py .
COPY config.py .
COPY events.py .
COPY database.py .
COPY llm.py .
COPY prompts.py .
COPY tools/ ./tools/

CMD ["python", "bot.py"]
