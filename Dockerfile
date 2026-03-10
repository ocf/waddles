FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install git and other utilities
RUN apt-get update && apt-get install -y git bash wget curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Sync dependencies
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
