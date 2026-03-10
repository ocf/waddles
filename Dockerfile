FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install git and other utilities
RUN apt-get update && apt-get install -y git bash wget curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

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
