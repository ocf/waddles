FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install git and bash (bash is usually included in slim, but good to be safe)
RUN apt-get update && apt-get install -y git bash wget curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY sync.sh .

COPY workflows.py .
COPY config.py .
COPY events.py .
COPY database.py .
COPY llm.py .
COPY prompts.py .
COPY tools/ ./tools/

CMD ["python", "bot.py"]
