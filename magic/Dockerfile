# Use Python 3.11 as the base
FROM python:3.11-slim

# Install system dependencies: Git, Node.js, and Curl
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Codex CLI globally
RUN npm install -g @openai/codex-cli

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code (we'll clone the repo later)
COPY bot.py .

# Create the repo directory
RUN mkdir -p /app/repo

CMD ["python", "bot.py"]
