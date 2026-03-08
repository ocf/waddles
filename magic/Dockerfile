# Use Python 3.11 as the base
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies: Git, Node.js, and Curl
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCode CLI globally
RUN npm install -g opencode-ai

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code and proxy code
COPY bot.py .
COPY proxy.py .

# Create the repo directory
RUN mkdir -p /app/repo

# Create a startup script to run both the proxy and the bot
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Proxy Server on port 4000..."\n\
python proxy.py &\n\
echo "⏳ Waiting for proxy to boot..."\n\
sleep 3\n\
echo "🤖 Starting Discord Bot..."\n\
python bot.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
