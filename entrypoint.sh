#!/bin/bash
set -e

echo "Starting Ollama service..."
ollama serve & 
OLLAMA_PID=$!

# Wait until the Ollama API is available on port 11434
echo "Waiting for Ollama API to be available..."
while ! curl -s http://localhost:11434/api/embeddings > /dev/null; do
  sleep 2
done

echo "Ollama API is up. Pulling model 'nomic-embed-text'..."
ollama pull nomic-embed-text

# Bring the Ollama service process to the foreground so the container keeps running
wait $OLLAMA_PID
