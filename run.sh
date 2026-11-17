#!/usr/bin/env bash
# EuniServ Invoice MVP — macOS / Linux launcher
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt
if [ ! -f ".env" ]; then
  echo "WARNING: .env not found. Copy .env.example to .env and add GEMINI_API_KEY."
  echo "The app will still start, but LLM answers will use the labelled extractive fallback."
  cp .env.example .env
fi
mkdir -p uploads chroma_db
echo "Starting at http://127.0.0.1:8000"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
