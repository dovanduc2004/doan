#!/bin/bash
set -e

cd "$(dirname "$0")"

PORT="${PORT:-8080}"

if [ ! -d ".venv" ]; then
  echo ">> creating venv (.venv)"
  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi

source .venv/bin/activate

echo ">> installing deps"
pip install --upgrade pip
pip install -r requirements.txt

echo ">> starting qdrant container"
docker compose up -d qdrant

echo ">> launching FastAPI on http://127.0.0.1:${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --reload
