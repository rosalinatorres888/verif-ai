#!/bin/bash

set -e

cd "$(dirname "$0")"

# Activate whichever virtualenv exists (README Quick Start creates venv/;
# .venv313 is the original dev environment)
if [ -d ".venv313" ]; then
    source .venv313/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No virtualenv found — run the README Quick Start setup first." >&2
    exit 1
fi

echo "Starting VerifAI backend on port 8000..."
python -m uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

sleep 3

echo "Starting VerifAI dashboard on port 8502..."
streamlit run frontend/streamlit_app.py --server.port 8502 &
FRONTEND_PID=$!

cleanup() {
    echo ""
    echo "Stopping VerifAI..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait
