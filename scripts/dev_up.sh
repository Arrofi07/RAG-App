#!/usr/bin/env bash
# Bring up Qdrant + the API server for local dev/smoke-testing.
# Idempotent: safe to run again if some pieces are already up.
#
# Usage: ./scripts/dev_up.sh
# Then:  ./scripts/smoke_test.sh
# Tear down with: ./scripts/dev_down.sh

set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_CONTAINER="rag_qdrant_dev"
LOG_FILE="/tmp/rag_api_server.log"

echo "== Qdrant =="
if curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/healthz 2>/dev/null | grep -q 200; then
  echo "  already up"
else
  if ! docker info >/dev/null 2>&1; then
    echo "  starting Docker Desktop (this can take a minute)..."
    open -a Docker
    until docker info >/dev/null 2>&1; do sleep 2; done
  fi

  if docker ps -a --format "{{.Names}}" | grep -qx "$QDRANT_CONTAINER"; then
    docker start "$QDRANT_CONTAINER" >/dev/null
  else
    docker run -d --name "$QDRANT_CONTAINER" -p 6333:6333 qdrant/qdrant >/dev/null
  fi

  echo "  waiting for it to become healthy..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/healthz 2>/dev/null | grep -q 200; do sleep 1; done
  echo "  ready"
fi

echo "== API server =="
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null | grep -q 200; then
  echo "  already up"
else
  echo "  starting (logs: $LOG_FILE)..."
  uv run uvicorn main:app --port 8000 > "$LOG_FILE" 2>&1 &
  disown

  echo "  waiting for model load to finish (~30-90s)..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null | grep -q 200; do sleep 2; done
  echo "  ready"
fi

echo ""
echo "Up. Run ./scripts/smoke_test.sh now, or ./scripts/dev_down.sh when done."
