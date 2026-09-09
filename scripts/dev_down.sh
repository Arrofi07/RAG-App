#!/usr/bin/env bash
# Tear down what ./scripts/dev_up.sh started. Leaves Docker Desktop itself
# running since quitting it isn't necessary and may interrupt other work.

QDRANT_CONTAINER="rag_qdrant_dev"

pkill -f "uvicorn main:app" && echo "API server stopped" || echo "API server was not running"

if docker ps -a --format "{{.Names}}" | grep -qx "$QDRANT_CONTAINER"; then
  docker stop "$QDRANT_CONTAINER" >/dev/null && docker rm "$QDRANT_CONTAINER" >/dev/null
  echo "Qdrant dev container removed"
else
  echo "Qdrant dev container was not present"
fi
