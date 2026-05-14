#!/usr/bin/env bash
# Start the Local RAG stack (API + folder watcher) in the background.
# Logs go to logs/, pid files to logs/*.pid.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# Load .env if present. Done in a subshell-safe block so a malformed line
# (e.g. stray spaces around `=`) never aborts the launcher. If sourcing
# fails we fall back to the API's built-in defaults
# (EMBEDDING_ID=tfidf:local, LLM_ID=mock:any) which need no external services.
if [[ -f "$ROOT/.env" ]]; then
  set +e
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env" 2>/dev/null
  env_rc=$?
  set +a
  set -e
  if [[ $env_rc -ne 0 ]]; then
    echo "Warning: .env could not be fully sourced ($env_rc). Continuing with defaults."
  fi
fi

API_HOST="${RAG_HOST:-127.0.0.1}"
API_PORT="${RAG_PORT:-5050}"
API_BASE="http://${API_HOST}:${API_PORT}"

is_pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# --- API ---
API_PID_FILE="$LOG_DIR/api.pid"
if [[ -f "$API_PID_FILE" ]] && is_pid_alive "$(cat "$API_PID_FILE")"; then
  echo "API already running (pid $(cat "$API_PID_FILE")). Skipping."
else
  echo "Starting API -> $API_BASE"
  nohup uv run python backend/local_rag_api.py >>"$LOG_DIR/api.log" 2>&1 &
  echo $! > "$API_PID_FILE"
fi

# Wait for /health
echo -n "Waiting for API health"
for _ in $(seq 1 60); do
  if curl -fsS "$API_BASE/health" >/dev/null 2>&1; then
    echo " ... ok"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -fsS "$API_BASE/health" >/dev/null 2>&1; then
  echo
  echo "ERROR: API did not become healthy. Last 30 log lines:"
  tail -n 30 "$LOG_DIR/api.log" || true
  exit 1
fi

# --- Watcher (only if watched_folder exists) ---
WATCH_DIR="$ROOT/watched_folder"
WATCH_PID_FILE="$LOG_DIR/watcher.pid"
if [[ -d "$WATCH_DIR" ]]; then
  if [[ -f "$WATCH_PID_FILE" ]] && is_pid_alive "$(cat "$WATCH_PID_FILE")"; then
    echo "Watcher already running (pid $(cat "$WATCH_PID_FILE")). Skipping."
  else
    EMB_ID="${EMBEDDING_ID:-tfidf:local}"
    VER="${CHROMA_COLLECTION_VERSION:-v1}"
    echo "Starting watcher on $WATCH_DIR (embedding=$EMB_ID, version=$VER)"
    nohup uv run python backend/folder_watcher.py \
      --watch "$WATCH_DIR" \
      --api "$API_BASE" \
      --embedding-id "$EMB_ID" \
      --version "$VER" \
      --reconcile-interval 300 \
      >>"$LOG_DIR/watcher.log" 2>&1 &
    echo $! > "$WATCH_PID_FILE"
  fi
else
  echo "No watched_folder/ at $WATCH_DIR -- skipping watcher."
fi

echo
echo "Stack is up:"
echo "  API:     $API_BASE   (logs: $LOG_DIR/api.log)"
[[ -f "$WATCH_PID_FILE" ]] && echo "  Watcher: pid $(cat "$WATCH_PID_FILE")  (logs: $LOG_DIR/watcher.log)"
echo
echo "Try:"
echo "  curl -s -X POST $API_BASE/answer -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\":\"hello\",\"llm_id\":\"mock:any\"}'"
echo "  uv run python frontend/launch.py --query 'hello' --llm-id mock:any"
echo "  uv run python frontend/gradio_app.py"
echo
echo "Stop with: bash scripts/stop-mac.sh"
