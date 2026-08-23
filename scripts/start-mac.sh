#!/usr/bin/env bash
# Start the Local RAG stack (API + Gradio UI) in the background.
# The API owns folder watching in-process, so there is no separate watcher.
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

# The Gradio UI is a separate server: $API_PORT serves the Flask API, which has
# no `/` route, so browsing it returns 404. Gradio reads these two env vars
# directly in demo.launch().
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
UI_BASE="http://${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"

EMB_ID="${EMBEDDING_ID:-tfidf:local}"
GEN_ID="${LLM_ID:-mock:any}"

# --- Preflight: external model servers ---
# An `ollama:` embedding_id or llm_id makes Ollama a hard dependency. The API's
# /health and /ready both return 200 without it -- embeddings are only contacted
# on the first ingest or query -- so check it here or the API's watcher fails
# silently on every file.
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"

model_present() {
  local m="$1" tags="$2"
  printf '%s' "$tags" | grep -q "\"name\":\"${m}\"" && return 0
  printf '%s' "$tags" | grep -q "\"name\":\"${m}:latest\"" && return 0
  return 1
}

if [[ "$EMB_ID" == ollama:* || "$GEN_ID" == ollama:* ]]; then
  if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "Ollama not responding at $OLLAMA_URL -- starting it"
    nohup ollama serve >>"$LOG_DIR/ollama.log" 2>&1 &
    # Record the pid only when we started it, so stop-mac.sh leaves an Ollama
    # the user was already running (or the desktop app) alone.
    echo $! > "$LOG_DIR/ollama.pid"
    for _ in $(seq 1 30); do
      curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break
      sleep 1
    done
  fi

  TAGS="$(curl -fsS "$OLLAMA_URL/api/tags" 2>/dev/null || true)"
  if [[ -z "$TAGS" ]]; then
    echo "ERROR: EMBEDDING_ID/LLM_ID needs Ollama but it is not reachable at $OLLAMA_URL."
    echo "       Start it with 'ollama serve' (logs: $LOG_DIR/ollama.log)."
    exit 1
  fi

  for id in "$EMB_ID" "$GEN_ID"; do
    [[ "$id" == ollama:* ]] || continue
    model="${id#ollama:}"
    if ! model_present "$model" "$TAGS"; then
      echo "ERROR: Ollama model '$model' is not pulled. Run: ollama pull $model"
      exit 1
    fi
  done
  echo "Ollama ready at $OLLAMA_URL (required models present)"
fi

# LM Studio is optional: only the lmstudio: prefix needs it, and the UI can pick
# a different model at request time, so warn rather than fail.
if [[ "$GEN_ID" == lmstudio:* ]]; then
  LMS_URL="${LMSTUDIO_BASE_URL:-http://localhost:1234/v1}"
  if ! curl -fsS "$LMS_URL/models" >/dev/null 2>&1; then
    echo "Warning: LLM_ID is '$GEN_ID' but LM Studio is not responding at $LMS_URL."
    echo "         Open LM Studio -> Developer -> Start Server."
  else
    echo "LM Studio ready at $LMS_URL"
  fi
fi

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
  # -u because the API now owns folder watching: its "file X added to the
  # database" lines must reach api.log live, not sit in a block buffer.
  nohup uv run python -u backend/local_rag_api.py >>"$LOG_DIR/api.log" 2>&1 &
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

# --- Watcher ---
# The API process now owns folder watching (see backend/watch_manager.py) and
# repoints itself when the UI switches folders. No separate watcher process.

# --- Gradio UI ---
# Started after the API so its startup probe of GET /config finds a live backend
# and the embedding_id / version boxes come up aligned.
UI_PID_FILE="$LOG_DIR/ui.pid"
if [[ -f "$UI_PID_FILE" ]] && is_pid_alive "$(cat "$UI_PID_FILE")"; then
  echo "UI already running (pid $(cat "$UI_PID_FILE")). Skipping."
else
  echo "Starting UI -> $UI_BASE"
  nohup uv run python -u frontend/gradio_app.py >>"$LOG_DIR/gradio.log" 2>&1 &
  echo $! > "$UI_PID_FILE"
fi

echo -n "Waiting for UI"
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null "$UI_BASE/" 2>/dev/null; then
    echo " ... ok"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -fsS -o /dev/null "$UI_BASE/" 2>/dev/null; then
  echo
  echo "Warning: UI did not come up on $UI_BASE. Last 30 log lines:"
  tail -n 30 "$LOG_DIR/gradio.log" || true
fi

echo
echo "Stack is up:"
echo "  UI:      $UI_BASE   (logs: $LOG_DIR/gradio.log)   <- open this in your browser"
echo "  API:     $API_BASE   (logs: $LOG_DIR/api.log)"
echo
echo "Try:"
echo "  curl -s -X POST $API_BASE/answer -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\":\"hello\",\"llm_id\":\"mock:any\"}'"
echo "  uv run python frontend/launch.py --query 'hello' --llm-id mock:any"
echo
echo "Stop with: bash scripts/stop-mac.sh"
