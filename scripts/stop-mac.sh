#!/usr/bin/env bash
# Stop the Local RAG stack started by scripts/start-mac.sh.
# Sends SIGTERM, waits up to 10s, then SIGKILL.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LOG_DIR="$ROOT/logs"

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  [[ -f "$pid_file" ]] || { echo "$label: no pid file ($pid_file)"; return 0; }

  local pid
  pid="$(cat "$pid_file" || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "$label: not running (stale pid file)"
    rm -f "$pid_file"
    return 0
  fi

  echo "$label: stopping pid $pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "$label: force killing pid $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

stop_pid_file "ui"      "$LOG_DIR/ui.pid"
stop_pid_file "api"     "$LOG_DIR/api.pid"

# Legacy: the watcher used to be its own process. Stop and clear any pid file
# left over from a stack started before watching moved into the API.
[[ -f "$LOG_DIR/watcher.pid" ]] && stop_pid_file "watcher" "$LOG_DIR/watcher.pid"

# Only present when start-mac.sh had to start Ollama itself. An Ollama the user
# was already running (or the desktop app) has no pid file and is left alone.
if [[ -f "$LOG_DIR/ollama.pid" ]]; then
  stop_pid_file "ollama" "$LOG_DIR/ollama.pid"
else
  echo "ollama: not started by us -- leaving it running"
fi

# Fallback: catch any leftover instances started outside start-mac.sh.
for pat in "backend/local_rag_api.py" "backend/folder_watcher.py" "frontend/gradio_app.py"; do
  pids="$(pgrep -f "$pat" || true)"
  if [[ -n "$pids" ]]; then
    echo "Killing leftover $pat: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
  fi
done

echo "Done."
