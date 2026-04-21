# Source from the project root: `source scripts/run_env.sh`
# Loads variables from the project's .env into the current shell.
# Alternative to run_bge.sh / run_e5.sh when keys live in .env.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
set -a
source "$ROOT/.env"
set +a
