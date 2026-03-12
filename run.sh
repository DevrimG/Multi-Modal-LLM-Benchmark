#!/bin/bash
# Convenience script to run the benchmark with a compatible Python
# Usage: ./run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python 3.10+ is required, but no Python interpreter was found."
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "Python 3.10+ is required. Found: $($PYTHON_BIN --version 2>&1)"
  exit 1
fi

exec "$PYTHON_BIN" -m llm_load_tester
