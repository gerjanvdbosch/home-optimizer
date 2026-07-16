#!/usr/bin/env bash

set -euo pipefail

echo "Starting Home Optimizer..."

exec uvicorn web.server:app \
    --app-dir src \
    --host 0.0.0.0 \
    --port 8899 \
    --workers 1
