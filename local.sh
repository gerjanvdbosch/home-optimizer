#!/usr/bin/env bash

set -euo pipefail

echo "Starting Home Optimizer..."

exec python -m web.app
