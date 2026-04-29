#!/usr/bin/env bash

set -euo pipefail

exec python -m home_optimizer.entrypoints.local --set database_path=database.sqlite "$@"
