#!/bin/bash
set -e

export MODEL_DIR="/config/models"
export DB_DIR="/config/db"

mkdir -p "${MODEL_DIR}"
mkdir -p "${DB_DIR}"

export LOG_LEVEL="DEBUG"

python3 -u ./src/coordinator.py
