#!/bin/bash
set -e

export LOG_LEVEL="DEBUG"

python3 -u ./src/coordinator.py
