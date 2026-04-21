#!/usr/bin/with-contenv bashio
# ==============================================================================
# Home Optimizer — addon entry point
#
# The HA supervisor injects:
#   /data/options.json   – addon options from the Configuration tab
#   SUPERVISOR_TOKEN     – HA REST API bearer token (via contenv)
# ==============================================================================

set -euo pipefail

bashio::log.info "Starting Home Optimizer addon..."
bashio::log.info "Options: $(bashio::config)"

exec python -m home_optimizer.addon

