from __future__ import annotations

from fastapi import Request

from home_optimizer.app.history_import_jobs import HistoryImportJobRunner
from home_optimizer.web.ports import WebAppContainer


def get_container(request: Request) -> WebAppContainer:
    return request.app.state.container


def get_history_import_jobs(request: Request) -> HistoryImportJobRunner:
    return request.app.state.history_import_jobs
