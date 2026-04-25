from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryImportRunResponse(BaseModel):
    status: str = Field(default="ok")
    imported_rows: dict[str, int]
    total_rows: int
    sensor_count: int


class DashboardViewModel(BaseModel):
    title: str
    import_enabled: bool
    import_window_days: int
    chunk_days: int
    sensor_count: int
    database_path: str
    api_port: int
