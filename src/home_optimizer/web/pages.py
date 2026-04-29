from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template

from home_optimizer.app import AppSettings, build_history_import_request
from home_optimizer.web.schemas import DashboardViewModel

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def build_dashboard_view_model(
    settings: AppSettings,
    *,
    title: str,
) -> DashboardViewModel:
    request = build_history_import_request(settings)
    return DashboardViewModel(
        title=title,
        import_window_days=settings.history_import_max_days_back,
        chunk_days=settings.history_import_chunk_days,
        sensor_count=len(request.specs),
        database_path=settings.database_path,
        api_port=settings.api_port,
    )


def render_template(template_name: str, view_model: DashboardViewModel) -> str:
    template = Template((TEMPLATES_DIR / template_name).read_text(encoding="utf-8"))
    return template.substitute(
        title=escape(view_model.title),
        import_window_days=view_model.import_window_days,
        chunk_days=view_model.chunk_days,
        sensor_count=view_model.sensor_count,
        database_path=escape(view_model.database_path),
        api_port=view_model.api_port,
    )
