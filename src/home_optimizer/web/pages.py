from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template

from home_optimizer.web.schemas import DashboardViewModel

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def render_dashboard(view_model: DashboardViewModel) -> str:
    button_disabled = "disabled" if not view_model.import_enabled else ""
    button_label = (
        "Import uitgeschakeld in configuratie"
        if not view_model.import_enabled
        else "Importeer geschiedenis"
    )

    template = Template(
        (TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")
    )
    return template.substitute(
        title=escape(view_model.title),
        import_enabled=str(view_model.import_enabled).lower(),
        import_window_days=view_model.import_window_days,
        chunk_days=view_model.chunk_days,
        sensor_count=view_model.sensor_count,
        database_path=escape(view_model.database_path),
        api_port=view_model.api_port,
        button_disabled=button_disabled,
        button_label=escape(button_label),
    )
