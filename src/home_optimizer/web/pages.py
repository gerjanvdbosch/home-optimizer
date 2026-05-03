from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template

from home_optimizer.app import AppSettings
from home_optimizer.web.schemas import DashboardPageViewModel

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def build_dashboard_view_model(
    settings: AppSettings,
    *,
    title: str,
) -> DashboardPageViewModel:
    return DashboardPageViewModel(
        title=title,
        database_path=settings.database_path,
    )


def render_template(template_name: str, view_model: DashboardPageViewModel) -> str:
    template = Template((TEMPLATES_DIR / template_name).read_text(encoding="utf-8"))
    return template.substitute(
        title=escape(view_model.title),
        database_path=escape(view_model.database_path),
    )
