from .dashboard import create_dashboard_router
from .history_import import create_history_import_router
from .identification import create_identification_router
from .prediction import create_prediction_router

__all__ = [
    "create_dashboard_router",
    "create_history_import_router",
    "create_identification_router",
    "create_prediction_router",
]
