"""Application-wide runtime settings for Home Optimizer.

Single source of truth for all environment-driven configuration.  Both the
FastAPI layer (``api.py``) and the addon entry-point (``addon.py``) import
from here so no constant or default is ever defined in two places.

Environment variables
---------------------
DATABASE_URL
    SQLAlchemy database URL used by ``/api/forecast/latest`` and the
    telemetry layer.  Set automatically by the HA addon from
    ``AddonOptions.database_path``.  For local development, either:

    * export the variable before starting uvicorn::

        export DATABASE_URL=sqlite:////absolute/path/to/dev.db
        uvicorn home_optimizer.api:app --reload

    * or create a ``.env`` file in the project root and load it with
      ``python-dotenv`` (optional dev dependency, not required at runtime)::

        DATABASE_URL=sqlite:///local_dev.db

    Default (when the variable is absent): ``sqlite:///database.sqlite3``
    relative to the current working directory — suitable for quick local runs.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Named constants — imported by addon.py and api.py; never repeated inline.
# ---------------------------------------------------------------------------

#: Name of the environment variable that holds the SQLAlchemy database URL.
DATABASE_URL_ENV: str = "DATABASE_URL"

#: Default SQLAlchemy URL used when DATABASE_URL is not set.
#: Relative path → SQLite file in the process CWD (local dev).
DATABASE_URL_DEFAULT: str = "sqlite:///database.sqlite3"


def get_database_url() -> str:
    """Return the active SQLAlchemy database URL.

    Reads ``DATABASE_URL`` from the process environment.  Falls back to
    :data:`DATABASE_URL_DEFAULT` when the variable is absent.

    Returns
    -------
    str
        A valid SQLAlchemy database URL, e.g.
        ``"sqlite:////data/optimizer.db"`` or ``"sqlite:///database.sqlite3"``.
    """
    return os.environ.get(DATABASE_URL_ENV, DATABASE_URL_DEFAULT)
