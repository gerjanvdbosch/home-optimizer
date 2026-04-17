"""Database configuration and lifecycle management for Home Optimizer.

Single source of truth for every database-related concern: URL resolution,
directory creation, schema initialisation, and repository construction.
Both the FastAPI layer (``api.py``) and the runner entry-points
(``addon.py``, ``local_runner.py``) import from here.

Environment variables
---------------------
DATABASE_URL
    SQLAlchemy connection URL.  Set automatically by the HA addon from
    ``AddonOptions.database_path``.  For local development::

        export DATABASE_URL=sqlite:////absolute/path/to/dev.db
        uvicorn home_optimizer.api:app --reload

    Default (when absent): ``sqlite:///database.sqlite3`` — SQLite file
    relative to the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .telemetry import TelemetryRepository

# ---------------------------------------------------------------------------
# Named constants — kept for the handful of callers that still read the
# env-var key or the default string directly (e.g. local_runner argparse help).
# ---------------------------------------------------------------------------

#: Name of the environment variable that holds the SQLAlchemy database URL.
DATABASE_URL_ENV: str = "DATABASE_URL"

#: Default SQLAlchemy URL used when DATABASE_URL is not set.
DATABASE_URL_DEFAULT: str = "sqlite:///database.sqlite3"


class Database:
    """Owns the database URL and all lifecycle operations for one database.

    Use :meth:`from_env` to read the URL from the environment, or
    :meth:`from_path` to derive it from a filesystem path.  Both factory
    methods validate the URL immediately so misconfiguration fails fast
    at startup rather than at the first query.

    Args:
        url: SQLAlchemy connection URL [str].  Must be non-empty.

    Raises:
        ValueError: If *url* is empty or blank.

    Example — environment-driven (production / addon)::

        db = Database.from_env()
        repo = db.repository()          # schema created automatically

    Example — explicit path (local runner)::

        db = Database.from_path(Path("./dev_data/local.db"))
        repo = db.repository()

    Example — inject into FastAPI env so /api/forecast/latest finds it::

        db = Database.from_env()
        db.export_to_env()              # writes DATABASE_URL to os.environ
    """

    def __init__(self, url: str) -> None:
        if not url or not url.strip():
            raise ValueError(
                "Database URL must not be blank.  "
                f"Set the {DATABASE_URL_ENV} environment variable or call "
                "Database.from_path()."
            )
        self._url: str = url

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """SQLAlchemy connection URL [str] (read-only)."""
        return self._url

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def url_from_env(cls) -> str:
        """Return the active SQLAlchemy URL from ``DATABASE_URL``.

        Falls back to :data:`DATABASE_URL_DEFAULT` when the variable is absent.

        Returns:
            SQLAlchemy connection URL [str].
        """
        return os.environ.get(DATABASE_URL_ENV, DATABASE_URL_DEFAULT)

    @classmethod
    def from_env(cls) -> "Database":
        """Create a :class:`Database` from the ``DATABASE_URL`` environment variable.

        Falls back to :data:`DATABASE_URL_DEFAULT` when the variable is absent.

        Returns:
            :class:`Database` with the URL from the environment.
        """
        return cls(cls.url_from_env())

    @classmethod
    def from_path(cls, path: str | Path) -> "Database":
        """Create a :class:`Database` from a filesystem path.

        Converts *path* to an absolute ``sqlite:///`` URL and ensures the
        parent directory exists.  This replaces the ad-hoc
        ``Path(...).parent.mkdir(...)`` + ``f"sqlite:///{db_path}"`` pattern
        that was previously duplicated in ``addon.py`` and ``local_runner.py``.

        Args:
            path: Filesystem path to the SQLite file (relative or absolute).

        Returns:
            :class:`Database` with the resolved SQLite URL.

        Raises:
            ValueError: If *path* points to an existing directory.
        """
        resolved = Path(path).resolve()
        if resolved.is_dir():
            raise ValueError(f"Database path {resolved!r} is a directory, not a file.")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{resolved}")

    # ------------------------------------------------------------------
    # Repository factory
    # ------------------------------------------------------------------

    def repository(self, *, init_schema: bool = True) -> "TelemetryRepository":
        """Return a fully initialised :class:`~home_optimizer.telemetry.TelemetryRepository`.

        This is the **single canonical entry point** for constructing a
        repository anywhere in the application.  It replaces the three
        separate ``TelemetryRepository(database_url=...)`` + ``create_schema()``
        call-pairs that existed across ``addon.py``, ``local_runner.py``, and
        ``api.py``.

        Args:
            init_schema: When ``True`` (default) calls
                :meth:`~home_optimizer.telemetry.TelemetryRepository.create_schema`
                immediately.  Pass ``False`` when the schema was already
                initialised (e.g. shared test fixture).

        Returns:
            :class:`~home_optimizer.telemetry.TelemetryRepository` ready for use.
        """
        # Deferred import to avoid a circular dependency:
        # database → telemetry → (nothing back in database).
        from .telemetry import TelemetryRepository  # noqa: PLC0415

        repo = TelemetryRepository(database_url=self._url)
        if init_schema:
            repo.create_schema()
        return repo

    # ------------------------------------------------------------------
    # Environment export
    # ------------------------------------------------------------------

    def export_to_env(self) -> None:
        """Write the URL to ``DATABASE_URL`` in the process environment.

        Required by the FastAPI ``/api/forecast/latest`` endpoint, which
        resolves the database via :meth:`Database.from_env`
        at request time rather than through a DI-injected object.  Call this
        once at startup so every component sees the same URL.

        Side effects:
            Sets ``os.environ["DATABASE_URL"]`` to ``self.url``.
        """
        os.environ[DATABASE_URL_ENV] = self._url

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"Database(url={self._url!r})"
