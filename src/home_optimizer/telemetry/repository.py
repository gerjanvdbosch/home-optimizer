"""Repository helpers for telemetry persistence.

The repository intentionally keeps SQLAlchemy session management out of the
scheduler logic so that storage can be tested independently from APScheduler.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from ..types import CalibrationSnapshotPayload
from .models import Base, CalibrationSnapshot, ForecastSnapshot, MPCLog, TelemetryAggregate


class TelemetryRepository:
    """Persist aggregated telemetry buckets and MPC log entries into a SQLAlchemy database.

    Parameters
    ----------
    database_url:
        SQLAlchemy database URL.  Used when ``engine`` is not supplied.
    engine:
        Optional pre-created engine, useful for tests.
    """

    def __init__(self, database_url: str | None = None, engine: Engine | None = None) -> None:
        if engine is None and database_url is None:
            raise ValueError("Either database_url or engine must be supplied.")
        self.engine = engine or create_engine(database_url, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_schema(self) -> None:
        """Create the telemetry tables if they do not already exist."""
        Base.metadata.create_all(self.engine)

    def add_aggregate(self, aggregate_data: dict[str, Any]) -> TelemetryAggregate:
        """Insert one aggregated telemetry bucket.

        Parameters
        ----------
        aggregate_data:
            Mapping with the exact :class:`TelemetryAggregate` constructor fields.
        """
        record = TelemetryAggregate(**aggregate_data)
        with self._session_factory() as session:
            session.add(record)
            session.flush()
            session.commit()
            session.refresh(record)
        return record

    def add_mpc_log(self, mpc_data: dict[str, Any]) -> MPCLog:
        """Insert one MPC solve record.

        Parameters
        ----------
        mpc_data:
            Mapping with the exact :class:`MPCLog` constructor fields.
            Required keys: ``solve_time_utc``, ``p_ufh_setpoint_kw``,
            ``p_dhw_setpoint_kw``, ``solver_status``, ``t_out_forecast_c``,
            ``gti_forecast_w_per_m2``, ``electricity_price_eur_per_kwh``,
            ``cop_ufh``, ``cop_dhw``, ``horizon_steps``,
            ``t_room_initial_c``, ``t_dhw_top_initial_c``.
        """
        record = MPCLog(**mpc_data)
        with self._session_factory() as session:
            session.add(record)
            session.flush()
            session.commit()
            session.refresh(record)
        return record

    def add_calibration_snapshot(self, payload: CalibrationSnapshotPayload) -> CalibrationSnapshot:
        """Persist one automatic-calibration payload snapshot.

        Args:
            payload: Structured calibration snapshot with effective MPC overrides
                and per-stage diagnostics.

        Returns:
            Persisted ORM row from ``calibration_snapshots``.
        """
        record = CalibrationSnapshot(
            generated_at_utc=payload.generated_at_utc,
            payload_json=payload.model_dump_json(),
        )
        with self._session_factory() as session:
            session.add(record)
            session.flush()
            session.commit()
            session.refresh(record)
        return record

    def list_aggregates(self) -> list[TelemetryAggregate]:
        """Return all telemetry buckets ordered by their start timestamp."""
        with self._session_factory() as session:
            stmt = select(TelemetryAggregate).order_by(TelemetryAggregate.bucket_start_utc.asc())
            return list(session.scalars(stmt).all())

    def list_mpc_logs(self) -> list[MPCLog]:
        """Return all MPC solve records ordered by solve time."""
        with self._session_factory() as session:
            stmt = select(MPCLog).order_by(MPCLog.solve_time_utc.asc())
            return list(session.scalars(stmt).all())

    def get_latest_calibration_snapshot(self) -> CalibrationSnapshotPayload | None:
        """Return the newest persisted automatic-calibration payload, if any."""
        with self._session_factory() as session:
            stmt = select(CalibrationSnapshot).order_by(CalibrationSnapshot.generated_at_utc.desc()).limit(1)
            row = session.scalars(stmt).first()
            if row is None:
                return None
            return CalibrationSnapshotPayload.model_validate_json(row.payload_json)

    def bulk_add_forecast_snapshots(self, rows: list[dict[str, Any]]) -> int:
        """Insert a batch of forecast steps, silently skipping duplicates.

        Each row must contain the exact :class:`ForecastSnapshot` constructor
        fields: ``fetched_at_utc``, ``valid_at_utc``, ``step_k``, ``dt_hours``,
        ``t_out_c``, ``gti_w_per_m2``, ``gti_pv_w_per_m2``.

        Uniqueness is enforced by the ``uq_forecast_step`` constraint
        (``fetched_at_utc``, ``step_k``).  Duplicate rows (e.g. from a restart
        within the same UTC hour) are skipped via ``INSERT OR IGNORE`` emulation:
        each row is attempted individually; :class:`~sqlalchemy.exc.IntegrityError`
        is caught and the row is discarded without rolling back the whole batch.

        Parameters
        ----------
        rows:
            List of dicts, one per forecast step.

        Returns
        -------
        int
            Number of rows successfully inserted (0 if all were duplicates).
        """
        inserted = 0
        for row in rows:
            record = ForecastSnapshot(**row)
            try:
                with self._session_factory() as session:
                    session.add(record)
                    session.flush()
                    session.commit()
                    inserted += 1
            except IntegrityError:
                # Duplicate (fetched_at_utc, step_k) — silently skip.
                pass
        return inserted

    def list_forecast_snapshots(
        self,
        *,
        fetched_at_utc=None,
    ) -> list[ForecastSnapshot]:
        """Return forecast snapshot rows, optionally filtered by fetch time.

        Parameters
        ----------
        fetched_at_utc:
            If provided, return only steps from this specific fetch
            (i.e. one complete forecast batch).  If ``None``, return all rows.

        Returns
        -------
        list[ForecastSnapshot]
            Rows ordered by ``(fetched_at_utc, step_k)``.
        """
        with self._session_factory() as session:
            stmt = select(ForecastSnapshot).order_by(
                ForecastSnapshot.fetched_at_utc.asc(),
                ForecastSnapshot.step_k.asc(),
            )
            if fetched_at_utc is not None:
                stmt = stmt.where(ForecastSnapshot.fetched_at_utc == fetched_at_utc)
            return list(session.scalars(stmt).all())

    def get_latest_forecast_fetched_at(self):
        """Return the most recent ``fetched_at_utc`` value in ``forecast_snapshots``.

        Returns
        -------
        datetime | None
            The UTC timestamp of the most recent forecast batch, or ``None``
            when the table is empty.
        """
        with self._session_factory() as session:
            stmt = (
                select(ForecastSnapshot.fetched_at_utc)
                .order_by(ForecastSnapshot.fetched_at_utc.desc())
                .limit(1)
            )
            result = session.scalars(stmt).first()
            return result

    def get_latest_forecast_batch(self) -> list[ForecastSnapshot]:
        """Return all steps of the most recently persisted forecast.

        Fetches the latest ``fetched_at_utc`` and returns every step from that
        batch, ordered by ``step_k``.

        Returns
        -------
        list[ForecastSnapshot]
            Steps ordered by ``step_k``.  Empty list when the table is empty.
        """
        fetched_at = self.get_latest_forecast_fetched_at()
        if fetched_at is None:
            return []
        return self.list_forecast_snapshots(fetched_at_utc=fetched_at)

    def get_aggregate_time_bounds(self) -> tuple[datetime | None, datetime | None]:
        """Return the oldest bucket start and newest bucket end in telemetry history."""
        with self._session_factory() as session:
            stmt = select(
                func.min(TelemetryAggregate.bucket_start_utc),
                func.max(TelemetryAggregate.bucket_end_utc),
            )
            min_start_utc, max_end_utc = session.execute(stmt).one()
            return min_start_utc, max_end_utc

    def get_forecast_time_bounds(self) -> tuple[datetime | None, datetime | None]:
        """Return the oldest and newest forecast validity timestamps in history."""
        with self._session_factory() as session:
            stmt = select(
                func.min(ForecastSnapshot.valid_at_utc),
                func.max(ForecastSnapshot.valid_at_utc),
            )
            min_valid_utc, max_valid_utc = session.execute(stmt).one()
            return min_valid_utc, max_valid_utc

    def count(self) -> int:
        """Return the number of persisted telemetry buckets."""
        return len(self.list_aggregates())

    def count_forecast_snapshots(self) -> int:
        """Return the number of persisted forecast rows."""
        with self._session_factory() as session:
            stmt = select(func.count(ForecastSnapshot.id))
            result = session.execute(stmt).scalar_one()
            return int(result)

    def forecast_training_fingerprint(
        self,
    ) -> tuple[str, int, datetime | None, datetime | None, int, datetime | None, datetime | None]:
        """Return a compact fingerprint of persisted ML training history.

        The forecasting cache uses this fingerprint to detect when new telemetry
        buckets or forecast rows have been appended and the model must be retrained.

        Returns:
            Tuple containing:
                database_url [-],
                aggregate_count [-],
                aggregate_min_start_utc [UTC],
                aggregate_max_end_utc [UTC],
                forecast_row_count [-],
                forecast_min_valid_utc [UTC],
                forecast_latest_fetch_utc [UTC].
        """

        aggregate_min_start_utc, aggregate_max_end_utc = self.get_aggregate_time_bounds()
        forecast_min_valid_utc, _forecast_max_valid_utc = self.get_forecast_time_bounds()
        return (
            self.engine.url.render_as_string(hide_password=True),
            self.count(),
            aggregate_min_start_utc,
            aggregate_max_end_utc,
            self.count_forecast_snapshots(),
            forecast_min_valid_utc,
            self.get_latest_forecast_fetched_at(),
        )

    def database_file_path(self) -> Path:
        """Return the resolved SQLite database file path used by this repository.

        Raises:
            ValueError: If the repository is not backed by a SQLite file URL.
        """

        parsed_url = make_url(self.engine.url.render_as_string(hide_password=False))
        if parsed_url.drivername != "sqlite":
            raise ValueError(
                "Disk-backed forecast artifacts currently require a SQLite repository; "
                f"got driver {parsed_url.drivername!r}."
            )
        if parsed_url.database in (None, "", ":memory:"):
            raise ValueError("Forecast artifacts require a SQLite file path, not an in-memory database.")
        return Path(parsed_url.database).resolve()

    def forecast_artifact_path(self, artifact_name: str, *, suffix: str = ".joblib") -> Path:
        """Return the canonical on-disk path for one persisted forecast model artifact.

        Args:
            artifact_name: Stable artifact base name, for example ``"shutter_model"``.
            suffix: Artifact file suffix, default ``.joblib``.

        Returns:
            Absolute artifact file path stored next to the SQLite database.
        """

        if not artifact_name or not artifact_name.strip():
            raise ValueError("artifact_name must not be blank.")
        database_path = self.database_file_path()
        safe_artifact_name = artifact_name.strip().replace(" ", "_")
        return database_path.with_name(f"{database_path.stem}.{safe_artifact_name}{suffix}")

