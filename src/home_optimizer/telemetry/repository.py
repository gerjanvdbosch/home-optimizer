"""Repository helpers for telemetry persistence.

The repository intentionally keeps SQLAlchemy session management out of the
scheduler logic so that storage can be tested independently from APScheduler.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from .models import Base, ForecastSnapshot, MPCLog, TelemetryAggregate


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

    def count(self) -> int:
        """Return the number of persisted telemetry buckets."""
        return len(self.list_aggregates())
