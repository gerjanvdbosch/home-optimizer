"""Repository helpers for telemetry persistence.

The repository intentionally keeps SQLAlchemy session management out of the
scheduler logic so that storage can be tested independently from APScheduler.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import sessionmaker

from .models import Base, MPCLog, TelemetryAggregate


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

    def count(self) -> int:
        """Return the number of persisted telemetry buckets."""
        return len(self.list_aggregates())

