from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert

from home_optimizer.domain.names import (
    DHW_BOTTOM_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    HP_ELECTRIC_POWER,
    HP_ELECTRIC_TOTAL_KWH,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TARGET_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    OUTDOOR_TEMPERATURE,
    P1_EXPORT_TOTAL_KWH,
    P1_IMPORT_TOTAL_KWH,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    PV_TOTAL_KWH,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
)
from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.infrastructure.database.orm_models import Sample15m
from home_optimizer.infrastructure.database.session import Database

DEFAULT_LEGACY_MEASUREMENT_SOURCE = "legacy_measurement_15m"
_INTERVAL_HOURS = 0.25


@dataclass(frozen=True)
class LegacyMeasurementImportSummary:
    measurement_rows: int
    generated_samples_15m: int
    written_samples_15m: int
    solar_forecast_rows: int = 0


@dataclass(frozen=True)
class _NumericColumnMapping:
    legacy_column: str
    name: str
    category: str
    unit: str | None


@dataclass
class _CumulativeTotals:
    p1_import_total_kwh: float = 0.0
    p1_export_total_kwh: float = 0.0
    pv_total_kwh: float = 0.0
    hp_electric_total_kwh: float = 0.0


_NUMERIC_COLUMN_MAPPINGS = (
    _NumericColumnMapping("room_temp", ROOM_TEMPERATURE, "building", "°C"),
    _NumericColumnMapping(
        "target_setpoint",
        HP_SUPPLY_TARGET_TEMPERATURE,
        "heatpump",
        "°C",
    ),
    _NumericColumnMapping("shutter_room", SHUTTER_LIVING_ROOM, "building", "%"),
    _NumericColumnMapping("supply_temp", HP_SUPPLY_TEMPERATURE, "heatpump", "°C"),
    _NumericColumnMapping("return_temp", HP_RETURN_TEMPERATURE, "heatpump", "°C"),
    _NumericColumnMapping("dhw_top", DHW_TOP_TEMPERATURE, "dhw", "°C"),
    _NumericColumnMapping("dhw_bottom", DHW_BOTTOM_TEMPERATURE, "dhw", "°C"),
    _NumericColumnMapping("pv_actual", PV_OUTPUT_POWER, "energy", "kW"),
    _NumericColumnMapping("wp_actual", HP_ELECTRIC_POWER, "energy", "kW"),
)

_HVAC_MODE_MAP = {
    0: "off",
    1: "dhw",
    2: "heat",
    4: "legionella",
}


def import_legacy_measurements(
    *,
    legacy_db_path: str,
    target_db_path: str,
    source: str = DEFAULT_LEGACY_MEASUREMENT_SOURCE,
    timezone_name: str = "UTC",
    replace: bool = False,
    dry_run: bool = False,
    batch_size: int = 1_000,
) -> LegacyMeasurementImportSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    database = Database(target_db_path)
    database.init_schema()

    totals = _CumulativeTotals()
    measurement_rows = 0
    solar_forecast_rows = 0
    generated_samples = 0
    written_samples = 0
    pending_rows: list[dict[str, object | None]] = []

    if replace and not dry_run:
        with database.session() as session:
            session.execute(delete(Sample15m).where(Sample15m.source == source))
            session.commit()

    with _connect_legacy_database(legacy_db_path) as connection:
        for row in connection.execute(
            """
            SELECT
                timestamp,
                grid_import,
                grid_export,
                pv_actual,
                wp_actual,
                room_temp,
                dhw_top,
                dhw_bottom,
                target_setpoint,
                supply_temp,
                return_temp,
                hvac_mode,
                shutter_room,
                wp_ufh,
                wp_dhw,
                wp_leg
            FROM measurement
            WHERE timestamp IS NOT NULL
            ORDER BY timestamp
            """
        ):
            measurement_rows += 1
            timestamp_utc = _parse_legacy_timestamp(row["timestamp"], timezone_name)
            samples = _map_measurement_row(
                row=row,
                timestamp_utc=timestamp_utc,
                source=source,
                totals=totals,
            )
            generated_samples += len(samples)
            pending_rows.extend(_sample_to_row(sample) for sample in samples)

            if not dry_run and len(pending_rows) >= batch_size:
                written_samples += _write_batch(database, pending_rows)
                pending_rows = []

        if _table_exists(connection, "solar_forecast"):
            for row in connection.execute(
                """
                SELECT timestamp, temp
                FROM solar_forecast
                WHERE timestamp IS NOT NULL AND temp IS NOT NULL
                ORDER BY timestamp
                """
            ):
                solar_forecast_rows += 1
                timestamp_utc = _parse_legacy_timestamp(row["timestamp"], timezone_name)
                sample = _map_solar_forecast_row(
                    row=row,
                    timestamp_utc=timestamp_utc,
                    source=source,
                )
                if sample is None:
                    continue

                generated_samples += 1
                pending_rows.append(_sample_to_row(sample))

                if not dry_run and len(pending_rows) >= batch_size:
                    written_samples += _write_batch(database, pending_rows)
                    pending_rows = []

    if not dry_run and pending_rows:
        written_samples += _write_batch(database, pending_rows)

    return LegacyMeasurementImportSummary(
        measurement_rows=measurement_rows,
        solar_forecast_rows=solar_forecast_rows,
        generated_samples_15m=generated_samples,
        written_samples_15m=written_samples,
    )


def _connect_legacy_database(legacy_db_path: str) -> sqlite3.Connection:
    database_path = Path(legacy_db_path)
    if not database_path.exists():
        raise FileNotFoundError(f"Legacy database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_legacy_timestamp(raw_value: str, timezone_name: str) -> datetime:
    naive_local = datetime.fromisoformat(raw_value.replace(" ", "T"))
    localized = naive_local.replace(tzinfo=ZoneInfo(timezone_name))
    return localized.astimezone(timezone.utc)


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _map_measurement_row(
    *,
    row: sqlite3.Row,
    timestamp_utc: datetime,
    source: str,
    totals: _CumulativeTotals,
) -> list[Sample15m]:
    samples: list[Sample15m] = []

    for mapping in _NUMERIC_COLUMN_MAPPINGS:
        value = _real_value(row[mapping.legacy_column])
        if value is None:
            continue
        samples.append(
            _numeric_sample(
                timestamp_utc=timestamp_utc,
                name=mapping.name,
                source=source,
                entity_id=_legacy_entity_id(mapping.legacy_column),
                category=mapping.category,
                unit=mapping.unit,
                value=value,
            )
        )

    grid_import_kw = _real_value(row["grid_import"])
    grid_export_kw = _real_value(row["grid_export"])
    if grid_import_kw is not None or grid_export_kw is not None:
        net_power_kw = (grid_import_kw or 0.0) - (grid_export_kw or 0.0)
        samples.append(
            _numeric_sample(
                timestamp_utc=timestamp_utc,
                name=P1_NET_POWER,
                source=source,
                entity_id=_legacy_entity_id("grid_net_power"),
                category="energy",
                unit="kW",
                value=net_power_kw,
            )
        )

    if grid_import_kw is not None:
        totals.p1_import_total_kwh += grid_import_kw * _INTERVAL_HOURS
        samples.append(
            _numeric_sample(
                timestamp_utc=timestamp_utc,
                name=P1_IMPORT_TOTAL_KWH,
                source=source,
                entity_id=_legacy_entity_id("grid_import_total"),
                category="energy",
                unit="kWh",
                value=totals.p1_import_total_kwh,
            )
        )

    if grid_export_kw is not None:
        totals.p1_export_total_kwh += grid_export_kw * _INTERVAL_HOURS
        samples.append(
            _numeric_sample(
                timestamp_utc=timestamp_utc,
                name=P1_EXPORT_TOTAL_KWH,
                source=source,
                entity_id=_legacy_entity_id("grid_export_total"),
                category="energy",
                unit="kWh",
                value=totals.p1_export_total_kwh,
            )
        )

    pv_output_kw = _real_value(row["pv_actual"])
    if pv_output_kw is not None:
        totals.pv_total_kwh += pv_output_kw * _INTERVAL_HOURS
        samples.append(
            _numeric_sample(
                timestamp_utc=timestamp_utc,
                name=PV_TOTAL_KWH,
                source=source,
                entity_id=_legacy_entity_id("pv_total"),
                category="energy",
                unit="kWh",
                value=totals.pv_total_kwh,
            )
        )

    hp_power_kw = _real_value(row["wp_actual"])
    if hp_power_kw is not None:
        totals.hp_electric_total_kwh += hp_power_kw * _INTERVAL_HOURS
        samples.append(
            _numeric_sample(
                timestamp_utc=timestamp_utc,
                name=HP_ELECTRIC_TOTAL_KWH,
                source=source,
                entity_id=_legacy_entity_id("wp_total"),
                category="energy",
                unit="kWh",
                value=totals.hp_electric_total_kwh,
            )
        )

    hvac_mode = _mode_value(row["hvac_mode"])
    if hvac_mode is not None:
        samples.append(
            _text_sample(
                timestamp_utc=timestamp_utc,
                name=HP_MODE,
                source=source,
                entity_id=_legacy_entity_id("hvac_mode"),
                category="heatpump",
                value=hvac_mode,
            )
        )

    return samples


def _map_solar_forecast_row(
    *,
    row: sqlite3.Row,
    timestamp_utc: datetime,
    source: str,
) -> Sample15m | None:
    temperature_c = _real_value(row["temp"])
    if temperature_c is None:
        return None

    return _numeric_sample(
        timestamp_utc=timestamp_utc,
        name=OUTDOOR_TEMPERATURE,
        source=source,
        entity_id="legacy.solar_forecast.temp",
        category="building",
        unit="°C",
        value=temperature_c,
    )


def _real_value(raw_value: object | None) -> float | None:
    if raw_value is None:
        return None
    return float(cast(float | int | str, raw_value))


def _mode_value(raw_value: object | None) -> str | None:
    if raw_value is None:
        return None
    mode_code = int(cast(int | str, raw_value))
    return _HVAC_MODE_MAP.get(mode_code, f"mode_{mode_code}")


def _legacy_entity_id(suffix: str) -> str:
    return f"legacy.measurement.{suffix}"


def _numeric_sample(
    *,
    timestamp_utc: datetime,
    name: str,
    source: str,
    entity_id: str,
    category: str,
    unit: str | None,
    value: float,
) -> Sample15m:
    timestamp = normalize_utc_timestamp(timestamp_utc)
    return Sample15m(
        timestamp_15m_utc=timestamp,
        name=name,
        source=source,
        entity_id=entity_id,
        category=category,
        unit=unit,
        mean_real=value,
        min_real=value,
        max_real=value,
        last_real=value,
        sample_count=1,
    )


def _text_sample(
    *,
    timestamp_utc: datetime,
    name: str,
    source: str,
    entity_id: str,
    category: str,
    value: str,
) -> Sample15m:
    return Sample15m(
        timestamp_15m_utc=normalize_utc_timestamp(timestamp_utc),
        name=name,
        source=source,
        entity_id=entity_id,
        category=category,
        unit=None,
        last_text=value,
        sample_count=1,
    )


def _sample_to_row(sample: Sample15m) -> dict[str, object | None]:
    return {
        "timestamp_15m_utc": sample.timestamp_15m_utc,
        "name": sample.name,
        "source": sample.source,
        "entity_id": sample.entity_id,
        "category": sample.category,
        "unit": sample.unit,
        "mean_real": sample.mean_real,
        "min_real": sample.min_real,
        "max_real": sample.max_real,
        "last_real": sample.last_real,
        "last_text": sample.last_text,
        "last_bool": sample.last_bool,
        "sample_count": sample.sample_count,
    }


def _write_batch(database: Database, rows: list[dict[str, object | None]]) -> int:
    with database.session() as session:
        result = session.execute(insert(Sample15m).values(rows).prefix_with("OR REPLACE"))
        session.commit()
    return int(getattr(result, "rowcount", 0) or 0)

