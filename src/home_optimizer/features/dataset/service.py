from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    DHW_BOTTOM_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_ELECTRIC_POWER,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    OUTDOOR_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TARGET_MAX_TEMPERATURE,
    ROOM_TARGET_MIN_TEMPERATURE,
    ROOM_TARGET_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    FixedPricing,
    NumericPoint,
    NumericSeries,
    TemperatureTargetWindow,
    build_daily_price_series,
    build_daily_target_band_series,
    dataset_numeric_signal_names,
    dataset_numeric_signal_spec,
    dataset_text_signal_names,
    ensure_utc,
    normalize_utc_timestamp,
)
from home_optimizer.domain.pricing import (
    DEFAULT_DYNAMIC_PRICE_SOURCE,
    DEFAULT_FIXED_PRICE_SOURCE,
    PriceInterval,
    price_series_from_intervals,
)
from home_optimizer.domain.time import parse_datetime
from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow, MpcDatasetSummary
from home_optimizer.features.dataset.ports import DatasetSampleFrameReader

_OCCUPIED_MARGIN_C = 0.25
_DHW_DRAW_DROP_THRESHOLD_C = 0.75
_MODE_SPACE_TOKENS = ("heat", "heating", "ufh")
_MODE_DHW_TOKENS = ("dhw", "sww")
_MODE_OFF_TOKENS = ("off", "idle", "standby", "none")
_MIN_PLAUSIBLE_COP = 1.0
_MAX_PLAUSIBLE_COP = 8.0
_THERMAL_FACTOR_KW_PER_LMIN_DELTA_C = 4186.0 / 60000.0


def _resolve_price_series(
    reader: DatasetSampleFrameReader,
    settings: AppSettings,
    *,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
) -> NumericSeries:
    source = (
        DEFAULT_DYNAMIC_PRICE_SOURCE
        if settings.electricity_pricing.mode == "dynamic"
        else DEFAULT_FIXED_PRICE_SOURCE
    )
    raw_intervals = reader.read_electricity_price_intervals(
        start_time=start_time,
        end_time=end_time,
        names=["electricity_price"],
        sources=[source],
    )
    if not raw_intervals.empty:
        intervals = [
            PriceInterval(
                start_time_utc=parse_datetime(row["start_time_utc"]),
                end_time_utc=parse_datetime(row["end_time_utc"]),
                source=str(row["source"]),
                name=str(row["name"]),
                unit=str(row["unit"]),
                value=float(row["value"]),
            )
            for row in raw_intervals.to_dict(orient="records")
        ]
        return price_series_from_intervals(
            intervals,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )

    if isinstance(settings.electricity_pricing, FixedPricing):
        return build_daily_price_series(
            settings.electricity_pricing,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )

    return NumericSeries(name="electricity_price", unit="EUR/kWh", points=[])


def _classify_hp_mode(mode: str | None) -> tuple[int, int, int]:
    if mode is None or not isinstance(mode, str):
        return 0, 0, 1

    normalized = mode.strip().lower()
    if any(token in normalized for token in _MODE_DHW_TOKENS):
        return 0, 1, 0
    if any(token in normalized for token in _MODE_SPACE_TOKENS):
        return 1, 0, 0
    if any(token in normalized for token in _MODE_OFF_TOKENS):
        return 0, 0, 1
    return 0, 0, 1


def _price_export_value(settings: AppSettings) -> float:
    if isinstance(settings.electricity_pricing, FixedPricing):
        return settings.electricity_pricing.feed_in_tariff
    return 0.0


def _occupied_flag(
    target_temperature: float | None,
    schedule: list[TemperatureTargetWindow],
) -> int:
    if target_temperature is None or not schedule:
        return 0

    minimum_target = min(window.target for window in schedule)
    return int(target_temperature > minimum_target + _OCCUPIED_MARGIN_C)


def _dhw_draw_proxy_c(
    current_value: float | None,
    previous_value: float | None,
    *,
    mode_dhw: int,
) -> float:
    if mode_dhw:
        return 0.0
    if current_value is None or previous_value is None:
        return 0.0
    return max(0.0, previous_value - current_value)


def _detect_dhw_draw(
    current_value: float | None,
    previous_value: float | None,
    *,
    mode_dhw: int,
) -> int:
    return int(
        _dhw_draw_proxy_c(current_value, previous_value, mode_dhw=mode_dhw)
        >= _DHW_DRAW_DROP_THRESHOLD_C
    )


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _validate_row(
    *,
    mode_space: int,
    mode_dhw: int,
    mode_off: int,
    defrost_active: int,
    booster_heater_active: int,
    flow_l_min: float | None,
    thermal_output_estimate: float | None,
    cop_estimate: float | None,
) -> tuple[bool, bool, bool, list[str]]:
    reasons: list[str] = []
    active_mode_count = mode_space + mode_dhw + mode_off

    if active_mode_count != 1:
        reasons.append("invalid_mode_combination")
    if defrost_active:
        reasons.append("defrost_active")
    if booster_heater_active:
        reasons.append("booster_heater_active")
    if flow_l_min is not None and flow_l_min < 0:
        reasons.append("negative_flow")
    if thermal_output_estimate is not None and thermal_output_estimate < 0:
        reasons.append("negative_thermal_output")

    room_valid = active_mode_count == 1 and not defrost_active and not booster_heater_active
    dhw_valid = active_mode_count == 1 and not defrost_active and not booster_heater_active

    cop_valid = (
        active_mode_count == 1
        and not defrost_active
        and not booster_heater_active
        and thermal_output_estimate is not None
        and thermal_output_estimate > 0
        and cop_estimate is not None
        and _MIN_PLAUSIBLE_COP <= cop_estimate <= _MAX_PLAUSIBLE_COP
    )
    if cop_estimate is not None and not (_MIN_PLAUSIBLE_COP <= cop_estimate <= _MAX_PLAUSIBLE_COP):
        reasons.append("cop_out_of_range")
    if thermal_output_estimate is None or thermal_output_estimate <= 0:
        reasons.append("missing_or_nonpositive_thermal_output")

    return room_valid, dhw_valid, cop_valid, reasons


def _numeric_series_points_frame(series_list: list[NumericSeries]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series in series_list:
        for point in series.points:
            rows.append(
                {
                    "name": series.name,
                    "timestamp_utc": pd.Timestamp(point.timestamp, tz="UTC"),
                    "value": float(point.value),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["name", "timestamp_utc", "value"])
    return pd.DataFrame(rows).sort_values(["name", "timestamp_utc"]).reset_index(drop=True)


def _raw_frame_covers_range(
    raw: pd.DataFrame,
    *,
    timestamp_column: str,
    start_time: datetime,
    end_time: datetime,
    source_interval_minutes: int,
) -> bool:
    if raw.empty or timestamp_column not in raw.columns:
        return False

    timestamps = pd.to_datetime(raw[timestamp_column], utc=True).dropna()
    if timestamps.empty:
        return False

    tolerance = pd.Timedelta(minutes=source_interval_minutes)
    earliest = timestamps.min()
    latest = timestamps.max()
    return earliest <= pd.Timestamp(start_time) + tolerance and latest >= pd.Timestamp(end_time) - tolerance


class MpcDatasetService:
    def __init__(self, samples_reader: DatasetSampleFrameReader, settings: AppSettings) -> None:
        self.samples_reader = samples_reader
        self.settings = settings

    def _measurement_frame(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
    ) -> pd.DataFrame:
        requested_names = dataset_numeric_signal_names(source="measurement") + dataset_text_signal_names(
            source="measurement"
        )
        raw = self.samples_reader.read_samples(
            interval_minutes=1,
            start_time=start_time,
            end_time=end_time,
            names=requested_names,
        ).copy()

        timestamp_column = (
            "timestamp_minute_utc" if "timestamp_minute_utc" in raw.columns else "timestamp_15m_utc"
        )
        if not _raw_frame_covers_range(
            raw,
            timestamp_column=timestamp_column,
            start_time=start_time,
            end_time=end_time,
            source_interval_minutes=1 if timestamp_column == "timestamp_minute_utc" else 15,
        ):
            raw = self.samples_reader.read_samples(
                interval_minutes=15,
                start_time=start_time,
                end_time=end_time,
                names=requested_names,
            ).copy()

        grid = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    start=start_time,
                    end=end_time,
                    freq=f"{interval_minutes}min",
                    inclusive="left",
                    tz="UTC",
                )
            }
        )
        if grid.empty:
            return grid

        if raw.empty:
            return grid

        timestamp_column = (
            "timestamp_minute_utc" if "timestamp_minute_utc" in raw.columns else "timestamp_15m_utc"
        )
        raw["timestamp_utc"] = pd.to_datetime(raw[timestamp_column], utc=True)
        raw = raw.sort_values(["timestamp_utc", "name", "source"]).reset_index(drop=True)

        numeric_cols = ["mean_real", "min_real", "max_real", "last_real", "last_bool"]
        for column in numeric_cols:
            if column in raw.columns:
                raw[column] = pd.to_numeric(raw[column], errors="coerce")

        if "mean_real" in raw.columns:
            raw["value_mean"] = (
                raw["mean_real"]
                .combine_first(raw["last_real"])
                .combine_first(raw["max_real"])
                .combine_first(raw["min_real"])
                .combine_first(raw["last_bool"])
            )
        if "last_real" in raw.columns:
            raw["value_sample"] = (
                raw["last_real"]
                .combine_first(raw["mean_real"])
                .combine_first(raw["max_real"])
                .combine_first(raw["min_real"])
                .combine_first(raw["last_bool"])
            )
        if "last_bool" in raw.columns:
            raw["value_flag"] = (
                raw["last_bool"]
                .combine_first(raw["last_real"])
                .combine_first(raw["mean_real"])
                .combine_first(raw["max_real"])
                .combine_first(raw["min_real"])
            )

        result = grid.copy()

        sample_numeric_names = [
            name
            for name in dataset_numeric_signal_names(source="measurement")
            if dataset_numeric_signal_spec(name).resample_method == "sample"
        ]
        mean_numeric_names = [
            name
            for name in dataset_numeric_signal_names(source="measurement")
            if dataset_numeric_signal_spec(name).resample_method == "mean"
        ]
        flag_numeric_names = [
            name
            for name in dataset_numeric_signal_names(source="measurement")
            if dataset_numeric_signal_spec(name).resample_method == "window_flag"
        ]

        for name in sample_numeric_names:
            signal = raw.loc[raw["name"] == name, ["timestamp_utc", "value_sample"]].dropna()
            if signal.empty:
                result[name] = pd.NA
                continue
            signal = signal.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values(
                "timestamp_utc"
            )
            merged = pd.merge_asof(
                result[["timestamp_utc"]],
                signal.rename(columns={"value_sample": name}),
                on="timestamp_utc",
                direction="backward",
            )
            result[name] = merged[name]

        if mean_numeric_names or flag_numeric_names:
            raw["bucket_start"] = start_time + pd.to_timedelta(
                ((raw["timestamp_utc"] - start_time).dt.total_seconds() // (interval_minutes * 60))
                * interval_minutes,
                unit="m",
            )
            raw = raw.loc[(raw["bucket_start"] >= start_time) & (raw["bucket_start"] < end_time)]

        for name in mean_numeric_names:
            signal = raw.loc[raw["name"] == name, ["bucket_start", "value_mean"]].dropna()
            if signal.empty:
                result[name] = pd.NA
                continue
            aggregated = (
                signal.groupby("bucket_start", as_index=False)["value_mean"].mean().rename(
                    columns={"bucket_start": "timestamp_utc", "value_mean": name}
                )
            )
            result = result.merge(aggregated, on="timestamp_utc", how="left")

        for name in flag_numeric_names:
            signal = raw.loc[raw["name"] == name, ["bucket_start", "value_flag"]].dropna()
            if signal.empty:
                result[name] = 0
                continue
            aggregated = (
                signal.groupby("bucket_start", as_index=False)["value_flag"]
                .max()
                .rename(columns={"bucket_start": "timestamp_utc", "value_flag": name})
            )
            result = result.merge(aggregated, on="timestamp_utc", how="left")
            result[name] = result[name].fillna(0).gt(0).astype(int)

        for name in dataset_text_signal_names(source="measurement"):
            signal = raw.loc[raw["name"] == name, ["timestamp_utc", "last_text"]].dropna()
            if signal.empty:
                result[name] = pd.NA
                continue
            signal = signal.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values(
                "timestamp_utc"
            )
            merged = pd.merge_asof(
                result[["timestamp_utc"]],
                signal.rename(columns={"last_text": name}),
                on="timestamp_utc",
                direction="backward",
            )
            result[name] = merged[name]

        return result

    def _forecast_frame(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
    ) -> pd.DataFrame:
        grid = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    start=start_time,
                    end=end_time,
                    freq=f"{interval_minutes}min",
                    inclusive="left",
                    tz="UTC",
                )
            }
        )
        names = dataset_numeric_signal_names(source="forecast")
        if not names:
            return grid

        forecast_frame = self.samples_reader.read_forecast_values(
            start_time=start_time,
            end_time=end_time,
            names=names,
        ).copy()
        if forecast_frame.empty:
            for name in names:
                grid[name] = pd.NA
            return grid

        forecast_frame["timestamp_utc"] = pd.to_datetime(
            forecast_frame["forecast_time_utc"],
            utc=True,
        )
        forecast_frame["value"] = pd.to_numeric(forecast_frame["value"], errors="coerce")
        forecast_frame["bucket_start"] = start_time + pd.to_timedelta(
            (
                (forecast_frame["timestamp_utc"] - start_time).dt.total_seconds()
                // (interval_minutes * 60)
            )
            * interval_minutes,
            unit="m",
        )
        forecast_frame = forecast_frame.loc[
            (forecast_frame["bucket_start"] >= start_time)
            & (forecast_frame["bucket_start"] < end_time)
        ]

        result = grid
        for name in names:
            signal = forecast_frame.loc[
                forecast_frame["name"] == name, ["bucket_start", "value"]
            ].dropna()
            if signal.empty:
                result[name] = pd.NA
                continue
            aggregated = (
                signal.groupby("bucket_start", as_index=False)["value"].mean().rename(
                    columns={"bucket_start": "timestamp_utc", "value": name}
                )
            )
            result = result.merge(aggregated, on="timestamp_utc", how="left")

        return result

    def _series_value_at_grid(
        self,
        series: NumericSeries,
        grid: pd.DataFrame,
        column_name: str,
    ) -> pd.Series:
        if not series.points:
            return pd.Series([pd.NA] * len(grid), index=grid.index, dtype="object")

        frame = pd.DataFrame(
            {
                "timestamp_utc": [pd.Timestamp(point.timestamp, tz="UTC") for point in series.points],
                column_name: [point.value for point in series.points],
            }
        ).sort_values("timestamp_utc")
        merged = pd.merge_asof(
            grid[["timestamp_utc"]],
            frame,
            on="timestamp_utc",
            direction="backward",
        )
        return merged[column_name]

    def build_dataset(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15,
    ) -> MpcDataset:
        start_time_utc = ensure_utc(start_time)
        end_time_utc = ensure_utc(end_time)
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")
        if end_time_utc <= start_time_utc:
            raise ValueError("end_time must be after start_time")

        measurement_frame = self._measurement_frame(
            start_time=start_time_utc,
            end_time=end_time_utc,
            interval_minutes=interval_minutes,
        )
        forecast_frame = self._forecast_frame(
            start_time=start_time_utc,
            end_time=end_time_utc,
            interval_minutes=interval_minutes,
        )
        frame = measurement_frame.merge(
            forecast_frame,
            on="timestamp_utc",
            how="left",
            suffixes=("", "_forecast"),
        )

        grid = frame[["timestamp_utc"]].copy()

        price_series = _resolve_price_series(
            self.samples_reader,
            self.settings,
            start_time=start_time_utc,
            end_time=end_time_utc,
            interval_minutes=interval_minutes,
        )
        room_target, room_target_min, room_target_max = build_daily_target_band_series(
            self.settings.room_target,
            start_time=start_time_utc,
            end_time=end_time_utc,
            target_name=ROOM_TARGET_TEMPERATURE,
            minimum_name=ROOM_TARGET_MIN_TEMPERATURE,
            maximum_name=ROOM_TARGET_MAX_TEMPERATURE,
            interval_minutes=interval_minutes,
        )

        frame["price_import_eur_kwh"] = self._series_value_at_grid(
            price_series,
            grid,
            "price_import_eur_kwh",
        )
        frame["room_target_temperature_c"] = self._series_value_at_grid(
            room_target,
            grid,
            "room_target_temperature_c",
        )
        frame["room_target_min_temperature_c"] = self._series_value_at_grid(
            room_target_min,
            grid,
            "room_target_min_temperature_c",
        )
        frame["room_target_max_temperature_c"] = self._series_value_at_grid(
            room_target_max,
            grid,
            "room_target_max_temperature_c",
        )

        frame["hp_mode_raw"] = frame.get(HP_MODE)
        mode_values = frame["hp_mode_raw"].map(_classify_hp_mode)
        frame["mode_space"] = mode_values.map(lambda item: item[0])
        frame["mode_dhw"] = mode_values.map(lambda item: item[1])
        frame["mode_off"] = mode_values.map(lambda item: item[2])

        frame["hp_delta_t_c"] = frame[HP_SUPPLY_TEMPERATURE] - frame[HP_RETURN_TEMPERATURE]
        frame["thermal_output_estimate_kw"] = (
            frame[HP_FLOW] * frame["hp_delta_t_c"] * _THERMAL_FACTOR_KW_PER_LMIN_DELTA_C
        )
        frame.loc[frame["thermal_output_estimate_kw"] < 0, "thermal_output_estimate_kw"] = 0.0

        frame["cop_estimate"] = (
            frame["thermal_output_estimate_kw"] / frame[HP_ELECTRIC_POWER]
        )
        frame.loc[frame[HP_ELECTRIC_POWER].isna() | (frame[HP_ELECTRIC_POWER] <= 0), "cop_estimate"] = pd.NA

        shutter_fraction = frame[SHUTTER_LIVING_ROOM].clip(lower=0, upper=100) / 100.0
        shutter_fraction = shutter_fraction.where(frame[SHUTTER_LIVING_ROOM].notna(), 1.0)
        frame["solar_gain_proxy_w_m2"] = frame[GTI_LIVING_ROOM_WINDOWS] * shutter_fraction
        frame["price_export_eur_kwh"] = _price_export_value(self.settings)

        frame["occupied_flag"] = frame["room_target_temperature_c"].map(
            lambda target: _occupied_flag(float(target) if pd.notna(target) else None, self.settings.room_target)
        )

        previous_dhw_top = frame[DHW_TOP_TEMPERATURE].shift(1)
        frame["dhw_draw_proxy_c"] = [
            _dhw_draw_proxy_c(
                float(current) if pd.notna(current) else None,
                float(previous) if pd.notna(previous) else None,
                mode_dhw=int(mode_dhw),
            )
            for current, previous, mode_dhw in zip(
                frame[DHW_TOP_TEMPERATURE],
                previous_dhw_top,
                frame["mode_dhw"],
                strict=False,
            )
        ]
        frame["dhw_draw_detected"] = [
            _detect_dhw_draw(
                float(current) if pd.notna(current) else None,
                float(previous) if pd.notna(previous) else None,
                mode_dhw=int(mode_dhw),
            )
            for current, previous, mode_dhw in zip(
                frame[DHW_TOP_TEMPERATURE],
                previous_dhw_top,
                frame["mode_dhw"],
                strict=False,
            )
        ]

        rows: list[MpcDatasetRow] = []
        for record in frame.to_dict(orient="records"):
            room_valid, dhw_valid, cop_valid, exclusion_reasons = _validate_row(
                mode_space=int(record["mode_space"]),
                mode_dhw=int(record["mode_dhw"]),
                mode_off=int(record["mode_off"]),
                defrost_active=int(record.get(DEFROST_ACTIVE, 0) or 0),
                booster_heater_active=int(record.get(BOOSTER_HEATER_ACTIVE, 0) or 0),
                flow_l_min=float(record[HP_FLOW]) if pd.notna(record.get(HP_FLOW)) else None,
                thermal_output_estimate=(
                    float(record["thermal_output_estimate_kw"])
                    if pd.notna(record.get("thermal_output_estimate_kw"))
                    else None
                ),
                cop_estimate=(
                    float(record["cop_estimate"]) if pd.notna(record.get("cop_estimate")) else None
                ),
            )
            rows.append(
                MpcDatasetRow(
                    timestamp_utc=record["timestamp_utc"].to_pydatetime(),
                    room_temperature_c=(
                        float(record[ROOM_TEMPERATURE]) if pd.notna(record.get(ROOM_TEMPERATURE)) else None
                    ),
                    outdoor_temperature_c=(
                        float(record[OUTDOOR_TEMPERATURE]) if pd.notna(record.get(OUTDOOR_TEMPERATURE)) else None
                    ),
                    dhw_top_temperature_c=(
                        float(record[DHW_TOP_TEMPERATURE]) if pd.notna(record.get(DHW_TOP_TEMPERATURE)) else None
                    ),
                    dhw_bottom_temperature_c=(
                        float(record[DHW_BOTTOM_TEMPERATURE])
                        if pd.notna(record.get(DHW_BOTTOM_TEMPERATURE))
                        else None
                    ),
                    hp_electric_power_kw=(
                        float(record[HP_ELECTRIC_POWER]) if pd.notna(record.get(HP_ELECTRIC_POWER)) else None
                    ),
                    hp_mode_raw=_optional_string(record.get("hp_mode_raw")),
                    mode_space=int(record["mode_space"]),
                    mode_dhw=int(record["mode_dhw"]),
                    mode_off=int(record["mode_off"]),
                    defrost_active=int(record.get(DEFROST_ACTIVE, 0) or 0),
                    booster_heater_active=int(record.get(BOOSTER_HEATER_ACTIVE, 0) or 0),
                    pv_output_power_kw=(
                        float(record[PV_OUTPUT_POWER]) if pd.notna(record.get(PV_OUTPUT_POWER)) else None
                    ),
                    net_power_kw=(
                        float(record[P1_NET_POWER]) if pd.notna(record.get(P1_NET_POWER)) else None
                    ),
                    shutter_position_pct=(
                        float(record[SHUTTER_LIVING_ROOM])
                        if pd.notna(record.get(SHUTTER_LIVING_ROOM))
                        else None
                    ),
                    thermostat_setpoint_c=(
                        float(record[THERMOSTAT_SETPOINT])
                        if pd.notna(record.get(THERMOSTAT_SETPOINT))
                        else None
                    ),
                    room_target_temperature_c=(
                        float(record["room_target_temperature_c"])
                        if pd.notna(record.get("room_target_temperature_c"))
                        else None
                    ),
                    room_target_min_temperature_c=(
                        float(record["room_target_min_temperature_c"])
                        if pd.notna(record.get("room_target_min_temperature_c"))
                        else None
                    ),
                    room_target_max_temperature_c=(
                        float(record["room_target_max_temperature_c"])
                        if pd.notna(record.get("room_target_max_temperature_c"))
                        else None
                    ),
                    supply_temperature_c=(
                        float(record[HP_SUPPLY_TEMPERATURE])
                        if pd.notna(record.get(HP_SUPPLY_TEMPERATURE))
                        else None
                    ),
                    return_temperature_c=(
                        float(record[HP_RETURN_TEMPERATURE])
                        if pd.notna(record.get(HP_RETURN_TEMPERATURE))
                        else None
                    ),
                    flow_l_min=float(record[HP_FLOW]) if pd.notna(record.get(HP_FLOW)) else None,
                    hp_delta_t_c=(
                        float(record["hp_delta_t_c"]) if pd.notna(record.get("hp_delta_t_c")) else None
                    ),
                    thermal_output_estimate_kw=(
                        float(record["thermal_output_estimate_kw"])
                        if pd.notna(record.get("thermal_output_estimate_kw"))
                        else None
                    ),
                    cop_estimate=(
                        float(record["cop_estimate"]) if pd.notna(record.get("cop_estimate")) else None
                    ),
                    solar_irradiance_w_m2=(
                        float(record[GTI_LIVING_ROOM_WINDOWS])
                        if pd.notna(record.get(GTI_LIVING_ROOM_WINDOWS))
                        else None
                    ),
                    solar_gain_proxy_w_m2=(
                        float(record["solar_gain_proxy_w_m2"])
                        if pd.notna(record.get("solar_gain_proxy_w_m2"))
                        else None
                    ),
                    price_import_eur_kwh=(
                        float(record["price_import_eur_kwh"])
                        if pd.notna(record.get("price_import_eur_kwh"))
                        else None
                    ),
                    price_export_eur_kwh=float(record["price_export_eur_kwh"]),
                    occupied_flag=int(record["occupied_flag"]),
                    dhw_draw_proxy_c=float(record["dhw_draw_proxy_c"]),
                    dhw_draw_detected=int(record["dhw_draw_detected"]),
                    is_valid_for_room_identification=room_valid,
                    is_valid_for_dhw_identification=dhw_valid,
                    is_valid_for_cop_identification=cop_valid,
                    exclusion_reasons=exclusion_reasons,
                )
            )

        return MpcDataset(
            interval_minutes=interval_minutes,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            rows=rows,
        )

    def summarize_dataset(self, dataset: MpcDataset) -> MpcDatasetSummary:
        exclusion_reason_counts: dict[str, int] = {}
        for row in dataset.rows:
            for reason in row.exclusion_reasons:
                exclusion_reason_counts[reason] = exclusion_reason_counts.get(reason, 0) + 1

        return MpcDatasetSummary(
            total_rows=len(dataset.rows),
            mode_space_rows=sum(row.mode_space for row in dataset.rows),
            mode_dhw_rows=sum(row.mode_dhw for row in dataset.rows),
            mode_off_rows=sum(row.mode_off for row in dataset.rows),
            defrost_rows=sum(row.defrost_active for row in dataset.rows),
            booster_rows=sum(row.booster_heater_active for row in dataset.rows),
            valid_room_rows=sum(int(row.is_valid_for_room_identification) for row in dataset.rows),
            valid_dhw_rows=sum(int(row.is_valid_for_dhw_identification) for row in dataset.rows),
            valid_cop_rows=sum(int(row.is_valid_for_cop_identification) for row in dataset.rows),
            exclusion_reason_counts=exclusion_reason_counts,
        )
