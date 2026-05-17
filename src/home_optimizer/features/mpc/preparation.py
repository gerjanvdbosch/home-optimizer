from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median

from home_optimizer.domain import (
    FORECAST_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    GTI_PV,
)
from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.pricing import ElectricityPricingConfig, PriceInterval
from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.domain.time import ensure_utc
from home_optimizer.features.dataset import MpcDatasetService
from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.dataset.ports import DatasetSampleFrameReader
from home_optimizer.features.modeling import (
    RoomRcModel,
    RoomRcTrainer,
    StoredModelVersion,
    TrainedLinearRoomModel,
)
from home_optimizer.features.modeling.room_2r2c import RoomRC2StateParams
from home_optimizer.features.mpc.exogenous_features import trailing_exp_filter
from home_optimizer.features.mpc.horizon_builder import MpcHorizonBuilder
from home_optimizer.features.mpc.models import (
    MpcHorizonBuildRequest,
    MpcHorizonStep,
    MpcInitialState,
    Rc2StateMpcInitialState,
)
from home_optimizer.features.mpc.ports import ActiveRoomModelReaderPort


class SpaceHeatingMpcPreparationService:
    def __init__(
        self,
        *,
        samples_reader: DatasetSampleFrameReader,
        active_room_model_reader: ActiveRoomModelReaderPort,
        target_schedule: list[TemperatureTargetWindow],
        electricity_pricing: ElectricityPricingConfig | None = None,
        rc_trainer: RoomRcTrainer | None = None,
    ) -> None:
        self.samples_reader = samples_reader
        self.active_room_model_reader = active_room_model_reader
        self.target_schedule = list(target_schedule)
        self.electricity_pricing = electricity_pricing or _DynamicPricingLike()
        self.rc_trainer = rc_trainer or RoomRcTrainer()
        self.horizon_builder = MpcHorizonBuilder()

    def resolve_room_model_version(self, model_id: str | None) -> StoredModelVersion | None:
        if model_id is not None:
            return self.active_room_model_reader.get_room_model_version(model_id)
        return self.active_room_model_reader.get_active_room_model_version()

    def load_initial_rows(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        history_rows: int,
    ) -> list[MpcDatasetRow]:
        dataset = self.build_dataset(
            start_time=start_time_utc - timedelta(minutes=interval_minutes * history_rows),
            end_time=start_time_utc + timedelta(minutes=interval_minutes),
            interval_minutes=interval_minutes,
        )
        rows = [row for row in dataset.rows if row.timestamp_utc <= start_time_utc]
        if not rows:
            raise ValueError("No dataset rows available to estimate MPC initial state")
        return rows

    def build_dataset(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
    ) -> MpcDataset:
        dataset_service = MpcDatasetService(
            self.samples_reader,
            _MpcDatasetSettings(
                self.target_schedule,
                electricity_pricing=self.electricity_pricing,
            ),
        )
        return dataset_service.build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )

    def load_forecast_entries(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        horizon_steps: int,
        created_at_end_time: datetime | None = None,
    ) -> list[ForecastEntry]:
        end_time_utc = start_time_utc + timedelta(minutes=interval_minutes * horizon_steps)
        frame = self.samples_reader.read_forecast_values(
            start_time=start_time_utc,
            end_time=end_time_utc,
            created_at_end_time=created_at_end_time,
        )
        entries: list[ForecastEntry] = []
        for row in frame.to_dict(orient="records"):
            entries.append(
                ForecastEntry(
                    created_at_utc=ensure_utc(row["created_at_utc"]),
                    forecast_time_utc=ensure_utc(row["forecast_time_utc"]),
                    name=str(row["name"]),
                    value=float(row["value"]),
                    unit=str(row["unit"]) if row.get("unit") is not None else None,
                    source=str(row["source"]),
                )
            )
        return entries

    def load_price_intervals(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        horizon_steps: int,
    ) -> list[PriceInterval]:
        end_time_utc = start_time_utc + timedelta(minutes=interval_minutes * horizon_steps)
        frame = self.samples_reader.read_electricity_price_intervals(
            start_time=start_time_utc,
            end_time=end_time_utc,
        )
        intervals: list[PriceInterval] = []
        for row in frame.to_dict(orient="records"):
            intervals.append(
                PriceInterval(
                    start_time_utc=ensure_utc(row["start_time_utc"]),
                    end_time_utc=ensure_utc(row["end_time_utc"]),
                    source=str(row["source"]),
                    name=str(row["name"]),
                    unit=str(row["unit"]),
                    value=float(row["value"]),
                )
            )
        return intervals

    def build_forecast_horizon(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        horizon_steps: int,
        source_model: TrainedLinearRoomModel | RoomRcModel,
        operating_rows: list[MpcDatasetRow],
        effective_heating_kw: float,
        hp_electric_power_kw: float,
        export_price_eur_kwh: float,
        base_load_power_kw: float | None = None,
        forecast_entries: list[ForecastEntry] | None = None,
        price_intervals: list[PriceInterval] | None = None,
        pv_power_input_scale: float | None = None,
        created_at_end_time: datetime | None = None,
    ) -> list[MpcHorizonStep]:
        resolved_start_time = ensure_utc(start_time_utc)
        resolved_forecast_entries = (
            forecast_entries
            if forecast_entries is not None
            else self.load_forecast_entries(
                start_time_utc=resolved_start_time,
                interval_minutes=interval_minutes,
                horizon_steps=horizon_steps,
                created_at_end_time=created_at_end_time,
            )
        )
        resolved_price_intervals = (
            price_intervals
            if price_intervals is not None
            else self.load_price_intervals(
                start_time_utc=resolved_start_time,
                interval_minutes=interval_minutes,
                horizon_steps=horizon_steps,
            )
        )
        resolved_base_load_power_kw = (
            float(base_load_power_kw)
            if base_load_power_kw is not None
            else self.resolve_base_load_power_kw(operating_rows)
        )
        resolved_pv_power_input_scale = (
            float(pv_power_input_scale)
            if pv_power_input_scale is not None
            else self.infer_pv_power_input_scale(
                operating_rows,
                forecast_entries=resolved_forecast_entries,
                start_time_utc=resolved_start_time,
            )
        )
        return self.horizon_builder.build(
            MpcHorizonBuildRequest(
                start_time_utc=resolved_start_time,
                horizon_steps=horizon_steps,
                interval_minutes=interval_minutes,
                target_schedule=self.target_schedule,
                forecast_entries=resolved_forecast_entries,
                price_intervals=resolved_price_intervals,
                default_effective_heating_kw=effective_heating_kw,
                default_hp_electric_power_kw=hp_electric_power_kw,
                default_base_load_power_kw=resolved_base_load_power_kw,
                default_export_price_eur_kwh=export_price_eur_kwh,
                pv_power_input_scale=resolved_pv_power_input_scale,
                solar_gain_filter_alpha=self._solar_gain_filter_alpha(source_model),
                initial_filtered_solar_gain_kw=self._initial_filtered_solar_gain_kw(
                    operating_rows,
                    source_model=source_model,
                    start_time_utc=resolved_start_time,
                ),
                local_timezone=self._local_timezone(source_model),
            )
        )

    def build_historical_horizon(
        self,
        rows: list[MpcDatasetRow],
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
        effective_heating_kw: float,
        hp_electric_power_kw: float,
        export_price_eur_kwh: float,
        history_rows: list[MpcDatasetRow] | None = None,
    ) -> list[MpcHorizonStep]:
        if not rows:
            return []
        resolved_rows = list(rows)
        resolved_history_rows = list(history_rows or [])
        operating_rows = [*resolved_history_rows, *resolved_rows]
        forecast_entries = self._forecast_entries_from_rows(
            resolved_rows,
            source_model=source_model,
        )
        price_intervals = self._price_intervals_from_rows(resolved_rows)
        horizon = self.horizon_builder.build(
            MpcHorizonBuildRequest(
                start_time_utc=resolved_rows[0].timestamp_utc,
                horizon_steps=len(resolved_rows),
                interval_minutes=self._interval_minutes_from_rows(resolved_rows),
                target_schedule=self.target_schedule,
                forecast_entries=forecast_entries,
                price_intervals=price_intervals,
                default_effective_heating_kw=effective_heating_kw,
                default_hp_electric_power_kw=hp_electric_power_kw,
                default_base_load_power_kw=0.0,
                default_export_price_eur_kwh=export_price_eur_kwh,
                pv_power_input_scale=1.0,
                solar_gain_input_scale=self._solar_gain_input_scale(source_model),
                solar_gain_filter_alpha=self._solar_gain_filter_alpha(source_model),
                initial_filtered_solar_gain_kw=self._initial_filtered_solar_gain_kw(
                    operating_rows,
                    source_model=source_model,
                    start_time_utc=resolved_rows[0].timestamp_utc,
                ),
                local_timezone=self._local_timezone(source_model),
                occupied_by_timestamp={
                    row.timestamp_utc: float(row.occupied_flag) for row in resolved_rows
                },
                hp_electric_power_by_timestamp={
                    row.timestamp_utc: float(row.hp_electric_power_kw or hp_electric_power_kw)
                    for row in resolved_rows
                },
                base_load_power_by_timestamp={
                    row.timestamp_utc: self._row_base_load_power_kw(row) for row in resolved_rows
                },
                export_price_by_timestamp={
                    row.timestamp_utc: float(row.price_export_eur_kwh or export_price_eur_kwh)
                    for row in resolved_rows
                },
                realized_room_temp_by_timestamp={
                    row.timestamp_utc: float(row.room_temperature_c)
                    for row in resolved_rows
                    if row.room_temperature_c is not None
                },
                realized_pv_power_by_timestamp={
                    row.timestamp_utc: float(row.pv_output_power_kw or 0.0)
                    for row in resolved_rows
                },
                realized_base_load_power_by_timestamp={
                    row.timestamp_utc: self._row_base_load_power_kw(row) for row in resolved_rows
                },
                realized_solar_irradiance_by_timestamp={
                    row.timestamp_utc: float(row.solar_irradiance_w_m2 or 0.0)
                    for row in resolved_rows
                },
                solar_irradiance_by_timestamp={
                    row.timestamp_utc: float(row.solar_irradiance_w_m2 or 0.0)
                    for row in resolved_rows
                },
                realized_solar_gain_by_timestamp={
                    row.timestamp_utc: self.row_solar_gain_kw(row, source_model=source_model)
                    for row in resolved_rows
                },
            )
        )
        return horizon

    def initial_state_from_rows(
        self,
        rows: list[MpcDatasetRow],
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
    ) -> MpcInitialState | Rc2StateMpcInitialState:
        latest = rows[-1]
        if latest.room_temperature_c is None:
            raise ValueError(
                "Latest dataset row is missing room_temperature_c for MPC initial state"
            )
        latest_hp_on = self.row_hp_on(latest)
        contiguous_steps = 0
        for row in reversed(rows):
            if self.row_hp_on(row) != latest_hp_on:
                break
            contiguous_steps += 1
        base_state = MpcInitialState(
            room_temp_c=float(latest.room_temperature_c),
            q_heat_eff_kw=max(float(latest.space_heating_output_estimate_kw or 0.0), 0.0),
            hp_on=latest_hp_on,
            on_steps=contiguous_steps if latest_hp_on else 0,
            off_steps=contiguous_steps if not latest_hp_on else 0,
        )
        if not isinstance(source_model, RoomRcModel):
            return base_state
        filtered_room_temp_c, filtered_mass_temp_c = self._estimate_filtered_rc_state(
            rows,
            source_model=source_model,
        )
        return Rc2StateMpcInitialState(
            room_temp_c=filtered_room_temp_c,
            mass_temp_c=filtered_mass_temp_c,
            q_heat_eff_kw=base_state.q_heat_eff_kw,
            hp_on=base_state.hp_on,
            on_steps=base_state.on_steps,
            off_steps=base_state.off_steps,
        )

    def history_rows_for_initial_state(
        self,
        source_model: TrainedLinearRoomModel | RoomRcModel,
        *,
        minimum_history_rows: int,
    ) -> int:
        if isinstance(source_model, RoomRcModel):
            return max(
                minimum_history_rows,
                self.rc_trainer.max_history_rows(source_model.config),
            )
        return minimum_history_rows

    def _estimate_filtered_rc_state(
        self,
        rows: list[MpcDatasetRow],
        *,
        source_model: RoomRcModel,
    ) -> tuple[float, float]:
        try:
            room_temp_c, mass_temp_c = self.rc_trainer.estimate_current_state(source_model, rows)
            return float(room_temp_c), float(mass_temp_c)
        except (ValueError, IndexError, KeyError):
            params = RoomRC2StateParams.from_dict(source_model.params)
            latest_room_temp_c = float(rows[-1].room_temperature_c or 0.0)
            return latest_room_temp_c, latest_room_temp_c + params.initial_mass_offset_c

    def resolve_effective_heating_kw(
        self,
        rows: list[MpcDatasetRow],
        *,
        fallback_kw: float | None,
    ) -> float:
        heating_values = [
            float(row.space_heating_output_estimate_kw)
            for row in rows
            if row.space_heating_output_estimate_kw is not None
            and row.space_heating_output_estimate_kw > 0.0
        ]
        if heating_values:
            return float(sum(heating_values) / len(heating_values))
        if fallback_kw is not None:
            return float(fallback_kw)
        if rows and all(self._row_has_explicitly_no_space_heating(row) for row in rows):
            return 0.0
        raise ValueError(
            "Unable to infer effective heating kW; provide default_effective_heating_kw"
        )

    @staticmethod
    def resolve_hp_electric_power_kw(
        rows: list[MpcDatasetRow],
        *,
        fallback_kw: float,
    ) -> float:
        electric_power_values = [
            float(row.hp_electric_power_kw)
            for row in rows
            if row.hp_electric_power_kw is not None and row.hp_electric_power_kw > 0.0
        ]
        if electric_power_values:
            return float(sum(electric_power_values) / len(electric_power_values))
        return float(fallback_kw)

    @staticmethod
    def resolve_export_price_eur_kwh(rows: list[MpcDatasetRow]) -> float:
        export_price_values = [
            float(row.price_export_eur_kwh)
            for row in rows
            if row.price_export_eur_kwh is not None and row.price_export_eur_kwh >= 0.0
        ]
        if not export_price_values:
            return 0.0
        return float(export_price_values[-1])

    @staticmethod
    def resolve_base_load_power_kw(rows: list[MpcDatasetRow]) -> float:
        base_load_values = [
            (
                float(row.net_power_kw)
                + float(row.pv_output_power_kw or 0.0)
                - float(row.hp_electric_power_kw or 0.0)
            )
            for row in rows
            if row.net_power_kw is not None
        ]
        if not base_load_values:
            return 0.0
        return float(sum(base_load_values) / len(base_load_values))

    @staticmethod
    def infer_pv_power_input_scale(
        rows: list[MpcDatasetRow],
        *,
        forecast_entries: list[ForecastEntry],
        start_time_utc: datetime,
    ) -> float:
        if not rows:
            return 0.0
        latest_row = rows[-1]
        latest_pv_output_kw = float(latest_row.pv_output_power_kw or 0.0)
        if latest_pv_output_kw <= 0.0:
            return 0.0

        latest_created_by_forecast_time: dict[datetime, tuple[datetime, float]] = {}
        for entry in forecast_entries:
            if entry.name != GTI_PV:
                continue
            forecast_time_utc = ensure_utc(entry.forecast_time_utc)
            created_at_utc = ensure_utc(entry.created_at_utc)
            existing = latest_created_by_forecast_time.get(forecast_time_utc)
            if existing is None or created_at_utc >= existing[0]:
                latest_created_by_forecast_time[forecast_time_utc] = (
                    created_at_utc,
                    float(entry.value),
                )

        overlap_ratios = [
            float(row.pv_output_power_kw) / latest_created_by_forecast_time[row.timestamp_utc][1]
            for row in rows
            if row.pv_output_power_kw is not None
            and float(row.pv_output_power_kw) > 0.0
            and row.timestamp_utc in latest_created_by_forecast_time
            and latest_created_by_forecast_time[row.timestamp_utc][1] > 0.0
        ]
        if overlap_ratios:
            return float(median(overlap_ratios))

        start_forecast_value = latest_created_by_forecast_time.get(start_time_utc)
        if start_forecast_value is None or start_forecast_value[1] <= 0.0:
            return 0.0
        return latest_pv_output_kw / start_forecast_value[1]

    def row_solar_gain_proxy_w_m2(self, row: MpcDatasetRow) -> float:
        if row.solar_gain_proxy_w_m2 is not None:
            return float(row.solar_gain_proxy_w_m2)
        irradiance_w_m2 = float(row.solar_irradiance_w_m2 or 0.0)
        shutter_fraction = 1.0
        if row.shutter_position_pct is not None:
            shutter_fraction = min(max(float(row.shutter_position_pct) / 100.0, 0.0), 1.0)
        return float(irradiance_w_m2 * shutter_fraction)

    def row_solar_gain_kw(
        self,
        row: MpcDatasetRow,
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
    ) -> float:
        solar_gain_proxy_w_m2 = self.row_solar_gain_proxy_w_m2(row)
        if isinstance(source_model, RoomRcModel):
            return float(
                solar_gain_proxy_w_m2
                * source_model.config.glass_area_m2
                * source_model.config.g_glass
                / 1000.0
            )
        return float(solar_gain_proxy_w_m2)

    @staticmethod
    def row_hp_on(row: MpcDatasetRow) -> bool:
        heating_output_kw = float(row.space_heating_output_estimate_kw or 0.0)
        return bool(row.mode_space) and not bool(row.mode_off) and heating_output_kw > 0.0

    @staticmethod
    def _row_has_explicitly_no_space_heating(row: MpcDatasetRow) -> bool:
        if bool(row.mode_off):
            return True
        if bool(row.mode_space):
            return row.space_heating_output_estimate_kw is not None and float(
                row.space_heating_output_estimate_kw
            ) <= 0.0
        return row.space_heating_output_estimate_kw is not None and float(
            row.space_heating_output_estimate_kw
        ) <= 0.0

    @staticmethod
    def _interval_minutes_from_rows(rows: list[MpcDatasetRow]) -> int:
        if len(rows) < 2:
            raise ValueError("At least two dataset rows are required to infer interval_minutes")
        delta_minutes = (
            ensure_utc(rows[1].timestamp_utc) - ensure_utc(rows[0].timestamp_utc)
        ).total_seconds() / 60.0
        if delta_minutes <= 0:
            raise ValueError("Dataset rows must be sorted ascending with unique timestamps")
        return int(delta_minutes)

    @staticmethod
    def _row_base_load_power_kw(row: MpcDatasetRow) -> float:
        return float(row.net_power_kw or 0.0) + float(row.pv_output_power_kw or 0.0) - float(
            row.hp_electric_power_kw or 0.0
        )

    def _forecast_entries_from_rows(
        self,
        rows: list[MpcDatasetRow],
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
    ) -> list[ForecastEntry]:
        entries: list[ForecastEntry] = []
        for row in rows:
            timestamp_utc = ensure_utc(row.timestamp_utc)
            entries.extend(
                [
                    ForecastEntry(
                        created_at_utc=timestamp_utc,
                        forecast_time_utc=timestamp_utc,
                        name=FORECAST_TEMPERATURE,
                        value=float(row.outdoor_temperature_c or 0.0),
                        unit="C",
                        source="historical_dataset",
                    ),
                    ForecastEntry(
                        created_at_utc=timestamp_utc,
                        forecast_time_utc=timestamp_utc,
                        name=GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
                        value=self.row_solar_gain_proxy_w_m2(row),
                        unit="W/m2",
                        source="historical_dataset",
                    ),
                    ForecastEntry(
                        created_at_utc=timestamp_utc,
                        forecast_time_utc=timestamp_utc,
                        name=GTI_PV,
                        value=float(row.pv_output_power_kw or 0.0),
                        unit="kW",
                        source="historical_dataset",
                    ),
                ]
            )
        return entries

    @staticmethod
    def _price_intervals_from_rows(rows: list[MpcDatasetRow]) -> list[PriceInterval]:
        interval_minutes = SpaceHeatingMpcPreparationService._interval_minutes_from_rows(rows)
        return [
            PriceInterval(
                start_time_utc=row.timestamp_utc,
                end_time_utc=row.timestamp_utc + timedelta(minutes=interval_minutes),
                source="historical_dataset",
                name="electricity_price",
                unit="EUR/kWh",
                value=float(row.price_import_eur_kwh or 0.0),
            )
            for row in rows
        ]

    @staticmethod
    def _solar_gain_input_scale(source_model: TrainedLinearRoomModel | RoomRcModel) -> float:
        if isinstance(source_model, RoomRcModel):
            return float(source_model.config.glass_area_m2 * source_model.config.g_glass / 1000.0)
        return 1.0

    @staticmethod
    def _solar_gain_filter_alpha(source_model: TrainedLinearRoomModel | RoomRcModel) -> float:
        if isinstance(source_model, RoomRcModel):
            return float(source_model.config.alpha_solar)
        return 0.0

    @staticmethod
    def _local_timezone(source_model: TrainedLinearRoomModel | RoomRcModel) -> str | None:
        if isinstance(source_model, RoomRcModel):
            return source_model.config.local_timezone
        return None

    def _initial_filtered_solar_gain_kw(
        self,
        rows: list[MpcDatasetRow],
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
        start_time_utc: datetime,
    ) -> float:
        if not isinstance(source_model, RoomRcModel):
            return 0.0
        solar_gain_history_kw = [
            self.row_solar_gain_kw(row, source_model=source_model)
            for row in rows
            if row.timestamp_utc < start_time_utc
        ]
        return trailing_exp_filter(
            solar_gain_history_kw,
            alpha=source_model.config.alpha_solar,
        )


class _MpcDatasetSettings:
    def __init__(
        self,
        target_schedule: list[TemperatureTargetWindow],
        *,
        electricity_pricing: ElectricityPricingConfig,
    ) -> None:
        self.room_target = list(target_schedule)
        self.dhw_target: list[TemperatureTargetWindow] = []
        self.electricity_pricing = electricity_pricing


class _DynamicPricingLike:
    mode = "dynamic"
