from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain import GTI_PV
from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.pricing import ElectricityPricingConfig
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
from home_optimizer.features.mpc.models import MpcInitialState, Rc2StateMpcInitialState
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

        start_forecast_value = latest_created_by_forecast_time.get(start_time_utc)
        if start_forecast_value is None or start_forecast_value[1] <= 0.0:
            return 0.0
        return latest_pv_output_kw / start_forecast_value[1]

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
