from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.features.dataset.models import MpcDatasetRow
from home_optimizer.features.modeling.models import StoredModelVersion
from home_optimizer.features.modeling.room_2r2c import RoomRC2StateParams, RoomRcConfig, RoomRcModel
from home_optimizer.features.modeling.room_arx import RoomArxConfig, RoomArxModel
from home_optimizer.features.mpc import (
    MpcInitialState,
    MpcPlan,
    Rc2StateMpcInitialState,
    SpaceHeatingMpcPlanningService,
)


class _UnusedSamplesReader:
    pass


class _StaticActiveRoomModelReader:
    def __init__(self, version: StoredModelVersion) -> None:
        self.version = version

    def get_room_model_version(self, model_id: str) -> StoredModelVersion | None:
        if self.version.model_id == model_id:
            return self.version
        return None

    def get_active_room_model_version(self) -> StoredModelVersion | None:
        return self.version


def test_space_heating_mpc_planning_service_builds_plan_from_active_model(monkeypatch) -> None:
    start_time = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    source_model = RoomArxModel(
        trained_from_utc=start_time - timedelta(days=1),
        trained_to_utc=start_time,
        interval_minutes=10,
        config=RoomArxConfig(
            room_temperature_lags=[0],
            outdoor_temperature_lags=[0],
            thermal_output_lags=[0],
            solar_gain_lags=[0],
            occupied_flag_lags=[0],
            shutter_position_lags=[0],
            solar_shutter_interaction_lags=[0],
        ),
        feature_names=[
            "room_temperature_lag_0",
            "outdoor_temperature_lag_0",
            "thermal_output_lag_0",
            "solar_gain_lag_0",
            "occupied_flag_lag_0",
        ],
        intercept=0.0,
        coefficients=[0.95, 0.03, 0.4, 0.0, 0.02],
        sample_count=100,
    )
    version = StoredModelVersion(
        model_id="active-room-model",
        model_type="room_arx",
        created_at_utc=start_time,
        is_active=True,
        model=source_model,
    )
    service = SpaceHeatingMpcPlanningService(
        samples_reader=_UnusedSamplesReader(),
        active_room_model_reader=_StaticActiveRoomModelReader(version),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
    )

    monkeypatch.setattr(
        service,
        "_load_initial_rows",
        lambda **kwargs: [
            MpcDatasetRow(
                timestamp_utc=start_time - timedelta(minutes=20),
                room_temperature_c=19.0,
                mode_space=0,
                mode_off=1,
                space_heating_output_estimate_kw=0.0,
            ),
            MpcDatasetRow(
                timestamp_utc=start_time - timedelta(minutes=10),
                room_temperature_c=19.2,
                mode_space=0,
                mode_off=1,
                space_heating_output_estimate_kw=0.0,
            ),
            MpcDatasetRow(
                timestamp_utc=start_time,
                room_temperature_c=19.4,
                mode_space=0,
                mode_off=1,
                space_heating_output_estimate_kw=0.0,
            ),
        ],
    )
    monkeypatch.setattr(
        service,
        "_load_forecast_entries",
        lambda **kwargs: [
            ForecastEntry(
                created_at_utc=start_time - timedelta(hours=1),
                forecast_time_utc=start_time + timedelta(minutes=10 * step),
                name="temperature",
                value=4.0,
                unit="C",
                source="test",
            )
            for step in range(6)
        ],
    )
    monkeypatch.setattr(
        service,
        "_load_price_intervals",
        lambda **kwargs: [
            PriceInterval(
                start_time_utc=start_time,
                end_time_utc=start_time + timedelta(hours=1),
                source="test",
                value=0.25,
            )
        ],
    )

    plan = service.plan(
        start_time_utc=start_time,
        horizon_steps=6,
        default_effective_heating_kw=2.0,
    )

    assert plan.feasible is True
    assert len(plan.steps) == 6
    assert plan.steps[0].timestamp_utc == start_time
    assert plan.steps[1].timestamp_utc == start_time + timedelta(minutes=10)


def test_space_heating_mpc_planning_service_uses_default_interval_setting(monkeypatch) -> None:
    start_time = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    source_model = RoomArxModel(
        trained_from_utc=start_time - timedelta(days=1),
        trained_to_utc=start_time,
        interval_minutes=15,
        config=RoomArxConfig(
            room_temperature_lags=[0],
            outdoor_temperature_lags=[0],
            thermal_output_lags=[0],
            solar_gain_lags=[0],
            occupied_flag_lags=[0],
            shutter_position_lags=[0],
            solar_shutter_interaction_lags=[0],
        ),
        feature_names=[
            "room_temperature_lag_0",
            "outdoor_temperature_lag_0",
            "thermal_output_lag_0",
            "solar_gain_lag_0",
            "occupied_flag_lag_0",
        ],
        intercept=0.0,
        coefficients=[0.95, 0.03, 0.4, 0.0, 0.02],
        sample_count=100,
    )
    version = StoredModelVersion(
        model_id="active-room-model",
        model_type="room_arx",
        created_at_utc=start_time,
        is_active=True,
        model=source_model,
    )
    service = SpaceHeatingMpcPlanningService(
        samples_reader=_UnusedSamplesReader(),
        active_room_model_reader=_StaticActiveRoomModelReader(version),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
        default_interval_minutes=10,
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(service, "_load_initial_rows", lambda **kwargs: [])
    monkeypatch.setattr(service, "_initial_state_from_rows", lambda *args, **kwargs: MpcInitialState(room_temp_c=19.0))
    monkeypatch.setattr(service, "_resolve_effective_heating_kw", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(service, "_resolve_hp_electric_power_kw", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(service, "_resolve_export_price_eur_kwh", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(service, "_resolve_base_load_power_kw", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(service, "_load_forecast_entries", lambda **kwargs: captured.update({"forecast_interval": kwargs["interval_minutes"]}) or [])
    monkeypatch.setattr(service, "_load_price_intervals", lambda **kwargs: [])
    monkeypatch.setattr(service, "_infer_pv_power_input_scale", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(service.controller, "build_horizon", lambda request: captured.update({"request_interval": request.interval_minutes}) or [])
    monkeypatch.setattr(service.controller, "plan_from_source_model", lambda request, **kwargs: captured.update({"controller_interval": request.interval_minutes}) or MpcPlan(status="ok", termination_condition="optimal", feasible=True, steps=[]))

    service.plan(start_time_utc=start_time, horizon_steps=2, default_effective_heating_kw=2.0)

    assert captured["forecast_interval"] == 10
    assert captured["request_interval"] == 10
    assert captured["controller_interval"] == 10


def test_space_heating_mpc_planning_service_uses_2state_initial_state_for_room_rc(
    monkeypatch,
) -> None:
    start_time = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    source_model = RoomRcModel(
        trained_from_utc=start_time - timedelta(days=1),
        trained_to_utc=start_time,
        interval_minutes=10,
        config=RoomRcConfig(),
        params=RoomRC2StateParams(initial_mass_offset_c=1.5).to_dict(),
        sample_count=100,
    )
    version = StoredModelVersion(
        model_id="active-room-rc-model",
        model_type="room_2r2c",
        created_at_utc=start_time,
        is_active=True,
        model=source_model,
    )
    service = SpaceHeatingMpcPlanningService(
        samples_reader=_UnusedSamplesReader(),
        active_room_model_reader=_StaticActiveRoomModelReader(version),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
    )

    monkeypatch.setattr(
        service,
        "_load_initial_rows",
        lambda **kwargs: [
            MpcDatasetRow(
                timestamp_utc=start_time - timedelta(minutes=10),
                room_temperature_c=19.0,
                mode_space=0,
                mode_off=1,
                space_heating_output_estimate_kw=0.0,
            ),
            MpcDatasetRow(
                timestamp_utc=start_time,
                room_temperature_c=19.4,
                mode_space=0,
                mode_off=1,
                space_heating_output_estimate_kw=0.0,
            ),
        ],
    )
    monkeypatch.setattr(service, "_load_forecast_entries", lambda **kwargs: [])
    monkeypatch.setattr(service, "_load_price_intervals", lambda **kwargs: [])

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        service.controller,
        "plan_from_source_model",
        lambda request, **kwargs: captured.update(
            {"request": request, **kwargs}
        )
        or MpcPlan(
            status="ok",
            termination_condition="optimal",
            feasible=True,
            steps=[],
        ),
    )

    plan = service.plan(
        start_time_utc=start_time,
        horizon_steps=4,
        default_effective_heating_kw=2.0,
    )

    assert plan.feasible is True
    assert isinstance(captured["initial_state"], Rc2StateMpcInitialState)
    assert captured["initial_state"].room_temp_c == 19.4
    assert captured["initial_state"].mass_temp_c == 20.9
