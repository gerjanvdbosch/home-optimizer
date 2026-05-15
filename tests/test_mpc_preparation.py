from __future__ import annotations

from datetime import datetime, timezone

import pytest

from home_optimizer.features.dataset.models import MpcDatasetRow
from home_optimizer.features.modeling import RoomRcConfig, RoomRcModel
from home_optimizer.features.mpc import Rc2StateMpcInitialState
from home_optimizer.features.mpc.preparation import SpaceHeatingMpcPreparationService


class _UnusedSamplesReader:
    pass


class _UnusedModelReader:
    def get_room_model_version(self, model_id: str):
        return None

    def get_active_room_model_version(self):
        return None


class _UnusedRcTrainer:
    def estimate_current_state(self, model: RoomRcModel, rows: list[MpcDatasetRow]) -> tuple[float, float]:
        raise ValueError("not configured for this test")

    def max_history_rows(self, config: RoomRcConfig) -> int:
        return 36


def _service(
    *,
    rc_trainer: _UnusedRcTrainer | None = None,
) -> SpaceHeatingMpcPreparationService:
    return SpaceHeatingMpcPreparationService(
        samples_reader=_UnusedSamplesReader(),
        active_room_model_reader=_UnusedModelReader(),
        target_schedule=[],
        rc_trainer=rc_trainer or _UnusedRcTrainer(),
    )


class _FilteredStateRcTrainer(_UnusedRcTrainer):
    def estimate_current_state(self, model: RoomRcModel, rows: list[MpcDatasetRow]) -> tuple[float, float]:
        return 19.3, 22.1


def test_initial_state_from_rows_uses_filtered_mass_state_for_2r2c() -> None:
    latest_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    model = RoomRcModel(
        trained_from_utc=latest_time,
        trained_to_utc=latest_time,
        interval_minutes=10,
        config=RoomRcConfig(),
        params={"initial_mass_offset_c": 5.0},
        sample_count=10,
    )
    rows = [
        MpcDatasetRow(
            timestamp_utc=latest_time,
            room_temperature_c=19.5,
            mode_space=0,
            mode_off=1,
            space_heating_output_estimate_kw=0.0,
        )
    ]

    initial_state = _service(
        rc_trainer=_FilteredStateRcTrainer()
    ).initial_state_from_rows(rows, source_model=model)

    assert isinstance(initial_state, Rc2StateMpcInitialState)
    assert initial_state.room_temp_c == 19.5
    assert initial_state.mass_temp_c == 22.1


def test_initial_state_from_rows_falls_back_to_model_mass_offset_when_filter_fails() -> None:
    latest_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    model = RoomRcModel(
        trained_from_utc=latest_time,
        trained_to_utc=latest_time,
        interval_minutes=10,
        config=RoomRcConfig(),
        params={"initial_mass_offset_c": 5.0},
        sample_count=10,
    )
    rows = [
        MpcDatasetRow(
            timestamp_utc=latest_time,
            room_temperature_c=19.5,
            mode_space=0,
            mode_off=1,
            space_heating_output_estimate_kw=0.0,
        )
    ]

    initial_state = _service().initial_state_from_rows(rows, source_model=model)

    assert isinstance(initial_state, Rc2StateMpcInitialState)
    assert initial_state.mass_temp_c == 24.5


def test_history_rows_for_initial_state_keeps_requested_context() -> None:
    model = RoomRcModel(
        trained_from_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomRcConfig(),
        params={},
        sample_count=10,
    )

    history_rows = _service().history_rows_for_initial_state(
        model,
        minimum_history_rows=3,
    )

    assert history_rows == 36


def test_resolve_effective_heating_kw_returns_zero_when_heating_is_explicitly_off() -> None:
    rows = [
        MpcDatasetRow(
            timestamp_utc=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            mode_space=0,
            mode_off=1,
            space_heating_output_estimate_kw=0.0,
        ),
        MpcDatasetRow(
            timestamp_utc=datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc),
            mode_space=0,
            mode_off=1,
            space_heating_output_estimate_kw=0.0,
        ),
    ]

    effective_heating_kw = _service().resolve_effective_heating_kw(
        rows,
        fallback_kw=None,
    )

    assert effective_heating_kw == 0.0


def test_resolve_effective_heating_kw_still_fails_when_heating_state_is_unknown() -> None:
    rows = [
        MpcDatasetRow(
            timestamp_utc=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            mode_space=1,
            mode_off=0,
            space_heating_output_estimate_kw=None,
        ),
    ]

    with pytest.raises(ValueError, match="Unable to infer effective heating kW"):
        _service().resolve_effective_heating_kw(
            rows,
            fallback_kw=None,
        )
