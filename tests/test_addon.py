"""Tests for addon configuration projection into the canonical runtime request."""

from __future__ import annotations

import pytest

from home_optimizer.addon import AddonOptions, _build_runtime_base_request


def _valid_addon_options(**overrides: object) -> AddonOptions:
    """Build a minimally valid addon options object for projection tests."""
    data: dict[str, object] = {
        "database_path": "/tmp/home-optimizer-test.sqlite3",
        "models_path": "/tmp/home-optimizer-models",
        "sensor_room_temperature": "sensor.room_temperature",
        "sensor_outdoor_temperature": "sensor.outdoor_temperature",
        "sensor_hp_supply_temperature": "sensor.hp_supply_temperature",
        "sensor_hp_supply_target_temperature": "sensor.hp_supply_target_temperature",
        "sensor_hp_return_temperature": "sensor.hp_return_temperature",
        "sensor_thermostat_setpoint": "sensor.thermostat_setpoint",
        "sensor_dhw_top_temperature": "sensor.dhw_top_temperature",
        "sensor_dhw_bottom_temperature": "sensor.dhw_bottom_temperature",
        "sensor_boiler_ambient_temperature": "sensor.boiler_ambient_temperature",
        "sensor_refrigerant_condensation_temperature": "sensor.refrigerant_condensation_temperature",
        "sensor_refrigerant_liquid_line_temperature": "sensor.refrigerant_liquid_line_temperature",
        "sensor_discharge_temperature": "sensor.discharge_temperature",
        "sensor_hp_flow_lpm": "sensor.hp_flow_lpm",
        "sensor_hp_electric_power": "sensor.hp_electric_power",
        "sensor_p1_net_power": "sensor.p1_net_power",
        "sensor_pv_output": "sensor.pv_output",
        "sensor_hp_mode": "sensor.hp_mode",
        "sensor_shutter_living_room": "sensor.shutter_living_room",
        "sensor_defrost_active": "binary_sensor.defrost_active",
        "sensor_booster_heater_active": "binary_sensor.booster_heater_active",
        "sensor_pv_total_kwh": "sensor.pv_total_kwh",
        "sensor_hp_electric_total_kwh": "sensor.hp_electric_total_kwh",
        "sensor_p1_import_total_kwh": "sensor.p1_import_total_kwh",
        "sensor_p1_export_total_kwh": "sensor.p1_export_total_kwh",
    }
    data.update(overrides)
    return AddonOptions.model_validate(data)


def test_build_runtime_base_request_projects_split_dhw_losses_and_exclusive_mode() -> None:
    """Addon runtime projection must expose split DHW losses and exclusive topology fields."""
    options = _valid_addon_options(
        mpc_dhw_R_loss_top=77.0,
        mpc_dhw_R_loss_bot=88.0,
        mpc_heat_pump_topology="exclusive",
        mpc_exclusive_heat_pump_mode="dhw",
    )

    request = _build_runtime_base_request(options)

    assert request.dhw_R_loss_top == 77.0
    assert request.dhw_R_loss_bot == 88.0
    assert request.heat_pump_topology == "exclusive"
    assert request.exclusive_heat_pump_mode == "dhw"


def test_addon_options_reject_exclusive_mode_when_topology_is_shared() -> None:
    """Shared topology must not carry an exclusive active mode in addon config."""
    with pytest.raises(ValueError, match="mpc_exclusive_heat_pump_mode"):
        _valid_addon_options(
            mpc_heat_pump_topology="shared",
            mpc_exclusive_heat_pump_mode="ufh",
        )
