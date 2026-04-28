from __future__ import annotations

ROOM_TEMPERATURE = "room_temperature"
OUTDOOR_TEMPERATURE = "outdoor_temperature"
THERMOSTAT_SETPOINT = "thermostat_setpoint"
SHUTTER_LIVING_ROOM = "shutter_living_room"

HP_SUPPLY_TEMPERATURE = "hp_supply_temperature"
HP_SUPPLY_TARGET_TEMPERATURE = "hp_supply_target_temperature"
HP_RETURN_TEMPERATURE = "hp_return_temperature"
HP_FLOW = "hp_flow"
HP_MODE = "hp_mode"
COMPRESSOR_FREQUENCY = "compressor_frequency"
DEFROST_ACTIVE = "defrost_active"
BOOSTER_HEATER_ACTIVE = "booster_heater_active"

REFRIGERANT_CONDENSATION_TEMPERATURE = "refrigerant_condensation_temperature"
REFRIGERANT_LIQUID_LINE_TEMPERATURE = "refrigerant_liquid_line_temperature"
DISCHARGE_TEMPERATURE = "discharge_temperature"

DHW_TOP_TEMPERATURE = "dhw_top_temperature"
DHW_BOTTOM_TEMPERATURE = "dhw_bottom_temperature"
BOILER_AMBIENT_TEMPERATURE = "boiler_ambient_temperature"

HP_ELECTRIC_POWER = "hp_electric_power"
P1_NET_POWER = "p1_net_power"
PV_OUTPUT_POWER = "pv_output_power"
PV_TOTAL_KWH = "pv_total_kwh"
HP_ELECTRIC_TOTAL_KWH = "hp_electric_total_kwh"
P1_IMPORT_TOTAL_KWH = "p1_import_total_kwh"
P1_EXPORT_TOTAL_KWH = "p1_export_total_kwh"

FORECAST_TEMPERATURE = "temperature"
FORECAST_HUMIDITY = "humidity"
FORECAST_WIND = "wind"
FORECAST_DEW_POINT = "dew_point"
FORECAST_DIRECT_RADIATION = "direct_radiation"
FORECAST_DIFFUSE_RADIATION = "diffuse_radiation"
GTI_PV = "gti_pv"
GTI_LIVING_ROOM_WINDOWS = "gti_living_room_windows"
GTI_LIVING_ROOM_WINDOWS_ADJUSTED = "gti_living_room_windows_adjusted"

THERMAL_OUTPUT = "thermal_output"
COP = "cop"
BASELOAD = "baseload"
HP_DELTA_T = "hp_delta_t"

__all__ = [
    "BASELOAD",
    "BOILER_AMBIENT_TEMPERATURE",
    "BOOSTER_HEATER_ACTIVE",
    "COMPRESSOR_FREQUENCY",
    "COP",
    "DEFROST_ACTIVE",
    "DHW_BOTTOM_TEMPERATURE",
    "DHW_TOP_TEMPERATURE",
    "DISCHARGE_TEMPERATURE",
    "FORECAST_DEW_POINT",
    "FORECAST_DIFFUSE_RADIATION",
    "FORECAST_DIRECT_RADIATION",
    "FORECAST_HUMIDITY",
    "FORECAST_TEMPERATURE",
    "FORECAST_WIND",
    "GTI_LIVING_ROOM_WINDOWS",
    "GTI_LIVING_ROOM_WINDOWS_ADJUSTED",
    "GTI_PV",
    "HP_DELTA_T",
    "HP_ELECTRIC_POWER",
    "HP_ELECTRIC_TOTAL_KWH",
    "HP_FLOW",
    "HP_MODE",
    "HP_RETURN_TEMPERATURE",
    "HP_SUPPLY_TARGET_TEMPERATURE",
    "HP_SUPPLY_TEMPERATURE",
    "OUTDOOR_TEMPERATURE",
    "P1_EXPORT_TOTAL_KWH",
    "P1_IMPORT_TOTAL_KWH",
    "P1_NET_POWER",
    "PV_OUTPUT_POWER",
    "PV_TOTAL_KWH",
    "REFRIGERANT_CONDENSATION_TEMPERATURE",
    "REFRIGERANT_LIQUID_LINE_TEMPERATURE",
    "ROOM_TEMPERATURE",
    "SHUTTER_LIVING_ROOM",
    "THERMAL_OUTPUT",
    "THERMOSTAT_SETPOINT",
]
