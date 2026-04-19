"""Shared physical constants and unit-conversion helpers for Home Optimizer."""

from __future__ import annotations

#: 1 kW = 1000 W — used to convert W/m² irradiance to kW.
W_PER_KW: float = 1000.0

#: Absolute zero expressed in degrees Celsius. Temperatures below this are non-physical.
ABSOLUTE_ZERO_C: float = -273.15

#: Water volumetric heat capacity λ = ρ·c_p [kWh/(m³·K)] (§8.4, §15).
LAMBDA_WATER_KWH_PER_M3_K: float = 1.1628

#: Number of litres in one cubic metre [L/m³].
LITERS_PER_CUBIC_METER: float = 1000.0

#: Cubic metres in one litre [m³/L].
M3_PER_LITER: float = 1.0 / LITERS_PER_CUBIC_METER

__all__ = [
    "ABSOLUTE_ZERO_C",
    "LAMBDA_WATER_KWH_PER_M3_K",
    "LITERS_PER_CUBIC_METER",
    "M3_PER_LITER",
    "W_PER_KW",
]

