"""Home Optimizer package for UFH thermal modeling, estimation, and control."""

from .kalman import KalmanEstimate, UFHKalmanFilter
from .mpc import MPCSolution, UFHMPCController
from .thermal_model import MEASUREMENT_MATRIX, ThermalModel, solar_gain_kw
from .types import ForecastHorizon, KalmanNoiseParameters, MPCParameters, ThermalParameters

__all__ = [
    "ForecastHorizon",
    "KalmanEstimate",
    "KalmanNoiseParameters",
    "MEASUREMENT_MATRIX",
    "MPCParameters",
    "MPCSolution",
    "ThermalModel",
    "ThermalParameters",
    "UFHKalmanFilter",
    "UFHMPCController",
    "solar_gain_kw",
]
