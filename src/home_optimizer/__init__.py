"""Home Optimizer – 2-state UFH thermal model with Kalman filter and MPC.

Physical model assumptions
--------------------------
* Heat transport between zones follows Newton's law of cooling (linear).
* Solar irradiance is split by fraction α directly to room air and (1-α) to the floor slab.
* All energy quantities use kW (power) and kWh (energy); temperatures in °C.
* Discretisation: forward-Euler with time step dt_hours.  Stability requires
  dt << min(C_r·R_br, C_b·R_br, C_r·R_ro).
"""

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
