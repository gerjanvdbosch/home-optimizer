"""Home Optimizer – UFH + DHW thermal model with Kalman filter and combined MPC.

Physical model assumptions
--------------------------
* UFH (§3–§6): 2-state grey-box model [T_r, T_b]; heat transport via Newton's
  law; solar irradiance split by fraction α; forward-Euler discretisation.
* DHW (§7–§12): 2-node stratification tank [T_top, T_bot]; time-varying LTV
  state-space (depends on V_tap[k]); λ = 1.1628 kWh/(m³·K); tap-stream split
  correctly between layers.
* Combined MPC (§13–§14): block-diagonal state-space; shared heat-pump power
  budget; legionella constraint as stiff soft constraint.
* All energy quantities use kW (power) and kWh (energy); temperatures in °C.
* Discretisation: forward-Euler with time step dt_hours.
"""

from .combined_mpc import CombinedMPCController, CombinedMPCSolution
from .dhw_model import MEASUREMENT_MATRIX_DHW, DHWModel
from .kalman import DHWKalmanFilter, KalmanEstimate, UFHKalmanFilter
from .mpc import MPCSolution, UFHMPCController
from .thermal_model import MEASUREMENT_MATRIX, ThermalModel, solar_gain_kw
from .types import (
    CombinedMPCParameters,
    DHWForecastHorizon,
    DHWMPCParameters,
    DHWParameters,
    ForecastHorizon,
    GreedySolverConfig,
    KalmanNoiseParameters,
    MPCParameters,
    ThermalParameters,
    W_PER_KW,
)

__all__ = [
    "CombinedMPCController",
    "CombinedMPCParameters",
    "CombinedMPCSolution",
    "DHWForecastHorizon",
    "DHWKalmanFilter",
    "DHWMPCParameters",
    "DHWModel",
    "DHWParameters",
    "ForecastHorizon",
    "GreedySolverConfig",
    "KalmanEstimate",
    "KalmanNoiseParameters",
    "MEASUREMENT_MATRIX",
    "MEASUREMENT_MATRIX_DHW",
    "MPCParameters",
    "MPCSolution",
    "ThermalModel",
    "ThermalParameters",
    "UFHKalmanFilter",
    "UFHMPCController",
    "W_PER_KW",
    "solar_gain_kw",
]
