"""Carnot-based heat pump COP model with heating curve (stooklijn).

Theory
------
The Coefficient of Performance of a heat pump in **heating** mode is bounded
from above by the Carnot (ideal) COP:

    COP_Carnot = T_cond_K / (T_cond_K - T_evap_K)

A real machine operates below this limit by a Carnot efficiency factor η:

    COP_actual = η · COP_Carnot,   with η ∈ (0, 1]

The condensing temperature exceeds the supply temperature by an approach
temperature Δ_cond [K] (heat exchange at the condenser), and the evaporating
temperature is below the outdoor temperature by Δ_evap [K] (heat extraction
at the evaporator):

    T_cond [°C] = T_supply + Δ_cond
    T_evap [°C] = T_outdoor - Δ_evap

Converting to Kelvin (via the named constant T_CELSIUS_TO_KELVIN):

    T_cond_K = T_cond + T_CELSIUS_TO_KELVIN
    T_evap_K = T_evap + T_CELSIUS_TO_KELVIN

For **UFH**, the supply temperature follows a *heating curve* (stooklijn):

    T_supply(T_out) = T_supply_min + slope · max(T_ref_outdoor - T_out, 0)

This captures the real control behaviour: colder outdoor air requires warmer
floor water (higher T_cond), AND simultaneously results in a colder evaporator
(lower T_evap).  Both effects simultaneously reduce the COP, as physically
expected.

For **DHW**, the supply temperature is approximately the hot-water target
temperature (T_dhw_min for normal operation, T_legionella during a legionella
cycle).  Only the outdoor temperature (evaporator side) varies over the
horizon.

Units
-----
All temperatures : °C for inputs; K used only internally
COP              : dimensionless
delta temperatures : K (equal in size to °C differences)

Usage example
-------------
>>> from home_optimizer.cop_model import HeatPumpCOPModel, HeatPumpCOPParameters
>>> import numpy as np
>>> cop_params = HeatPumpCOPParameters(
...     eta_carnot=0.45,
...     delta_T_cond=5.0,
...     delta_T_evap=5.0,
...     T_supply_min=25.0,
...     T_ref_outdoor=18.0,
...     heating_curve_slope=1.0,
...     cop_min=1.5,
...     cop_max=7.0,
... )
>>> model = HeatPumpCOPModel(cop_params)
>>> t_out = np.array([10.0, 5.0, 0.0, -5.0, -10.0])
>>> model.cop_ufh(t_out)   # UFH COP drops with colder weather (double effect)
array([...])
>>> model.cop_dhw(t_out, t_dhw_supply=55.0)  # DHW COP only from evaporator side
array([...])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants — named to comply with the no-magic-numbers requirement
# ---------------------------------------------------------------------------

#: 0 °C expressed in Kelvin.  All Carnot computations use absolute temperature.
T_CELSIUS_TO_KELVIN: float = 273.15

#: Floor on the temperature lift (T_cond_K − T_evap_K) to prevent division by
#: zero when condenser and evaporator temperatures are nearly equal.  This is a
#: numerical guard, not a physical bound; set at 1 milli-Kelvin.
_MIN_TEMP_LIFT_K: float = 1e-3


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeatPumpCOPParameters:
    """Physical parameters for the Carnot-based heat pump COP model.

    Parameters
    ----------
    eta_carnot:
        Carnot efficiency factor η_carnot [dimensionless].  Relates actual COP
        to the theoretical Carnot maximum.  Typical air-source HP: 0.35–0.55.
        Must be in (0, 1].
    delta_T_cond:
        Condensing approach temperature Δ_cond [K].  The refrigerant condenses
        at T_supply + Δ_cond.  Accounts for heat-exchanger imperfection.
        Typical: 2–8 K.  Must be ≥ 0.
    delta_T_evap:
        Evaporating approach temperature Δ_evap [K].  The refrigerant evaporates
        at T_outdoor − Δ_evap.  Typical: 2–8 K.  Must be ≥ 0.
    T_supply_min:
        Minimum UFH supply temperature [°C].  The heating curve output is
        clamped to this floor, i.e., the HP never heats the floor below this
        temperature.  Typically the minimum design supply temperature (e.g.,
        25–30 °C for low-temperature UFH).
    T_ref_outdoor:
        Reference outdoor temperature [°C] at which the heating curve equals
        T_supply_min (the balance point / switchover temperature).  Typically
        15–18 °C for a well-insulated dwelling.
    heating_curve_slope:
        Heating-curve slope [K/K].  Defines how many degrees the supply
        temperature increases per degree the outdoor temperature drops below
        T_ref_outdoor.  Typical: 0.5–1.5 for UFH systems.  Must be ≥ 0.
        A slope of 0 means constant supply temperature (no heating curve).
    cop_min:
        Lower bound on the computed COP [dimensionless].  Acts as a physical
        floor: even in extreme cold the HP must maintain a minimum COP.
        Prevents numerically negative or trivially low COP values.
        Must be > 1 (heat pump, not resistive heater).
    cop_max:
        Upper bound on the computed COP [dimensionless].  Guards against
        unrealistically high COP values that would indicate model or sensor
        errors.  Must be > cop_min.
    """

    eta_carnot: float
    delta_T_cond: float
    delta_T_evap: float
    T_supply_min: float
    T_ref_outdoor: float
    heating_curve_slope: float
    cop_min: float
    cop_max: float

    def __post_init__(self) -> None:
        if not 0.0 < self.eta_carnot <= 1.0:
            raise ValueError("eta_carnot must be in (0, 1].")
        for field_name in ("delta_T_cond", "delta_T_evap"):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be ≥ 0.")
        if self.heating_curve_slope < 0.0:
            raise ValueError("heating_curve_slope must be ≥ 0.")
        if self.cop_min <= 1.0:
            raise ValueError("cop_min must be > 1 (heat pump, not resistive heater).")
        if self.cop_max <= self.cop_min:
            raise ValueError("cop_max must be strictly greater than cop_min.")


# ---------------------------------------------------------------------------
# COP model
# ---------------------------------------------------------------------------


class HeatPumpCOPModel:
    """Carnot-based COP model for heat pumps.

    Translates forecast arrays (outdoor temperature, supply temperature) into
    time-varying COP arrays that can be passed to the MPC via
    ``ForecastHorizon.cop_ufh_k`` and ``DHWForecastHorizon.cop_dhw_k``.

    Parameters
    ----------
    params:
        Physical parameters of the COP model.

    Notes
    -----
    The MPC decision variables remain **thermal** power [kW].  The COP arrays
    computed here are used exclusively in the cost function and the shared
    electrical power constraint (§14.1):

        P_elec = P_thermal / COP(k)

    This class has no side effects and is fully stateless beyond the stored
    parameters — all methods are pure functions of their inputs.
    """

    def __init__(self, params: HeatPumpCOPParameters) -> None:
        self.params = params

    def heating_curve(self, t_out: np.ndarray) -> np.ndarray:
        """Compute the UFH supply temperature from the outdoor temperature.

        Heating curve (stooklijn):

            T_supply(T_out) = T_supply_min + slope · max(T_ref_outdoor - T_out, 0)

        When T_out ≥ T_ref_outdoor the heating demand is zero and T_supply is
        clamped to T_supply_min.

        Args:
            t_out: Outdoor temperature forecast [°C], shape (N,).

        Returns:
            Supply temperature array [°C], shape (N,).  Guaranteed ≥ T_supply_min.

        Examples
        --------
        >>> cop_model.heating_curve(np.array([20.0, 10.0, 0.0, -10.0]))
        array([25., 33., 43., 53.])  # for slope=0.9, T_ref=18°C, T_supply_min=25°C
        """
        p = self.params
        t_out_arr = np.asarray(t_out, dtype=float)
        return p.T_supply_min + p.heating_curve_slope * np.maximum(p.T_ref_outdoor - t_out_arr, 0.0)

    def cop_from_temperatures(
        self,
        t_supply: np.ndarray | float,
        t_out: np.ndarray | float,
    ) -> np.ndarray:
        """Compute heat pump COP from supply temperature and outdoor temperature.

        Implements the Carnot formula:

            T_cond_K = (T_supply + Δ_cond) + T_CELSIUS_TO_KELVIN    [K]
            T_evap_K = (T_out    − Δ_evap) + T_CELSIUS_TO_KELVIN    [K]
            lift_K   = max(T_cond_K − T_evap_K, _MIN_TEMP_LIFT_K)   [K]
            COP      = clip(η · T_cond_K / lift_K, cop_min, cop_max)

        A higher T_supply (hotter condenser) lowers COP.
        A lower T_out (colder evaporator) lowers COP.

        Args:
            t_supply: Condenser supply temperature [°C].  Scalar or array; if
                      scalar it is broadcast over t_out.
            t_out:    Outdoor / evaporator-side temperature [°C].  Scalar or
                      array; if scalar it is broadcast over t_supply.

        Returns:
            COP array [dimensionless], shape = broadcast(t_supply, t_out).
            Values are clipped to ``[cop_min, cop_max]``.
        """
        p = self.params
        t_supply_arr = np.asarray(t_supply, dtype=float)
        t_out_arr = np.asarray(t_out, dtype=float)

        # Convert to absolute temperatures [K] — uses named constant, not 273.15
        t_cond_k = t_supply_arr + p.delta_T_cond + T_CELSIUS_TO_KELVIN
        t_evap_k = t_out_arr - p.delta_T_evap + T_CELSIUS_TO_KELVIN

        # Temperature lift [K]; floor at _MIN_TEMP_LIFT_K to prevent division by
        # zero when T_cond ≈ T_evap (e.g., very warm outdoor in summer).
        lift_k = np.maximum(t_cond_k - t_evap_k, _MIN_TEMP_LIFT_K)

        cop_carnot = t_cond_k / lift_k
        cop_actual = p.eta_carnot * cop_carnot
        return np.clip(cop_actual, p.cop_min, p.cop_max)

    def cop_ufh(self, t_out: np.ndarray) -> np.ndarray:
        """Compute UFH COP over the forecast horizon.

        The supply temperature is derived from the **heating curve** at each
        time step.  This correctly models the **double COP penalty** in cold
        weather: the evaporator temperature drops (from lower T_out) AND the
        condenser temperature rises (from the heating curve demanding warmer
        floor water).

        Args:
            t_out: Outdoor temperature forecast [°C], shape (N,).

        Returns:
            UFH COP array, shape (N,) [dimensionless, ∈ [cop_min, cop_max]].

        Notes
        -----
        The returned array can be passed directly to
        ``ForecastHorizon(cop_ufh_k=...)``.  The MPC then uses
        ``P_UFH_elec[k] = P_UFH[k] / cop_ufh[k]`` in its cost function (§14.1).
        """
        t_supply = self.heating_curve(np.asarray(t_out, dtype=float))
        return self.cop_from_temperatures(t_supply, t_out)

    def cop_dhw(self, t_out: np.ndarray, t_dhw_supply: float) -> np.ndarray:
        """Compute DHW COP over the forecast horizon.

        For DHW, the **supply temperature is approximately fixed** at the
        hot-water target temperature (T_dhw_min for normal operation,
        T_legionella during a legionella cycle).  Only the evaporator side
        (outdoor temperature) varies over the horizon.

        Args:
            t_out: Outdoor temperature forecast [°C], shape (N,).
            t_dhw_supply: Target DHW supply temperature [°C].  Typically
                          ``DHWMPCParameters.T_dhw_min`` for normal mode, or
                          ``DHWMPCParameters.T_legionella`` when a legionella
                          cycle is scheduled in the horizon.

        Returns:
            DHW COP array, shape (N,) [dimensionless, ∈ [cop_min, cop_max]].

        Notes
        -----
        The returned array can be passed to
        ``DHWForecastHorizon(cop_dhw_k=...)``.  DHW typically has a lower COP
        than UFH at the same outdoor temperature because the higher target
        temperature (e.g., 55 °C vs. 35 °C) increases the temperature lift.
        """
        return self.cop_from_temperatures(t_dhw_supply, np.asarray(t_out, dtype=float))

    def cop_from_measured_refrigerant(
        self,
        t_cond_c: np.ndarray | float,
        t_evap_c: np.ndarray | float,
    ) -> np.ndarray:
        """Compute COP directly from **measured** refrigerant temperatures.

        When the heat pump exposes ``refrigerant_condensation_temp_c`` and
        ``refrigerant_liquid_line_temp_c`` (liquid-line / sub-cooled side) as live sensor
        readings (see :class:`~home_optimizer.sensors.base.LiveReadings`),
        this method bypasses the approximations

            T_cond ≈ T_supply + Δ_cond
            T_evap ≈ T_out   − Δ_evap

        and uses the actual refrigerant cycle temperatures directly in the
        Carnot formula (§14.1).  This gives higher COP accuracy, especially
        during transients or when the heat pump operates outside its design
        envelope.

        Args:
            t_cond_c: Measured refrigerant condensation temperature [°C].
                      Scalar or array; shape must be broadcastable with
                      ``t_evap_c``.
            t_evap_c: Measured refrigerant evaporator / suction temperature
                      [°C].  Scalar or array.

        Returns:
            COP array [dimensionless], clipped to ``[cop_min, cop_max]``.

        Notes
        -----
        The ``delta_T_cond`` and ``delta_T_evap`` parameters from
        :class:`HeatPumpCOPParameters` are **not** applied here because the
        refrigerant temperatures are measured directly — no approach-temperature
        correction is needed.

        Examples
        --------
        >>> readings = backend.read_all()
        >>> cop_now = model.cop_from_measured_refrigerant(
        ...     t_cond_c=readings.refrigerant_condensation_temp_c,
        ...     t_evap_c=readings.discharge_temp_c,
        ... )
        """
        p = self.params
        t_cond_arr = np.asarray(t_cond_c, dtype=float)
        t_evap_arr = np.asarray(t_evap_c, dtype=float)

        # Convert directly to absolute temperatures [K] — no approach-temperature
        # correction because the sensors measure the refrigerant cycle directly.
        t_cond_k = t_cond_arr + T_CELSIUS_TO_KELVIN
        t_evap_k = t_evap_arr + T_CELSIUS_TO_KELVIN

        # Temperature lift [K]; floor to prevent division by zero (same guard as
        # cop_from_temperatures).
        lift_k = np.maximum(t_cond_k - t_evap_k, _MIN_TEMP_LIFT_K)

        cop_carnot = t_cond_k / lift_k
        cop_actual = p.eta_carnot * cop_carnot
        return np.clip(cop_actual, p.cop_min, p.cop_max)
