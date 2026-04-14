"""Factory helper: combine WeatherForecast + caller-supplied values into a ForecastHorizon.

This keeps the MPC core (thermal_model, mpc) fully independent of data sources.

PV and electricity price
------------------------
The MPC minimises ``p[k] · P_UFH[k] · Δt``.  If PV is producing, some of that
draw is covered by self-generated electricity — reducing the effective grid-import
cost.  This factory does **not** compute that correction internally (it would need
to know P_UFH, which is the optimisation variable).

Instead, pass the already-corrected price array::

    p_eff[k] = p[k]                    # no sun
    p_eff[k] = p[k] * (1 - P_pv/P_hp) # partial PV coverage
    p_eff[k] = 0                        # PV fully covers HP

Use :func:`effective_price` to compute this outside the factory, then pass the
result as ``price_eur_per_kwh``.
"""

from __future__ import annotations

import numpy as np

from ..types import ForecastHorizon
from .open_meteo import WeatherForecast


def effective_price(
    price_eur_per_kwh: float | np.ndarray,
    pv_power_kw: float | np.ndarray,
    hp_power_kw: float | np.ndarray,
) -> np.ndarray:
    """Compute the net grid-import price after PV self-consumption.

    When the PV system produces electricity that would otherwise be imported,
    the effective cost of running the heat pump is lower.

    Formula (per timestep k)::

        p_eff[k] = p[k] × max(0, P_hp[k] − P_pv[k]) / P_hp[k]

    When P_hp = 0, p_eff = 0 (HP is not running, no import cost).

    Parameters
    ----------
    price_eur_per_kwh:
        Grid electricity tariff [€/kWh] — scalar or array.
    pv_power_kw:
        PV production [kW] — scalar or array.  Use ``readings.pv_power_kw``
        for the current value, or a forecast array for the full horizon.
    hp_power_kw:
        Heat-pump electrical draw [kW] — scalar or array.  Use
        ``readings.hp_power_kw`` for the current value.  Must be > 0 where
        the HP is running.

    Returns
    -------
    np.ndarray
        Effective price [€/kWh], same shape as the broadcast of the inputs.
        Always ≥ 0.

    Examples
    --------
    HP draws 4 kW, PV produces 2 kW → 50 % self-consumed → half price::

        effective_price(0.28, pv_power_kw=2.0, hp_power_kw=4.0)
        # → 0.14 €/kWh

    HP draws 4 kW, PV produces 5 kW → fully covered → free::

        effective_price(0.28, pv_power_kw=5.0, hp_power_kw=4.0)
        # → 0.00 €/kWh
    """
    p = np.asarray(price_eur_per_kwh, dtype=float)
    pv = np.asarray(pv_power_kw, dtype=float)
    hp = np.asarray(hp_power_kw, dtype=float)

    # Fraction of HP draw that must still be imported from the grid
    net_import_fraction = np.where(hp > 0.0, np.clip((hp - pv) / hp, 0.0, 1.0), 0.0)
    return np.maximum(p * net_import_fraction, 0.0)


def build_forecast(
    weather: WeatherForecast,
    internal_gains_kw: float | np.ndarray,
    price_eur_per_kwh: float | np.ndarray,
    room_temperature_ref_c: float | np.ndarray,
    horizon_steps: int | None = None,
) -> ForecastHorizon:
    """Build a :class:`~home_optimizer.types.ForecastHorizon` from a WeatherForecast.

    Parameters
    ----------
    weather:
        Result of :meth:`~home_optimizer.sensors.OpenMeteoClient.get_forecast`.
    internal_gains_kw:
        Q_int [kW] — scalar or length-N array.
        Typical household: 0.2–0.8 kW.  Must come from caller's config; no default
        is provided to prevent silent magic-number injection.
    price_eur_per_kwh:
        Electricity tariff p[k] [€/kWh] — scalar or length-N array.
        If PV is installed, pass the output of :func:`effective_price` here.
        Must come from caller's config or energy-price source.
    room_temperature_ref_c:
        Comfort setpoint T_ref [°C] — scalar or length-(N+1) array.
        Must come from caller's config (user preference).
    horizon_steps:
        Override the number of MPC steps N.
        Defaults to ``weather.horizon_steps``.

    Returns
    -------
    ForecastHorizon
        Ready to pass directly to :meth:`~home_optimizer.mpc.UFHMPCController.solve`.

    Examples
    --------
    No PV::

        forecast = build_forecast(
            weather,
            internal_gains_kw=cfg["internal_gains_kw"],
            price_eur_per_kwh=cfg["price_eur_per_kwh"],
            room_temperature_ref_c=cfg["room_setpoint_c"],
        )

    With PV (use :func:`effective_price` to pre-correct the tariff)::

        readings = backend.read_all()
        p_eff = effective_price(cfg["price_eur_per_kwh"], readings.pv_power_kw, readings.hp_power_kw)
        forecast = build_forecast(
            weather,
            internal_gains_kw=cfg["internal_gains_kw"],
            price_eur_per_kwh=p_eff,
            room_temperature_ref_c=cfg["room_setpoint_c"],
        )
    """
    N = horizon_steps if horizon_steps is not None else weather.horizon_steps

    # --- outdoor temperature & irradiance from Open-Meteo ---
    t_out = weather.outdoor_temperature_c[:N]
    gti = weather.gti_w_per_m2[:N]

    # --- internal gains ---
    if np.isscalar(internal_gains_kw):
        q_int = np.full(N, float(internal_gains_kw))
    else:
        q_int = np.asarray(internal_gains_kw, dtype=float)[:N]

    # --- base electricity price (caller is responsible for PV correction) ---
    if np.isscalar(price_eur_per_kwh):
        prices = np.full(N, float(price_eur_per_kwh))
    else:
        prices = np.asarray(price_eur_per_kwh, dtype=float)[:N]

    # --- comfort setpoint ---
    if np.isscalar(room_temperature_ref_c):
        t_ref = np.full(N + 1, float(room_temperature_ref_c))
    else:
        t_ref = np.asarray(room_temperature_ref_c, dtype=float)
        if t_ref.size < N + 1:
            # Pad the last known value to reach length N+1
            t_ref = np.pad(t_ref, (0, N + 1 - t_ref.size), mode="edge")
        t_ref = t_ref[: N + 1]

    return ForecastHorizon(
        outdoor_temperature_c=t_out,
        gti_w_per_m2=gti,
        internal_gains_kw=q_int,
        price_eur_per_kwh=prices,
        room_temperature_ref_c=t_ref,
    )

