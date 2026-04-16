"""Open-Meteo weather forecast client (free, no API key required).

Fetches hourly ``temperature_2m`` and ``global_tilted_irradiance`` (GTI)
and returns them as a :class:`WeatherForecast` ready for :func:`build_forecast`.

Open-Meteo tilt / azimuth conventions
--------------------------------------
tilt    : 0 = horizontal surface, 90 = vertical wall
azimuth : 0 = South, −90 = East, +90 = West  (solar convention)

Defaults (south-facing windows, vertical)
  tilt=90, azimuth=0

For south-facing PV panels at 35 °:
  tilt=35, azimuth=0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import numpy as np

_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo returns hourly data; these constants control how many raw hourly
# points are fetched relative to the requested horizon.
_HOURS_PER_DAY: int = 24  # universele tijdconstante
_MIN_FETCH_DAYS: int = 2  # minimaal 2 dagen ophalen voor buffer
_FETCH_BUFFER_HOURS: int = 2  # extra uur-buffer voor uitlijning op huidig uur
_DT_FLOAT_TOLERANCE: float = 1e-9  # numerieke tolerantie voor dt ≈ 1 h vergelijking

#: Days per year approximation used in the seasonal cosine model.
#: Leap years introduce < 0.3 % error — within seasonal model uncertainty.
_DAYS_PER_YEAR: float = 365.0

#: Valid range for day-of-year input.
_DAY_OF_YEAR_MIN: int = 1
_DAY_OF_YEAR_MAX: int = 366  # day 366 occurs in leap years


@dataclass(frozen=True, slots=True)
class SeasonalMainsModel:
    """Seasonal cosine model for cold mains water temperature.

    Implements a sinusoidal annual cycle (grey-box, calibrated against water-utility
    measurement data):

        T_mains(d) = t_mean_c + t_amplitude_k × cos(2π × (d − d_peak) / 365)

    where d is the day-of-year (1–365/366).

    For the Netherlands, representative parameters (KIWA / Vitens seasonal data):

        t_mean_c              ≈ 10.5 °C  (annual mean)
        t_amplitude_k         ≈  3.5 K   (seasonal swing ± 3.5 K around mean)
        day_of_year_peak_warm ≈ 246       (≈ 3 September — annual maximum)

    These are provided by :meth:`for_netherlands` as named constants — not
    hardcoded in any formula.  Validate against your local water utility before
    deploying in a DHW energy-balance model.

    Parameters
    ----------
    t_mean_c:
        Annual mean cold mains temperature [°C].
    t_amplitude_k:
        Half-amplitude of the seasonal cycle [K].  Must be ≥ 0.
    day_of_year_peak_warm:
        Day-of-year on which the mains temperature is highest (1–366).
        NL typical: 246 (first week of September).
    """

    t_mean_c: float
    t_amplitude_k: float
    day_of_year_peak_warm: int

    def __post_init__(self) -> None:
        if self.t_amplitude_k < 0.0:
            raise ValueError("t_amplitude_k must be ≥ 0.")
        if not _DAY_OF_YEAR_MIN <= self.day_of_year_peak_warm <= _DAY_OF_YEAR_MAX:
            raise ValueError(
                f"day_of_year_peak_warm must be in "
                f"[{_DAY_OF_YEAR_MIN}, {_DAY_OF_YEAR_MAX}], "
                f"got {self.day_of_year_peak_warm}."
            )

    def estimate(self, dt: datetime) -> float:
        """Estimate cold mains temperature for a given date [°C].

        Only the calendar day is used; sub-daily variation is below model accuracy.

        Parameters
        ----------
        dt:
            Datetime (timezone-aware or naive).

        Returns
        -------
        float
            Estimated T_mains [°C].
        """
        day = dt.timetuple().tm_yday
        return self.t_mean_c + self.t_amplitude_k * math.cos(
            2.0 * math.pi * (day - self.day_of_year_peak_warm) / _DAYS_PER_YEAR
        )

    @classmethod
    def for_netherlands(cls) -> "SeasonalMainsModel":
        """Return a SeasonalMainsModel with typical Dutch cold-mains parameters.

        Source: KIWA / Vitens seasonal water temperature profiles (NL national average).
        Validate against your regional water utility for high-accuracy DHW modelling.

        Returns
        -------
        SeasonalMainsModel
            t_mean_c=10.5 °C, t_amplitude_k=3.5 K, day_of_year_peak_warm=246.
        """
        # Named constants — never use raw literals in formulas (§anti-pattern).
        _NL_T_MEAN_C: float = 10.5
        _NL_T_AMPLITUDE_K: float = 3.5
        _NL_DAY_PEAK_WARM: int = 246
        return cls(
            t_mean_c=_NL_T_MEAN_C,
            t_amplitude_k=_NL_T_AMPLITUDE_K,
            day_of_year_peak_warm=_NL_DAY_PEAK_WARM,
        )


@dataclass(frozen=True, slots=True)
class WeatherForecast:
    """Parsed Open-Meteo forecast aligned to the current UTC hour.

    Attributes
    ----------
    outdoor_temperature_c:
        T_out forecast [°C], length ``horizon_steps``.
    gti_w_per_m2:
        GTI for south-facing windows [W/m²], length ``horizon_steps``.
        Used by the UFH thermal model (solar gain through glazing).  Always ≥ 0.
    gti_pv_w_per_m2:
        GTI for PV panels [W/m²], length ``horizon_steps``, or ``None`` when
        ``pv_tilt`` / ``pv_azimuth`` were not set on the client.  Always ≥ 0.
    horizon_steps:
        Number of forecast steps N.
    dt_hours:
        Time step used when resampling [h].
    valid_from:
        UTC datetime of the first forecast step (= current hour rounded down).
    """

    outdoor_temperature_c: np.ndarray
    gti_w_per_m2: np.ndarray
    horizon_steps: int
    dt_hours: float
    valid_from: datetime
    gti_pv_w_per_m2: np.ndarray | None = None


class OpenMeteoClient:
    """Fetch temperature and irradiance forecasts from Open-Meteo.

    Two separate surfaces can be configured in one client:

    - **Windows** (``tilt`` / ``azimuth``): GTI used by the UFH thermal model
      to calculate solar heat gain through glazing.
      Typical south-facing wall:  ``tilt=90, azimuth=0``  *(default)*

    - **PV panels** (``pv_tilt`` / ``pv_azimuth``): GTI for estimating PV
      production.  When set, a second API call is made and the result is stored
      in :attr:`WeatherForecast.gti_pv_w_per_m2`.
      Typical south-facing panel:  ``pv_tilt=35, pv_azimuth=0``

    Parameters
    ----------
    latitude:
        Site latitude [°N], e.g. ``52.37`` for Amsterdam.
    longitude:
        Site longitude [°E], e.g. ``4.90`` for Amsterdam.
    tilt:
        Window surface tilt [°].  ``90`` = vertical wall.  Default ``90``.
    azimuth:
        Window surface azimuth [°].  ``0`` = South.  Default ``0``.
    pv_tilt:
        PV panel tilt [°].  ``None`` = no PV forecast (default).
    pv_azimuth:
        PV panel azimuth [°].  ``0`` = South.  Default ``0``.
        Ignored when ``pv_tilt`` is ``None``.
    timeout:
        HTTP request timeout [s].  Default ``15.0``.

    Examples
    --------
    Windows only (no PV)::

        client = OpenMeteoClient(latitude=52.37, longitude=4.90)
        forecast = client.get_forecast(horizon_hours=24)
        # forecast.gti_pv_w_per_m2 is None

    Windows + PV panels at 35 °::

        client = OpenMeteoClient(
            latitude=52.37, longitude=4.90,
            tilt=90, azimuth=0,       # south-facing windows
            pv_tilt=35, pv_azimuth=0, # south-facing PV panels
        )
        forecast = client.get_forecast(horizon_hours=24)
        # forecast.gti_w_per_m2      → solar gain through glazing
        # forecast.gti_pv_w_per_m2   → irradiance on PV panels
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        tilt: float = 90.0,
        azimuth: float = 0.0,
        pv_tilt: float | None = None,
        pv_azimuth: float = 0.0,
        timeout: float = 15.0,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.tilt = tilt
        self.azimuth = azimuth
        self.pv_tilt = pv_tilt
        self.pv_azimuth = pv_azimuth
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_forecast(
        self,
        horizon_hours: int = 24,
        dt_hours: float = 1.0,
    ) -> WeatherForecast:
        """Fetch forecast starting at the current UTC hour.

        Makes one API call for temperature + window GTI, and — when
        ``pv_tilt`` is configured — a second call for PV panel GTI.

        Parameters
        ----------
        horizon_hours:
            Total forecast window [h].  E.g. ``24`` for a one-day horizon.
        dt_hours:
            Desired time step [h].  Must be > 0.  Default ``1.0``.

        Returns
        -------
        WeatherForecast
            Arrays of length ``ceil(horizon_hours / dt_hours)``.
            ``gti_pv_w_per_m2`` is populated only when ``pv_tilt`` is set.
        """
        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive.")

        fetch_days = max(
            _MIN_FETCH_DAYS,
            int(np.ceil((horizon_hours + _FETCH_BUFFER_HOURS) / _HOURS_PER_DAY)) + 1,
        )
        n_steps = int(np.ceil(horizon_hours / dt_hours))
        need_h = max(
            int(np.ceil(n_steps * dt_hours)) + _FETCH_BUFFER_HOURS,
            horizon_hours + _FETCH_BUFFER_HOURS,
        )

        # --- call 1: temperature + window GTI ---
        params: dict[str, str | int | float | bool | None] = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m,global_tilted_irradiance",
            "tilt": self.tilt,
            "azimuth": self.azimuth,
            "forecast_days": fetch_days,
            "timezone": "UTC",
        }
        hourly, start_idx, valid_from = self._fetch_hourly(params)

        temps_h = self._extract(hourly.get("temperature_2m", []), start_idx, need_h)
        gti_h = np.maximum(
            self._extract(hourly.get("global_tilted_irradiance", []), start_idx, need_h), 0.0
        )
        temps_out, gti_out = self._resample(temps_h, gti_h, n_steps, dt_hours)

        # --- call 2 (optional): PV panel GTI ---
        gti_pv_out: np.ndarray | None = None
        if self.pv_tilt is not None:
            pv_params: dict[str, str | int | float | bool | None] = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": "global_tilted_irradiance",
                "tilt": self.pv_tilt,
                "azimuth": self.pv_azimuth,
                "forecast_days": fetch_days,
                "timezone": "UTC",
            }
            pv_hourly, pv_start, _ = self._fetch_hourly(pv_params)
            gti_pv_h = np.maximum(
                self._extract(pv_hourly.get("global_tilted_irradiance", []), pv_start, need_h),
                0.0,
            )
            _, gti_pv_out = self._resample(gti_pv_h, gti_pv_h, n_steps, dt_hours)

        return WeatherForecast(
            outdoor_temperature_c=temps_out,
            gti_w_per_m2=gti_out,
            horizon_steps=n_steps,
            dt_hours=dt_hours,
            valid_from=valid_from,
            gti_pv_w_per_m2=gti_pv_out,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_hourly(
        self, params: dict[str, str | int | float | bool | None]
    ) -> tuple[dict, int, datetime]:
        """Execute one Open-Meteo request and return (hourly_dict, start_idx, valid_from)."""
        try:
            response = httpx.get(_BASE_URL, params=params, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ConnectionError(f"Open-Meteo request failed: {exc}") from exc

        hourly: dict = response.json().get("hourly", {})
        times_raw: list[str] = hourly.get("time", [])
        if not times_raw:
            raise ValueError("Open-Meteo returned no hourly data.")

        start_idx, valid_from = self._find_start_index(times_raw)
        return hourly, start_idx, valid_from

    @staticmethod
    def _parse_time(t_str: str) -> datetime:
        if "T" not in t_str:
            t_str = t_str + "T00:00"
        return datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)

    def _find_start_index(self, times_raw: list[str]) -> tuple[int, datetime]:
        now_utc = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        for i, t_str in enumerate(times_raw):
            parsed = self._parse_time(t_str)
            if parsed >= now_utc:
                return i, parsed
        raise ValueError(
            "Current UTC hour not found in Open-Meteo response. "
            "The forecast may not start from today."
        )

    @staticmethod
    def _extract(raw: list, start: int, n: int) -> np.ndarray:
        """Slice raw list, replace None with NaN, forward-fill NaN."""
        chunk = raw[start : start + n]
        arr = np.array(
            [float(v) if v is not None else float("nan") for v in chunk],
            dtype=float,
        )
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            idx = np.where(~nan_mask, np.arange(len(arr)), 0)
            np.maximum.accumulate(idx, out=idx)
            arr = arr[idx]
        return arr

    @staticmethod
    def _resample(
        a: np.ndarray,
        b: np.ndarray,
        n_steps: int,
        dt_hours: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample a pair of hourly arrays to the requested dt_hours grid."""
        if abs(dt_hours - 1.0) < _DT_FLOAT_TOLERANCE:
            return a[:n_steps], b[:n_steps]
        t_src = np.arange(len(a), dtype=float)
        t_tgt = np.arange(n_steps, dtype=float) * dt_hours
        return np.interp(t_tgt, t_src, a), np.interp(t_tgt, t_src, b)
