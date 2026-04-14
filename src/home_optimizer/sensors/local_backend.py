"""Local / standalone sensor backend.

Reads sensor values from a JSON file, environment variables, or fixed constants.
Suitable for local development and testing without Home Assistant.

JSON file format (sensors.json)
--------------------------------
{
    "room_temperature_c": 20.5,
    "pv_power_kw": 1.2,
    "hp_power_kw": 2.3
}

The file is re-read on every sensor call, so you can update it while the
optimizer is running (e.g. from a cron job or a simple write script).

Environment variables (all optional)
--------------------------------------
HOME_OPT_T_R   room temperature  [°C]   default 20.0
HOME_OPT_P_PV  PV production     [kW]   default 0.0
HOME_OPT_P_HP  heat-pump power   [kW]   default 0.0
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Union

from .base import SensorBackend

ValueSource = Union[float, Callable[[], float]]

# Keys expected in the JSON file
_KEY_T_R = "room_temperature_c"
_KEY_P_PV = "pv_power_kw"
_KEY_P_HP = "hp_power_kw"


def _resolve(source: ValueSource) -> float:
    """Evaluate a ValueSource: call it if callable, otherwise cast to float."""
    return float(source() if callable(source) else source)


class LocalBackend(SensorBackend):
    """Sensor backend for local / offline use.

    Parameters
    ----------
    room_temperature_c:
        T_r [°C] — fixed float or zero-arg callable.
    pv_power_kw:
        P_pv [kW] — same.  Pass ``0.0`` if there is no PV installation.
    hp_power_kw:
        P_hp_total [kW] — same.

    Examples
    --------
    Read from a JSON file (recommended for local use)::

        backend = LocalBackend.from_json_file("sensors.json")

    Fixed values (unit tests, quick demos)::

        backend = LocalBackend(room_temperature_c=20.5, pv_power_kw=1.2, hp_power_kw=2.0)

    Read from environment variables::

        backend = LocalBackend.from_env()
        # export HOME_OPT_T_R=21.3  HOME_OPT_P_PV=2.5  HOME_OPT_P_HP=3.1
    """

    ENV_T_R = "HOME_OPT_T_R"
    ENV_P_PV = "HOME_OPT_P_PV"
    ENV_P_HP = "HOME_OPT_P_HP"

    def __init__(
        self,
        room_temperature_c: ValueSource = 20.0,
        pv_power_kw: ValueSource = 0.0,
        hp_power_kw: ValueSource = 0.0,
    ) -> None:
        self._room_temp = room_temperature_c
        self._pv_power = pv_power_kw
        self._hp_power = hp_power_kw

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        defaults: dict[str, float] | None = None,
    ) -> "LocalBackend":
        """Create a LocalBackend that reads sensor values from a JSON file.

        The file is re-read on **every** sensor call, so updates take effect
        immediately without restarting the optimizer.

        Parameters
        ----------
        path:
            Path to the JSON file.  The file must contain a JSON object with
            the keys ``room_temperature_c``, ``pv_power_kw``, ``hp_power_kw``
            (all in the units stated).  Missing keys fall back to ``defaults``.
        defaults:
            Fallback values used when a key is absent from the file.
            Defaults to ``{"room_temperature_c": 20.0, "pv_power_kw": 0.0,
            "hp_power_kw": 0.0}``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist when a sensor value is first requested.
        ValueError
            If the file is not valid JSON or a value is not numeric.

        Example JSON file (``sensors.json``)::

            {
                "room_temperature_c": 20.8,
                "pv_power_kw": 2.4,
                "hp_power_kw": 3.1
            }

        Example usage::

            backend = LocalBackend.from_json_file("sensors.json")
            readings = backend.read_all()
        """
        _defaults = {"room_temperature_c": 20.0, "pv_power_kw": 0.0, "hp_power_kw": 0.0}
        if defaults:
            _defaults.update(defaults)

        resolved_path = Path(path)

        def _read(key: str) -> float:
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"Sensor file not found: {resolved_path.resolve()}\n"
                    f"Create it with at least: "
                    f'{{"{_KEY_T_R}": 20.0, "{_KEY_P_PV}": 0.0, "{_KEY_P_HP}": 0.0}}'
                )
            try:
                data: dict = json.loads(resolved_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {resolved_path}: {exc}") from exc

            value = data.get(key, _defaults[key])
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Key {key!r} in {resolved_path} is not numeric: {value!r}"
                ) from exc

        return cls(
            room_temperature_c=lambda: _read(_KEY_T_R),
            pv_power_kw=lambda: _read(_KEY_P_PV),
            hp_power_kw=lambda: _read(_KEY_P_HP),
        )

    @classmethod
    def from_env(cls) -> "LocalBackend":
        """Create a LocalBackend that re-reads environment variables on every call."""

        def _env(key: str, default: float) -> float:
            return float(os.environ.get(key, default))

        return cls(
            room_temperature_c=lambda: _env(cls.ENV_T_R, 20.0),
            pv_power_kw=lambda: _env(cls.ENV_P_PV, 0.0),
            hp_power_kw=lambda: _env(cls.ENV_P_HP, 0.0),
        )

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def get_room_temperature_c(self) -> float:
        return _resolve(self._room_temp)

    def get_pv_power_kw(self) -> float:
        return max(_resolve(self._pv_power), 0.0)

    def get_hp_power_kw(self) -> float:
        return max(_resolve(self._hp_power), 0.0)

