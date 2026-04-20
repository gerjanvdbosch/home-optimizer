"""History-based DHW tap-flow forecaster.

This provider estimates the exogenous tap-flow disturbance ``V_tap[k]`` from
persisted telemetry using the DHW total-energy balance and then projects that
history onto a recurring hour-of-day profile. The result is a horizon-wide
``dhw_v_tap_forecast`` array that the runtime MPC can use as a known LTV
parameter instead of assuming a constant draw all day long.

Units: power [kW], energy [kWh], temperature [°C], volume flow [m³/h], time [h].
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    PersistedRegressorArtifactMetadata,
    load_regressor_artifact,
    persist_regressor_artifact,
)
from .models import DHWTapForecastSettings
from ..telemetry.models import TelemetryAggregate

if TYPE_CHECKING:
    from ..telemetry.repository import TelemetryRepository

_SECONDS_PER_HOUR: float = 3600.0
_DHW_MODE_NAME: str = "dhw"
_HOURS_PER_DAY: int = 24
_ARTIFACT_NAME: str = "dhw_tap_profile"
_ARTIFACT_VERSION: int = 1

log = logging.getLogger("home_optimizer.forecasting.dhw_tap")


@dataclass(frozen=True, slots=True)
class _PersistedDHWTapProfile:
    """Persisted recurring hour-of-day DHW tap profile.

    Attributes:
        hourly_mean_m3_per_h: Mean inferred tap flow per UTC hour [m³/h], shape
            ``(24,)``.
        hourly_sample_count: Historical sample count per UTC hour [-], shape
            ``(24,)``. Sparse hours remain explicit so runtime inference can keep
            falling back to zero instead of inventing demand.
    """

    hourly_mean_m3_per_h: np.ndarray
    hourly_sample_count: np.ndarray

    def __post_init__(self) -> None:
        if self.hourly_mean_m3_per_h.shape != (_HOURS_PER_DAY,):
            raise ValueError("hourly_mean_m3_per_h must contain exactly 24 hourly values.")
        if self.hourly_sample_count.shape != (_HOURS_PER_DAY,):
            raise ValueError("hourly_sample_count must contain exactly 24 hourly values.")
        if np.any(self.hourly_mean_m3_per_h < 0.0):
            raise ValueError("hourly_mean_m3_per_h must remain non-negative [m³/h].")
        if np.any(self.hourly_sample_count < 0):
            raise ValueError("hourly_sample_count must remain non-negative [-].")


@dataclass(frozen=True, slots=True)
class _CachedDHWTapProfile:
    """Cached trained DHW tap profile or cached artifact miss marker."""

    profile: _PersistedDHWTapProfile | None
    sample_count: int
    artifact_mtime_ns: int | None = None


class DHWTapForecaster:
    """Forecast recurring DHW tap flow from persisted telemetry history.

    The forecaster keeps the model intentionally transparent: it derives a
    physically interpretable hour-of-day draw profile directly from historical tank
    energy balances, and optionally persists that recurring profile to disk for the
    nightly forecast-training workflow. Runtime inference still falls back to the
    direct history path when no compatible artifact is available.
    """

    _model_cache: dict[tuple[object, ...], _CachedDHWTapProfile] = {}
    _cache_lock: Lock = Lock()

    def __init__(self, settings: DHWTapForecastSettings | None = None) -> None:
        self._settings = settings or DHWTapForecastSettings()

    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear all cached persisted DHW tap profiles in the current process."""

        with cls._cache_lock:
            cls._model_cache.clear()

    @staticmethod
    def _total_energy_kwh(*, t_top_c: float, t_bot_c: float, c_top_kwh_per_k: float, c_bot_kwh_per_k: float) -> float:
        """Return total tank energy proxy ``C_top·T_top + C_bot·T_bot`` [kWh]."""
        return c_top_kwh_per_k * t_top_c + c_bot_kwh_per_k * t_bot_c

    def _is_profile_runtime_trustworthy(self, profile: _PersistedDHWTapProfile) -> bool:
        """Return whether a recurring DHW profile is informative enough for runtime MPC use.

        The hour-of-day forecaster estimates ``V_tap`` indirectly from the DHW
        total-energy balance. Small bias errors in ``R_loss`` or sensor corrections can
        create a *flat positive floor* across all hours, even when the real demand is
        sparse (e.g. a single evening shower at 22:00).  Feeding such an artifact to
        the MPC makes it maintain ``T_top`` almost continuously, which wastes energy.

        The guard uses a **peak-to-background ratio**:

            ratio = peak(sampled hourly means) / median(sampled hourly means)

        A truly flat uniform bias yields ratio ≈ 1.0 → rejected.
        A sparse tap event (e.g. 22:00 peak = 0.10 m³/h, background median = 0.02)
        yields ratio ≈ 5.0 → accepted.

        The ``mean/peak`` approach used previously was fragile: with realistic noise
        (background ≈ 0.03, peak ≈ 0.10, 24 hours) the mean/peak ratio ≈ 0.33,
        which exceeded the old threshold of 0.30 and silently rejected valid profiles.
        The median-based ratio is immune to this: the median of a sparse-tap profile
        is dominated by the many low-background hours, so the ratio clearly separates
        real demand from calibration artifacts.

        Args:
            profile: Persisted recurring hourly tap profile [m³/h].

        Returns:
            ``True`` when the profile is trustworthy for runtime MPC use.
        """

        sampled_hour_mask = profile.hourly_sample_count >= self._settings.min_samples_per_hour
        if not np.any(sampled_hour_mask):
            return False

        sampled_hourly_means = profile.hourly_mean_m3_per_h[sampled_hour_mask]
        peak_m3_per_h = float(np.max(sampled_hourly_means))
        if peak_m3_per_h <= 0.0:
            # All sampled hours are zero → no detectable tap demand at all.
            return False

        # Use the median as the background level; it is robust against a single
        # large peak hour and represents "what a typical non-shower hour looks like".
        background_m3_per_h = float(np.median(sampled_hourly_means))

        if background_m3_per_h <= 0.0:
            # Median is zero: at least half the hours have no inferred tap flow,
            # which means the profile is sparse and the peak is clearly real.
            return True

        peak_to_background_ratio = peak_m3_per_h / background_m3_per_h
        if peak_to_background_ratio < self._settings.min_peak_to_background_ratio:
            log.info(
                "Rejecting DHW tap profile because it is too flat for runtime use: "
                "peak/background=%.3f < %.3f (peak=%.4f m³/h, background_median=%.4f m³/h).",
                peak_to_background_ratio,
                self._settings.min_peak_to_background_ratio,
                peak_m3_per_h,
                background_m3_per_h,
            )
            return False
        return True

    def _infer_v_tap_m3_per_h(
        self,
        *,
        previous_row: TelemetryAggregate,
        next_row: TelemetryAggregate,
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
        top_temperature_bias_c: float,
        bottom_temperature_bias_c: float,
        boiler_ambient_bias_c: float,
    ) -> float | None:
        """Infer one tap-flow sample from the DHW total-energy balance.

        Implements the rearranged tank balance from §9.5:

            V_tap = (P_dhw - Q_loss - ΔE/Δt) / (λ·(T_top - T_mains))

        using the mean outlet/ambient temperatures over the interval and clamping the
        result to the physically feasible interval ``[0, max_implied_tap_m3_per_h]``.
        """
        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > self._settings.max_history_gap_hours:
            return None

        start_energy_kwh = self._total_energy_kwh(
            t_top_c=float(previous_row.dhw_top_temperature_last_c) + top_temperature_bias_c,
            t_bot_c=float(previous_row.dhw_bottom_temperature_last_c) + bottom_temperature_bias_c,
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
        )
        end_energy_kwh = self._total_energy_kwh(
            t_top_c=float(next_row.dhw_top_temperature_last_c) + top_temperature_bias_c,
            t_bot_c=float(next_row.dhw_bottom_temperature_last_c) + bottom_temperature_bias_c,
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
        )
        delta_energy_rate_kw = (end_energy_kwh - start_energy_kwh) / dt_hours
        mean_t_top_c = 0.5 * (
            float(previous_row.dhw_top_temperature_last_c)
            + top_temperature_bias_c
            + float(next_row.dhw_top_temperature_last_c)
            + top_temperature_bias_c
        )
        mean_t_bot_c = 0.5 * (
            float(previous_row.dhw_bottom_temperature_last_c)
            + bottom_temperature_bias_c
            + float(next_row.dhw_bottom_temperature_last_c)
            + bottom_temperature_bias_c
        )
        t_amb_c = float(next_row.boiler_ambient_temp_mean_c) + boiler_ambient_bias_c
        t_mains_c = float(next_row.t_mains_estimated_mean_c)
        q_loss_kw = (mean_t_top_c - t_amb_c) / r_loss_k_per_kw + (mean_t_bot_c - t_amb_c) / r_loss_k_per_kw
        p_dhw_kw = (
            max(float(next_row.hp_thermal_power_mean_kw), 0.0)
            if str(next_row.hp_mode_last).strip().lower() == _DHW_MODE_NAME
            else 0.0
        )

        tap_denominator = lambda_water_kwh_per_m3_k * (mean_t_top_c - t_mains_c)
        if tap_denominator <= 0.0:
            return 0.0

        implied_v_tap = (p_dhw_kw - q_loss_kw - delta_energy_rate_kw) / tap_denominator
        clamped_v_tap = float(np.clip(implied_v_tap, 0.0, self._settings.max_implied_tap_m3_per_h))
        return clamped_v_tap

    def train_and_persist_from_repository(
        self,
        *,
        repository: "TelemetryRepository",
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
        top_temperature_bias_c: float,
        bottom_temperature_bias_c: float,
        boiler_ambient_bias_c: float,
    ) -> PersistedRegressorArtifactMetadata | None:
        """Train and persist one recurring hourly DHW tap profile.

        Args:
            repository: Telemetry repository supplying historical DHW buckets and
                the canonical artifact storage path.
            c_top_kwh_per_k: DHW top-layer heat capacity ``C_top`` [kWh/K].
            c_bot_kwh_per_k: DHW bottom-layer heat capacity ``C_bot`` [kWh/K].
            r_loss_k_per_kw: DHW standby-loss resistance ``R_loss`` [K/kW].
            lambda_water_kwh_per_m3_k: Water volumetric heat capacity ``λ`` [kWh/(m³·K)].

        Returns:
            Persisted artifact metadata, or ``None`` when history is too sparse to
            train a trustworthy recurring profile.
        """

        profile, valid_sample_count = self._build_hourly_profile_from_history(
            repository=repository,
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
            r_loss_k_per_kw=r_loss_k_per_kw,
            lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
            top_temperature_bias_c=top_temperature_bias_c,
            bottom_temperature_bias_c=bottom_temperature_bias_c,
            boiler_ambient_bias_c=boiler_ambient_bias_c,
        )
        if profile is None:
            log.info(
                "Skipping DHW tap-profile training: need at least %d valid samples, found %d.",
                self._settings.min_training_samples,
                valid_sample_count,
            )
            return None

        metadata = PersistedRegressorArtifactMetadata(
            model_version=_ARTIFACT_VERSION,
            trained_at_utc=datetime.now(tz=timezone.utc),
            sample_count=valid_sample_count,
            settings_signature=self._settings_signature(
                physical_signature=self._physical_signature(
                    c_top_kwh_per_k=c_top_kwh_per_k,
                    c_bot_kwh_per_k=c_bot_kwh_per_k,
                    r_loss_k_per_kw=r_loss_k_per_kw,
                    lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
                    top_temperature_bias_c=top_temperature_bias_c,
                    bottom_temperature_bias_c=bottom_temperature_bias_c,
                    boiler_ambient_bias_c=boiler_ambient_bias_c,
                )
            ),
            training_fingerprint=repository.forecast_training_fingerprint(),
        )
        artifact_path = repository.forecast_artifact_path(_ARTIFACT_NAME)
        self._persist_artifact(artifact_path=artifact_path, profile=profile, metadata=metadata)
        return metadata

    def _build_hourly_profile_from_history(
        self,
        *,
        repository: "TelemetryRepository",
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
        top_temperature_bias_c: float,
        bottom_temperature_bias_c: float,
        boiler_ambient_bias_c: float,
    ) -> tuple[_PersistedDHWTapProfile | None, int]:
        """Infer a recurring hourly tap profile from historical telemetry.

        The inferred profile is the direct runtime implementation of the DHW total
        energy balance from §9.5: first infer one physically clamped ``V_tap``
        sample per admissible telemetry transition, then aggregate those samples by
        UTC hour of day.
        """

        self._validate_physical_parameters(
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
            r_loss_k_per_kw=r_loss_k_per_kw,
            lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
        )
        rows = sorted(repository.list_aggregates(), key=lambda row: row.bucket_end_utc)
        if len(rows) < 2:
            return None, 0

        samples_by_hour: dict[int, list[float]] = defaultdict(list)
        valid_sample_count = 0
        for previous_row, next_row in zip(rows, rows[1:]):
            inferred_v_tap = self._infer_v_tap_m3_per_h(
                previous_row=previous_row,
                next_row=next_row,
                c_top_kwh_per_k=c_top_kwh_per_k,
                c_bot_kwh_per_k=c_bot_kwh_per_k,
                r_loss_k_per_kw=r_loss_k_per_kw,
                lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
                top_temperature_bias_c=top_temperature_bias_c,
                bottom_temperature_bias_c=bottom_temperature_bias_c,
                boiler_ambient_bias_c=boiler_ambient_bias_c,
            )
            if inferred_v_tap is None:
                continue
            samples_by_hour[int(next_row.bucket_end_utc.hour)].append(inferred_v_tap)
            valid_sample_count += 1

        if valid_sample_count < self._settings.min_training_samples:
            return None, valid_sample_count

        hourly_mean_m3_per_h = np.zeros(_HOURS_PER_DAY, dtype=float)
        hourly_sample_count = np.zeros(_HOURS_PER_DAY, dtype=int)
        for hour in range(_HOURS_PER_DAY):
            hour_samples = samples_by_hour.get(hour, [])
            hourly_sample_count[hour] = len(hour_samples)
            if hour_samples:
                hourly_mean_m3_per_h[hour] = float(np.mean(hour_samples))

        profile = _PersistedDHWTapProfile(
            hourly_mean_m3_per_h=hourly_mean_m3_per_h,
            hourly_sample_count=hourly_sample_count,
        )
        if not self._is_profile_runtime_trustworthy(profile):
            return None, valid_sample_count
        return profile, valid_sample_count

    def _predict_from_history(
        self,
        *,
        repository: "TelemetryRepository",
        horizon_valid_at_utc: Sequence[datetime],
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
        top_temperature_bias_c: float,
        bottom_temperature_bias_c: float,
        boiler_ambient_bias_c: float,
    ) -> np.ndarray | None:
        """Infer one horizon-wide recurring DHW tap forecast directly from history."""

        profile, _valid_sample_count = self._build_hourly_profile_from_history(
            repository=repository,
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
            r_loss_k_per_kw=r_loss_k_per_kw,
            lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
            top_temperature_bias_c=top_temperature_bias_c,
            bottom_temperature_bias_c=bottom_temperature_bias_c,
            boiler_ambient_bias_c=boiler_ambient_bias_c,
        )
        if profile is None:
            return None
        return self._forecast_from_profile(profile=profile, horizon_valid_at_utc=horizon_valid_at_utc)

    def _forecast_from_profile(
        self,
        *,
        profile: _PersistedDHWTapProfile,
        horizon_valid_at_utc: Sequence[datetime],
    ) -> np.ndarray:
        """Project one recurring hourly DHW profile onto the requested horizon."""

        forecast_values: list[float] = []
        for valid_at_utc in horizon_valid_at_utc:
            hour_index = int(valid_at_utc.hour)
            if int(profile.hourly_sample_count[hour_index]) < self._settings.min_samples_per_hour:
                forecast_values.append(0.0)
                continue
            forecast_values.append(float(profile.hourly_mean_m3_per_h[hour_index]))
        return np.asarray(forecast_values, dtype=float)

    @staticmethod
    def _physical_signature(
        *,
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
        top_temperature_bias_c: float,
        bottom_temperature_bias_c: float,
        boiler_ambient_bias_c: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return the physical parameter tuple that the persisted profile depends on."""

        return (
            float(c_top_kwh_per_k),
            float(c_bot_kwh_per_k),
            float(r_loss_k_per_kw),
            float(lambda_water_kwh_per_m3_k),
            float(top_temperature_bias_c),
            float(bottom_temperature_bias_c),
            float(boiler_ambient_bias_c),
        )

    def _settings_signature(
        self,
        *,
        physical_signature: tuple[float, float, float, float],
    ) -> tuple[object, ...]:
        """Return the full artifact-compatibility signature for this DHW profile."""

        return (
            self._settings.min_training_samples,
            self._settings.min_samples_per_hour,
            self._settings.max_history_gap_hours,
            self._settings.max_implied_tap_m3_per_h,
            self._settings.min_peak_to_background_ratio,
            *physical_signature,
        )

    @staticmethod
    def _validate_physical_parameters(
        *,
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
    ) -> None:
        """Fail fast on non-physical DHW parameters before energy-balance inference."""

        for parameter_name, value in (
            ("c_top_kwh_per_k", c_top_kwh_per_k),
            ("c_bot_kwh_per_k", c_bot_kwh_per_k),
            ("r_loss_k_per_kw", r_loss_k_per_kw),
            ("lambda_water_kwh_per_m3_k", lambda_water_kwh_per_m3_k),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{parameter_name} must be strictly positive.")

    def _load_cached_artifact(
        self,
        repository: "TelemetryRepository",
        *,
        physical_signature: tuple[float, float, float, float],
    ) -> _CachedDHWTapProfile:
        """Load the persisted DHW tap profile, reusing an in-process cache."""

        artifact_path = repository.forecast_artifact_path(_ARTIFACT_NAME)
        if not artifact_path.exists():
            return _CachedDHWTapProfile(profile=None, sample_count=0, artifact_mtime_ns=None)

        artifact_mtime_ns = artifact_path.stat().st_mtime_ns
        cache_key = (str(artifact_path), self._settings_signature(physical_signature=physical_signature))
        with type(self)._cache_lock:
            cached = type(self)._model_cache.get(cache_key)
        if cached is not None and cached.artifact_mtime_ns == artifact_mtime_ns:
            return cached

        loaded = self._load_artifact_from_disk(artifact_path)
        if loaded is None:
            return _CachedDHWTapProfile(profile=None, sample_count=0, artifact_mtime_ns=None)

        metadata, profile = loaded
        if metadata.model_version != _ARTIFACT_VERSION:
            log.warning(
                "Ignoring DHW tap-profile artifact %s with version %s (expected %s).",
                artifact_path,
                metadata.model_version,
                _ARTIFACT_VERSION,
            )
            return _CachedDHWTapProfile(profile=None, sample_count=0, artifact_mtime_ns=None)
        if metadata.settings_signature != self._settings_signature(physical_signature=physical_signature):
            log.info(
                "Ignoring DHW tap-profile artifact %s because the settings/physics signature changed.",
                artifact_path,
            )
            return _CachedDHWTapProfile(profile=None, sample_count=0, artifact_mtime_ns=None)
        if not self._is_profile_runtime_trustworthy(profile):
            log.info(
                "Ignoring DHW tap-profile artifact %s because the recurring profile is too flat.",
                artifact_path,
            )
            return _CachedDHWTapProfile(profile=None, sample_count=0, artifact_mtime_ns=None)

        cached = _CachedDHWTapProfile(
            profile=profile,
            sample_count=metadata.sample_count,
            artifact_mtime_ns=artifact_mtime_ns,
        )
        with type(self)._cache_lock:
            type(self)._model_cache[cache_key] = cached
        return cached

    def _persist_artifact(
        self,
        *,
        artifact_path,
        profile: _PersistedDHWTapProfile,
        metadata: PersistedRegressorArtifactMetadata,
    ) -> None:
        """Persist one trained recurring DHW tap profile atomically to disk."""

        persist_regressor_artifact(
            artifact_path=artifact_path,
            regressor={
                "hourly_mean_m3_per_h": profile.hourly_mean_m3_per_h,
                "hourly_sample_count": profile.hourly_sample_count,
            },
            metadata=metadata,
        )

        cached = _CachedDHWTapProfile(
            profile=profile,
            sample_count=metadata.sample_count,
            artifact_mtime_ns=artifact_path.stat().st_mtime_ns,
        )
        cache_key = (
            str(artifact_path),
            tuple(metadata.settings_signature),
        )
        with type(self)._cache_lock:
            type(self)._model_cache[cache_key] = cached

    def _load_artifact_from_disk(
        self,
        artifact_path,
    ) -> tuple[PersistedRegressorArtifactMetadata, _PersistedDHWTapProfile] | None:
        """Deserialize one persisted DHW tap profile artifact from disk."""

        loaded = load_regressor_artifact(artifact_path)
        if loaded is None:
            return None
        metadata, artifact_payload = loaded
        if not isinstance(artifact_payload, dict):
            log.warning("Ignoring DHW tap-profile artifact %s with invalid payload type.", artifact_path)
            return None

        hourly_mean_raw = artifact_payload.get("hourly_mean_m3_per_h")
        hourly_count_raw = artifact_payload.get("hourly_sample_count")
        try:
            profile = _PersistedDHWTapProfile(
                hourly_mean_m3_per_h=np.asarray(hourly_mean_raw, dtype=float),
                hourly_sample_count=np.asarray(hourly_count_raw, dtype=int),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Ignoring malformed DHW tap-profile artifact at %s: %s", artifact_path, exc)
            return None
        return metadata, profile

    def predict_from_repository(
        self,
        *,
        repository: Any,
        horizon_valid_at_utc: Sequence[datetime],
        c_top_kwh_per_k: float,
        c_bot_kwh_per_k: float,
        r_loss_k_per_kw: float,
        lambda_water_kwh_per_m3_k: float,
        top_temperature_bias_c: float,
        bottom_temperature_bias_c: float,
        boiler_ambient_bias_c: float,
    ) -> np.ndarray | None:
        """Return an hour-of-day DHW tap-flow forecast for the requested horizon.

        Args:
            repository: Telemetry repository exposing ``list_aggregates()``.
            horizon_valid_at_utc: Forecast validity timestamps for the upcoming MPC
                horizon [UTC].
            c_top_kwh_per_k: DHW top-layer heat capacity ``C_top`` [kWh/K].
            c_bot_kwh_per_k: DHW bottom-layer heat capacity ``C_bot`` [kWh/K].
            r_loss_k_per_kw: DHW standby-loss resistance ``R_loss`` [K/kW].
            lambda_water_kwh_per_m3_k: Water heat-capacity constant ``λ`` [kWh/(m³·K)].

        Returns:
            1-D NumPy array with one ``V_tap`` estimate per requested timestamp, or
            ``None`` when insufficient history is available.
        """
        if not horizon_valid_at_utc:
            raise ValueError("horizon_valid_at_utc must not be empty.")
        physical_signature = self._physical_signature(
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
            r_loss_k_per_kw=r_loss_k_per_kw,
            lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
            top_temperature_bias_c=top_temperature_bias_c,
            bottom_temperature_bias_c=bottom_temperature_bias_c,
            boiler_ambient_bias_c=boiler_ambient_bias_c,
        )
        cached_profile = self._load_cached_artifact(repository, physical_signature=physical_signature)
        if cached_profile.profile is not None:
            return self._forecast_from_profile(
                profile=cached_profile.profile,
                horizon_valid_at_utc=horizon_valid_at_utc,
            )

        return self._predict_from_history(
            repository=repository,
            horizon_valid_at_utc=horizon_valid_at_utc,
            c_top_kwh_per_k=c_top_kwh_per_k,
            c_bot_kwh_per_k=c_bot_kwh_per_k,
            r_loss_k_per_kw=r_loss_k_per_kw,
            lambda_water_kwh_per_m3_k=lambda_water_kwh_per_m3_k,
            top_temperature_bias_c=top_temperature_bias_c,
            bottom_temperature_bias_c=bottom_temperature_bias_c,
            boiler_ambient_bias_c=boiler_ambient_bias_c,
        )

