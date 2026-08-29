import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DisturbanceEstimator:
    alpha: float = 0.7
    max_power_w: float = 2000.0
    min_threshold_w: float = 50.0

    def estimate(
        self,
        forecast: list[float],
        recent_actuals: list[float],
    ) -> list[float]:
        if not forecast or not recent_actuals:
            return forecast

        actual_now = recent_actuals[-1]
        forecast_now = forecast[0]

        if forecast_now < self.min_threshold_w and actual_now < self.min_threshold_w:
            return forecast

        n = min(3, len(recent_actuals))
        history_actual = recent_actuals[-n:]

        raw_weights = [0.10, 0.30, 0.60][-n:]
        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]

        filtered_actual = sum(
            a * w
            for a, w in zip(
                history_actual,
                weights,
                strict=False,
            )
        )

        bias = filtered_actual - forecast_now

        trajectory: list[float] = []
        for k, base_val in enumerate(forecast):
            decay = self.alpha**k
            adjusted = base_val + (bias * decay)
            clamped = max(0.0, min(self.max_power_w, adjusted))
            trajectory.append(clamped)

        logger.debug(
            "Disturbance estimated: actual_now=%.1f filtered=%.1f forecast=%.1f bias=%.1f (alpha=%.2f)",
            actual_now,
            filtered_actual,
            forecast_now,
            bias,
            self.alpha,
        )

        return trajectory
