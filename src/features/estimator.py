import logging

logger = logging.getLogger(__name__)


class DisturbanceEstimator:
    def __init__(
        self,
        alpha: float = 0.7,
        min: float = 0.0,
        max: float | None = None,
    ) -> None:
        self.alpha = alpha
        self.min = min
        self.max = max

    def estimate(
        self,
        forecast: list[float],
        actual_now: float | None,
    ) -> list[float]:
        if not forecast or actual_now is None:
            return forecast

        forecast_now = forecast[0]

        if forecast_now < self.min and actual_now < self.min:
            return forecast

        bias = actual_now - forecast_now

        trajectory: list[float] = []
        for k, base_val in enumerate(forecast):
            decay = self.alpha**k
            adjusted = base_val + (bias * decay)
            clamped = max(
                0.0, min(self.max, adjusted) if self.max is not None else adjusted
            )
            trajectory.append(clamped)

        logger.debug(
            "Disturbance estimated: actual=%.1f forecast=%.1f bias=%.1f (alpha=%.2f)",
            actual_now,
            forecast_now,
            bias,
            self.alpha,
        )

        return trajectory
