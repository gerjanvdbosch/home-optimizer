from datetime import datetime, timedelta, timezone

from domain.models import OptimizerState, SolarForecastState


class StateService:
    def __init__(
        self,
        influx,
        resolver,
        storage,
    ):
        self.influx = influx
        self.resolver = resolver
        self.storage = storage

    def update(self, request) -> None:
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=1)

        forecast = {}

        for name, sensor in request.solar_forecast.model_dump().items():
            measurement, entity_id, field = self.resolver.resolve(sensor)

            forecast[name] = [
                point
                for point in self.influx.query_series(
                    measurement=measurement,
                    entity_id=entity_id,
                    field=field,
                    start=start,
                    end=end,
                )
            ]

        state = OptimizerState(
            updated=start,
            solar_forecast=SolarForecastState(**forecast),
        )

        self.storage.save(state.model_dump())
