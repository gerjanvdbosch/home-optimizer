from datetime import datetime, timezone

from domain.models import OptimizerState, SolarForecastState, UpdateRequest
from domain.parser import parse_solar_forecast
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.storage import JsonStorage


class StateService:
    def __init__(
        self,
        influx: InfluxDatabase,
        resolver: InfluxSensorResolver,
        storage: JsonStorage,
    ):
        self.influx = influx
        self.resolver = resolver
        self.storage = storage

    def load(self) -> OptimizerState:
        return OptimizerState(**self.storage.load())

    def update(
        self,
        request: UpdateRequest,
    ) -> None:
        now = datetime.now(timezone.utc)

        forecast = {}

        for name, sensor in request.solar_forecast.items():
            influx_sensor = self.resolver.resolve(sensor)

            point = self.influx.find(
                measurement=influx_sensor.measurement,
                entity_id=influx_sensor.entity_id,
                field=influx_sensor.field,
            )

            forecast[name] = parse_solar_forecast(point)

        state = OptimizerState(
            updated=now,
            solar_forecast=SolarForecastState(**forecast),
        )

        self.storage.save(state.model_dump())
