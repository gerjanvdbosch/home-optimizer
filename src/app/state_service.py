from datetime import datetime, timezone

from domain.models import OptimizerState, Resample, SolarForecastState, Storage, UpdateRequest
from domain.parser import parse_pv_production, parse_solar_forecast
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver


class StateService:
    def __init__(
        self,
        influx: InfluxDatabase,
        resolver: InfluxSensorResolver,
        storage: Storage,
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

        influx_sensor = self.resolver.resolve(request.pv_production)

        start = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        production_points = self.influx.find_series(
            measurement=influx_sensor.measurement,
            entity_id=influx_sensor.entity_id,
            field=influx_sensor.field,
            start=start,
            end=now,
            resample=Resample(
                aggregation="mean",
                interval="5m",
            ),
        )

        state = OptimizerState(
            updated=now,
            solar_forecast=SolarForecastState(**forecast),
            pv_production=parse_pv_production(production_points),
        )

        self.storage.save(state.model_dump())
