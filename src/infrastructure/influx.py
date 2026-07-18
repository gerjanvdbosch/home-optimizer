from typing import cast

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet

from domain.models import InfluxPoint, InfluxSensor, SensorReferenceRequest, Settings


class InfluxDatabase:
    def __init__(self, settings: Settings):
        self.client = InfluxDBClient(
            host=settings.influx_host,
            port=settings.influx_port,
            username=settings.influx_username,
            password=settings.influx_password,
            database=settings.influx_database,
        )

    def query(self, query: str) -> ResultSet:
        return cast(ResultSet, self.client.query(query))

    def find(
        self,
        measurement: str,
        entity_id: str,
        field: str,
    ) -> InfluxPoint | None:
        query = f"""
        SELECT "{field}" AS value
        FROM "{measurement}"
        WHERE "entity_id" = '{entity_id}'
        ORDER BY time DESC
        LIMIT 1
        """

        result = self.query(query)

        points = list(result.get_points())
        if not points:
            return None

        return InfluxPoint(**points[0])


class InfluxSensorResolver:
    def __init__(self, db: InfluxDatabase):
        self.db = db
        self.cache: dict[str, InfluxSensor] = {}

    def resolve(
        self,
        sensor: SensorReferenceRequest,
    ) -> InfluxSensor:
        measurements = self.db.query("SHOW MEASUREMENTS")

        entity_id = sensor.entity_id.removeprefix("sensor.")

        cache_key = f"{entity_id}.{sensor.attribute}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        for measurement in measurements.get_points():
            name = measurement["name"]

            fields = self.db.query(f'SHOW FIELD KEYS FROM "{name}"')

            for field in fields.get_points():
                field_name = field["fieldKey"]

                if field_name.startswith(sensor.attribute + "_"):
                    influx_sensor = InfluxSensor(
                        measurement=name,
                        entity_id=entity_id,
                        field=field_name,
                    )

                    self.cache[cache_key] = influx_sensor

                    return influx_sensor

        raise ValueError(f"Sensor not found: {sensor.entity_id}.{sensor.attribute}")
