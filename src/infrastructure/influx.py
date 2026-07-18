from datetime import datetime, timezone
from typing import cast

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet

from domain.models import InfluxSensor, SensorReferenceRequest, Settings, TimeSeriesPoint


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

    # def query_series(
    #     self, measurement: str, entity_id: str, field: str, start: datetime, end: datetime
    # ) -> list[TimeSeriesPoint]:
    #     start = start.astimezone(timezone.utc)
    #     end = end.astimezone(timezone.utc)
    #
    #     query = f"""
    #            SELECT {field} AS value
    #            FROM "{measurement}"
    #            WHERE
    #                "entity_id" = '{entity_id}'
    #                AND time >= '{start.isoformat()}'
    #                AND time < '{end.isoformat()}'
    #            fill(null)
    #        """
    #
    #     result = self.query(query)
    #
    #     return [TimeSeriesPoint(**point) for point in result.get_points()]

    def query_last(
        self,
        measurement: str,
        entity_id: str,
        field: str,
    ) -> TimeSeriesPoint | None:
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

        return TimeSeriesPoint(**points[0])


class InfluxSensorResolver:
    def __init__(self, db: InfluxDatabase):
        self.db = db

    def resolve(
        self,
        sensor: SensorReferenceRequest,
    ) -> InfluxSensor:
        measurements = self.db.query("SHOW MEASUREMENTS")

        entity_id = sensor.entity_id.removeprefix("sensor.")

        for measurement in measurements.get_points():
            name = measurement["name"]

            fields = self.db.query(f'SHOW FIELD KEYS FROM "{name}"')

            for field in fields.get_points():
                field_name = field["fieldKey"]

                if field_name.startswith(sensor.attribute):
                    return InfluxSensor(
                        measurement=name,
                        entity_id=entity_id,
                        field=field_name,
                    )

        raise ValueError(f"Sensor not found: {sensor.entity_id}.{sensor.attribute}")
