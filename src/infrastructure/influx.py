from datetime import datetime
from typing import Any, cast

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet

from domain.models import InfluxSensor, Resample, SensorReferenceRequest, Settings


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
    ) -> dict[str, Any] | None:
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

        return points[0]

    def find_series(
        self,
        measurement: str,
        entity_id: str,
        field: str,
        start: datetime,
        end: datetime,
        resample: Resample | None = None,
    ) -> list[dict[str, Any]]:
        if resample:
            select = f'{resample.aggregation}("{field}")'
        else:
            select = f'"{field}"'

        query = f"""
        SELECT {select} AS value
        FROM "{measurement}"
        WHERE
            "entity_id" = '{entity_id}'
            AND time >= '{start.isoformat()}'
            AND time < '{end.isoformat()}'
        """

        if resample:
            query += f"""
        GROUP BY time({resample.interval})
        fill(previous)
        """
        print(query)
        result = self.query(query)

        return list(result.get_points())


class InfluxSensorResolver:
    def __init__(self, db: InfluxDatabase):
        self.db = db
        self.cache: dict[str, InfluxSensor] = {}
        self.schema: list[InfluxSensor] = []
        self.schema_loaded = False

    def load_schema(self) -> None:
        if self.schema_loaded:
            return

        measurements = self.db.query("SHOW MEASUREMENTS")

        for measurement in measurements.get_points():
            name = measurement["name"]

            fields = self.db.query(f'SHOW FIELD KEYS FROM "{name}"')

            for field in fields.get_points():
                self.schema.append(
                    InfluxSensor(
                        measurement=name,
                        entity_id="",
                        field=field["fieldKey"],
                    )
                )

        self.schema_loaded = True

    def resolve(
        self,
        sensor: SensorReferenceRequest,
    ) -> InfluxSensor:
        self.load_schema()

        entity_id = sensor.entity_id.removeprefix("sensor.")
        cache_key = f"{entity_id}.{sensor.attribute}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        for influx_sensor in self.schema:
            field_name = influx_sensor.field

            if not (
                field_name == sensor.attribute or field_name.startswith(sensor.attribute + "_")
            ):
                continue

            query = f"""
            SELECT "{field_name}"
            FROM "{influx_sensor.measurement}"
            WHERE "entity_id" = '{entity_id}'
            LIMIT 1
            """

            if list(self.db.query(query).get_points()):
                resolved = InfluxSensor(
                    measurement=influx_sensor.measurement,
                    entity_id=entity_id,
                    field=field_name,
                )

                self.cache[cache_key] = resolved

                return resolved

        raise ValueError(f"Sensor not found: {sensor.entity_id}.{sensor.attribute}")
