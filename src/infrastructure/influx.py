from typing import cast

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet

from app.settings import Settings


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

    def get_entity_values(
        self,
        measurement: str,
        entity_id: str,
        limit: int = 10,
    ) -> list[dict]:
        query = (
            f'SELECT "value" '
            f'FROM "{measurement}" '
            f"WHERE \"entity_id\" = '{entity_id}' "
            f"LIMIT {limit}"
        )

        return list(self.query(query).get_points())
