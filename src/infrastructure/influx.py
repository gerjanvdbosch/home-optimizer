from typing import cast

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet


class InfluxDatabase:
    def __init__(self):
        self.client = InfluxDBClient(
            host="homeassistant.local",
            port=8086,
            username="home_assistant",
            password="home_assistant",
            database="home_assistant",
        )

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

        result = cast(ResultSet, self.client.query(query))
        return list(result.get_points())
