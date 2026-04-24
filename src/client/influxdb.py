from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules


class InfluxStore:
    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        measurement: str,
    ) -> None:
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.measurement = measurement

        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org,
        )

        self.write_api = self.client.write_api(
            write_options=SYNCHRONOUS
        )

        self.query_api = self.client.query_api()
        self.buckets_api = self.client.buckets_api()
        self.orgs_api = self.client.organizations_api()

    def close(self) -> None:
        self.client.close()

    def ensure_bucket(
        self,
        retention_days: int = 730,
    ) -> None:
        existing = self.buckets_api.find_bucket_by_name(
            self.bucket
        )

        if existing:
            print(f"Bucket already exists: {self.bucket}")
            return

        org = self.orgs_api.find_organizations(
            org=self.org
        )[0]

        retention = BucketRetentionRules(
            type="expire",
            every_seconds=retention_days * 24 * 60 * 60,
        )

        self.buckets_api.create_bucket(
            bucket_name=self.bucket,
            org_id=org.id,
            retention_rules=retention,
        )

        print(
            f"Created bucket: {self.bucket} "
            f"(retention={retention_days}d)"
        )

    def _build_base_point(
        self,
        name: str,
        entity_id: str,
        source: str | None = None,
        category: str | None = None,
        unit: str | None = None,
        timestamp: datetime | None = None,
    ) -> Point:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        point = (
            Point(self.measurement)
            .tag("name", name)
            .tag("entity_id", entity_id)
            .time(timestamp, WritePrecision.NS)
        )

        if source:
            point = point.tag("source", source)

        if category:
            point = point.tag("category", category)

        if unit:
            point = point.tag("unit", unit)

        return point

    def write_sensor(
        self,
        name: str,
        entity_id: str,
        value: Any,
        category: str | None = None,
        unit: str | None = None,
        source: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        if value is None:
            return

        point = self._build_base_point(
            name=name,
            entity_id=entity_id,
            source=source,
            category=category,
            unit=unit,
            timestamp=timestamp,
        )

        if isinstance(value, bool):
            point = point.field("value_bool", value)

        elif isinstance(value, (int, float)):
            point = point.field("value", float(value))

        else:
            point = point.field("value_str", str(value))

        self.write_api.write(
            bucket=self.bucket,
            org=self.org,
            record=point,
        )

    def query_last_value(
        self,
        sensor_name: str,
        hours: int = 24,
    ) -> list[dict[str, Any]]:
        flux = f"""
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
          |> filter(fn: (r) => r["name"] == "{sensor_name}")
          |> yield(name: "result")
        """

        result = self.query_api.query(org=self.org, query=flux)

        rows = []

        for table in result:
            for record in table.records:
                rows.append(
                    {
                        "time": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }
                )

        return rows
