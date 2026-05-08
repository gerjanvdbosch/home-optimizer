from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Sample1m(Base):
    __tablename__ = "samples_1m"

    timestamp_minute_utc: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    mean_real: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_real: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_real: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_real: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_bool: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)


Index("idx_samples_1m_name_time", Sample1m.name, Sample1m.timestamp_minute_utc)
Index("idx_samples_1m_category_time", Sample1m.category, Sample1m.timestamp_minute_utc)
Index("idx_samples_1m_time", Sample1m.timestamp_minute_utc)


class ForecastValue(Base):
    __tablename__ = "forecast_values"

    created_at_utc: Mapped[str] = mapped_column(String, primary_key=True)
    forecast_time_utc: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)


Index("idx_forecast_values_name_time", ForecastValue.name, ForecastValue.forecast_time_utc)
Index("idx_forecast_values_created", ForecastValue.created_at_utc)



class ElectricityPriceIntervalValue(Base):
    __tablename__ = "electricity_price_intervals"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    start_time_utc: Mapped[str] = mapped_column(String, primary_key=True)
    end_time_utc: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    unit: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)


Index(
    "idx_electricity_price_intervals_name_start",
    ElectricityPriceIntervalValue.name,
    ElectricityPriceIntervalValue.start_time_utc,
)
Index(
    "idx_electricity_price_intervals_source_end",
    ElectricityPriceIntervalValue.source,
    ElectricityPriceIntervalValue.end_time_utc,
)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at_utc: Mapped[str] = mapped_column(String, nullable=False)
    trained_from_utc: Mapped[str] = mapped_column(String, nullable=False)
    trained_to_utc: Mapped[str] = mapped_column(String, nullable=False)
    interval_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_mae_1h_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_mae_6h_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_mae_12h_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_mae_24h_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_bias_6h_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_p95_12h_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_json: Mapped[str] = mapped_column(Text, nullable=False)
    validation_report_json: Mapped[str | None] = mapped_column(Text, nullable=True)


Index("idx_model_versions_type_created", ModelVersion.model_type, ModelVersion.created_at_utc)
Index("idx_model_versions_type_active", ModelVersion.model_type, ModelVersion.is_active)

