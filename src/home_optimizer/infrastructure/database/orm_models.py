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


class IdentifiedModelRecord(Base):
    __tablename__ = "identified_models"

    model_kind: Mapped[str] = mapped_column(String, primary_key=True)
    trained_at_utc: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    training_start_time_utc: Mapped[str] = mapped_column(String, nullable=False)
    training_end_time_utc: Mapped[str] = mapped_column(String, nullable=False)
    interval_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    train_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    coefficients_json: Mapped[str] = mapped_column(Text, nullable=False)
    intercept: Mapped[float] = mapped_column(Float, nullable=False)
    train_rmse: Mapped[float] = mapped_column(Float, nullable=False)
    test_rmse: Mapped[float] = mapped_column(Float, nullable=False)
    test_rmse_recursive: Mapped[float] = mapped_column(Float, nullable=False)
    target_name: Mapped[str] = mapped_column(String, nullable=False)


Index("idx_identified_models_kind_trained_at", IdentifiedModelRecord.model_kind, IdentifiedModelRecord.trained_at_utc)
