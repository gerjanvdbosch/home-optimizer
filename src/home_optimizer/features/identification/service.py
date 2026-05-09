from __future__ import annotations

from datetime import date, datetime, time, timedelta

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain import (
    COMPRESSOR_FREQUENCY,
    DHW_TARGET_MAX_TEMPERATURE,
    DHW_TARGET_MIN_TEMPERATURE,
    DHW_TARGET_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_ELECTRIC_POWER,
    HP_ELECTRIC_TOTAL_KWH,
    HP_MODE,
    OUTDOOR_TEMPERATURE,
    P1_EXPORT_TOTAL_KWH,
    P1_IMPORT_TOTAL_KWH,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    PV_TOTAL_KWH,
    ROOM_TARGET_MAX_TEMPERATURE,
    ROOM_TARGET_MIN_TEMPERATURE,
    ROOM_TARGET_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    BaselineKpiSummary,
    DailyKpis,
    FixedPricing,
    NumericSeries,
    TextSeries,
    build_daily_price_series,
    build_daily_target_band_series,
    compute_baseline_kpi_summary,
    compute_daily_kpis,
)
from home_optimizer.domain.pricing import (
    DEFAULT_DYNAMIC_PRICE_SOURCE,
    DEFAULT_FIXED_PRICE_SOURCE,
)
from home_optimizer.domain.time import current_local_timezone
from home_optimizer.features.dataset import MpcDatasetService
from home_optimizer.features.dataset.ports import DatasetSampleFrameReader
from home_optimizer.features.identification.models import (
    IdentificationDataset,
    IdentificationDatasetRow,
    IdentificationDatasetSummary,
)
from home_optimizer.features.identification.ports import KpiDataReader


def _series_by_name(series_list: list[NumericSeries]) -> dict[str, NumericSeries]:
    return {series.name: series for series in series_list}


def _text_series_by_name(series_list: list[TextSeries]) -> dict[str, TextSeries]:
    return {series.name: series for series in series_list}


def _resolve_price_series(
    reader: KpiDataReader,
    settings: AppSettings,
    *,
    start_time: datetime,
    end_time: datetime,
) -> NumericSeries:
    fetched_series = reader.read_electricity_price_series(
        start_time=start_time,
        end_time=end_time,
        source=(
            DEFAULT_DYNAMIC_PRICE_SOURCE
            if settings.electricity_pricing.mode == "dynamic"
            else DEFAULT_FIXED_PRICE_SOURCE
        ),
    )
    if fetched_series.points:
        return fetched_series

    if isinstance(settings.electricity_pricing, FixedPricing):
        return build_daily_price_series(
            settings.electricity_pricing,
            start_time=start_time,
            end_time=end_time,
        )

    return NumericSeries(name="electricity_price", unit="EUR/kWh", points=[])


class DailyKpiService:
    def __init__(self, reader: KpiDataReader, settings: AppSettings) -> None:
        self.reader = reader
        self.settings = settings

    def get_day_kpis(self, day: date) -> DailyKpis:
        local_timezone = current_local_timezone()
        start_time = datetime.combine(day, time.min, tzinfo=local_timezone)
        end_time = start_time + timedelta(days=1)

        series = self.reader.read_series(
            names=[
                ROOM_TEMPERATURE,
                THERMOSTAT_SETPOINT,
                COMPRESSOR_FREQUENCY,
                HP_ELECTRIC_POWER,
                HP_ELECTRIC_TOTAL_KWH,
                P1_NET_POWER,
                P1_IMPORT_TOTAL_KWH,
                P1_EXPORT_TOTAL_KWH,
                PV_OUTPUT_POWER,
                PV_TOTAL_KWH,
                OUTDOOR_TEMPERATURE,
                DHW_TOP_TEMPERATURE,
            ],
            start_time=start_time,
            end_time=end_time,
        )
        shutter_series = self.reader.read_series(
            names=[SHUTTER_LIVING_ROOM],
            start_time=start_time,
            end_time=end_time,
        )
        forecast_series = self.reader.read_forecast_series(
            names=[GTI_LIVING_ROOM_WINDOWS],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = self.reader.read_text_series(
            names=[HP_MODE],
            start_time=start_time,
            end_time=end_time,
        )
        series_by_name = _series_by_name(series)
        shutter_by_name = _series_by_name(shutter_series)
        forecast_by_name = _series_by_name(forecast_series)
        text_by_name = _text_series_by_name(text_series)

        room_target, room_target_min, room_target_max = build_daily_target_band_series(
            self.settings.room_target,
            start_time=start_time,
            end_time=end_time,
            target_name=ROOM_TARGET_TEMPERATURE,
            minimum_name=ROOM_TARGET_MIN_TEMPERATURE,
            maximum_name=ROOM_TARGET_MAX_TEMPERATURE,
            interval_minutes=15,
        )
        _, dhw_target_min, _ = build_daily_target_band_series(
            self.settings.dhw_target,
            start_time=start_time,
            end_time=end_time,
            target_name=DHW_TARGET_TEMPERATURE,
            minimum_name=DHW_TARGET_MIN_TEMPERATURE,
            maximum_name=DHW_TARGET_MAX_TEMPERATURE,
            interval_minutes=15,
        )

        feed_in_tariff = (
            self.settings.electricity_pricing.feed_in_tariff
            if isinstance(self.settings.electricity_pricing, FixedPricing)
            else 0.0
        )
        price_series = _resolve_price_series(
            self.reader,
            self.settings,
            start_time=start_time,
            end_time=end_time,
        )

        return compute_daily_kpis(
            room_temperature=series_by_name.get(ROOM_TEMPERATURE),
            room_target=room_target,
            room_target_min=room_target_min,
            room_target_max=room_target_max,
            thermostat_setpoint=series_by_name.get(THERMOSTAT_SETPOINT),
            compressor_frequency=series_by_name.get(COMPRESSOR_FREQUENCY),
            hp_mode=text_by_name.get(HP_MODE),
            hp_electric_power=series_by_name.get(HP_ELECTRIC_POWER),
            hp_electric_total_kwh=series_by_name.get(HP_ELECTRIC_TOTAL_KWH),
            net_power=series_by_name.get(P1_NET_POWER),
            import_total_kwh=series_by_name.get(P1_IMPORT_TOTAL_KWH),
            export_total_kwh=series_by_name.get(P1_EXPORT_TOTAL_KWH),
            pv_output_power=series_by_name.get(PV_OUTPUT_POWER),
            pv_total_kwh=series_by_name.get(PV_TOTAL_KWH),
            solar_irradiance=forecast_by_name.get(GTI_LIVING_ROOM_WINDOWS),
            shutter_open_pct=shutter_by_name.get(SHUTTER_LIVING_ROOM),
            outdoor_temperature=series_by_name.get(OUTDOOR_TEMPERATURE),
            dhw_top_temperature=series_by_name.get(DHW_TOP_TEMPERATURE),
            dhw_target_min=dhw_target_min,
            electricity_price=price_series,
            start_time=start_time,
            end_time=end_time,
            feed_in_tariff=feed_in_tariff,
        )

    def get_baseline_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> BaselineKpiSummary:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        day_count = (end_date - start_date).days + 1
        daily_kpis = [
            self.get_day_kpis(start_date + timedelta(days=day_offset))
            for day_offset in range(day_count)
        ]
        return compute_baseline_kpi_summary(daily_kpis)


class IdentificationDatasetService:
    def __init__(self, samples_reader: DatasetSampleFrameReader, settings: AppSettings) -> None:
        self.dataset_service = MpcDatasetService(samples_reader, settings)

    def build_dataset(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15,
    ) -> IdentificationDataset:
        dataset = self.dataset_service.build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        return IdentificationDataset(
            interval_minutes=dataset.interval_minutes,
            start_time_utc=dataset.start_time_utc,
            end_time_utc=dataset.end_time_utc,
            rows=[
                IdentificationDatasetRow.model_validate(row.model_dump())
                for row in dataset.rows
            ],
        )

    def summarize_dataset(
        self,
        dataset: IdentificationDataset,
    ) -> IdentificationDatasetSummary:
        summary = self.dataset_service.summarize_dataset(dataset)
        return IdentificationDatasetSummary(
            total_rows=summary.total_rows,
            mode_space_rows=summary.mode_space_rows,
            mode_dhw_rows=summary.mode_dhw_rows,
            mode_off_rows=summary.mode_off_rows,
            defrost_rows=summary.defrost_rows,
            booster_rows=summary.booster_rows,
            valid_room_rows=summary.valid_room_rows,
            valid_dhw_rows=summary.valid_dhw_rows,
            valid_cop_rows=summary.valid_cop_rows,
            exclusion_reason_counts=summary.exclusion_reason_counts,
        )
