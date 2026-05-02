from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from home_optimizer.app.forecast_scheduler import ForecastScheduler
from home_optimizer.app.historical_weather_scheduler import HistoricalWeatherScheduler
from home_optimizer.app.model_training_runner import FullDatasetModelTrainingRunner
from home_optimizer.app.model_training_scheduler import ModelTrainingScheduler
from home_optimizer.app.model_training_service import MultiModelTrainingService
from home_optimizer.app.ports import SensorGateway
from home_optimizer.app.settings import AppSettings
from home_optimizer.app.telemetry_scheduler import TelemetryScheduler
from home_optimizer.domain.sensor_factory import build_sensor_specs
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.features.forecast.service import OpenMeteoForecastService
from home_optimizer.features.backtesting import RoomTemperatureBacktestingService
from home_optimizer.features.history_import.history_import_service import (
    HistoryImportService,
)
from home_optimizer.features.history_import.historical_weather_import_service import (
    HistoricalWeatherImportService,
)
from home_optimizer.features.history_import.weather_import_service import WeatherImportService
from home_optimizer.features.identification.room_temperature import (
    RoomTemperatureModelIdentificationService,
)
from home_optimizer.features.identification.thermal_output import (
    ThermalOutputModelIdentificationService,
)
from home_optimizer.features.mpc import (
    ThermostatSetpointCandidateGenerator,
    ThermostatSetpointMpcEvaluator,
    ThermostatSetpointMpcPlanner,
)
from home_optimizer.features.prediction.service import RoomTemperaturePredictionService
from home_optimizer.features.telemetry.service import TelemetryService
from home_optimizer.infrastructure.database.forecast_repository import ForecastRepository
from home_optimizer.infrastructure.database.historical_weather_repository import (
    HistoricalWeatherRepository,
)
from home_optimizer.infrastructure.database.identified_model_repository import (
    IdentifiedModelRepository,
)
from home_optimizer.infrastructure.database.session import Database
from home_optimizer.infrastructure.database.time_series_read_repository import (
    TimeSeriesReadRepository,
)
from home_optimizer.infrastructure.database.time_series_write_repository import (
    TimeSeriesWriteRepository,
)
from home_optimizer.infrastructure.home_assistant.gateway import HomeAssistantGateway
from home_optimizer.infrastructure.weather.openmeteo import OpenMeteoGateway

GatewayFactory = Callable[[list[SensorSpec]], SensorGateway]


@dataclass
class AppContainer:
    settings: AppSettings
    database: Database
    home_assistant: SensorGateway
    open_meteo: OpenMeteoGateway
    history_import_repository: TimeSeriesWriteRepository
    history_import_service: HistoryImportService
    weather_import_service: WeatherImportService
    historical_weather_repository: HistoricalWeatherRepository
    historical_weather_import_service: HistoricalWeatherImportService
    telemetry_repository: TimeSeriesWriteRepository
    time_series_read_repository: TimeSeriesReadRepository
    identified_model_repository: IdentifiedModelRepository
    identification_service: RoomTemperatureModelIdentificationService
    model_training_service: MultiModelTrainingService
    prediction_service: RoomTemperaturePredictionService
    mpc_planner: ThermostatSetpointMpcPlanner
    backtesting_service: RoomTemperatureBacktestingService
    telemetry_service: TelemetryService
    telemetry_scheduler: TelemetryScheduler
    historical_weather_scheduler: HistoricalWeatherScheduler
    model_training_scheduler: ModelTrainingScheduler
    forecast_repository: ForecastRepository
    forecast_service: OpenMeteoForecastService
    forecast_scheduler: ForecastScheduler

    def close(self) -> None:
        self.home_assistant.close()
        self.open_meteo.close()


def build_container(
    settings: AppSettings,
    gateway_factory: GatewayFactory | None = None,
    history_source: str = "home_assistant_history",
    telemetry_source: str = "home_assistant_telemetry",
) -> AppContainer:
    database = Database(settings.database_path)
    database.init_schema()

    sensor_specs = build_sensor_specs(settings)
    gateway = gateway_factory(sensor_specs) if gateway_factory else HomeAssistantGateway()
    location = gateway.get_location()
    open_meteo = OpenMeteoGateway()
    history_import_repository = TimeSeriesWriteRepository(database, source=history_source)
    telemetry_repository = TimeSeriesWriteRepository(database, source=telemetry_source)
    time_series_read_repository = TimeSeriesReadRepository(database)
    identified_model_repository = IdentifiedModelRepository(database)
    identification_service = RoomTemperatureModelIdentificationService(
        time_series_read_repository,
        model_repository=identified_model_repository,
    )
    thermal_output_identification_service = ThermalOutputModelIdentificationService(
        time_series_read_repository,
        model_repository=identified_model_repository,
    )
    prediction_service = RoomTemperaturePredictionService(
        time_series_read_repository,
        identified_model_repository,
    )
    model_training_service = MultiModelTrainingService(
        thermal_output_identification_service,
        identification_service,
    )
    mpc_planner = ThermostatSetpointMpcPlanner(
        ThermostatSetpointCandidateGenerator(),
        ThermostatSetpointMpcEvaluator(prediction_service),
    )
    backtesting_service = RoomTemperatureBacktestingService(
        time_series_read_repository,
        identified_model_repository,
        prediction_service,
    )
    forecast_repository = ForecastRepository(database)
    historical_weather_repository = HistoricalWeatherRepository(database)
    history_import_service = HistoryImportService(
        gateway=gateway,
        repository=history_import_repository,
        chunk_days=settings.history_import_chunk_days,
    )
    weather_import_service = WeatherImportService(
        gateway=open_meteo,
        location=location,
        repository=forecast_repository,
        pv_tilt=settings.pv_tilt,
        pv_azimuth=settings.pv_azimuth,
        living_room_window_azimuth=settings.living_room_window_azimuth,
        history_days_back=settings.history_import_max_days_back,
    )
    historical_weather_import_service = HistoricalWeatherImportService(
        gateway=open_meteo,
        location=location,
        repository=historical_weather_repository,
        pv_tilt=settings.pv_tilt,
        pv_azimuth=settings.pv_azimuth,
        living_room_window_azimuth=settings.living_room_window_azimuth,
        history_days_back=settings.history_import_max_days_back,
    )
    telemetry_service = TelemetryService(
        gateway=gateway,
        repository=telemetry_repository,
        specs=sensor_specs,
    )
    telemetry_scheduler = TelemetryScheduler(telemetry_service)
    historical_weather_scheduler = HistoricalWeatherScheduler(
        historical_weather_import_service,
    )
    model_training_runner = FullDatasetModelTrainingRunner(
        identification_service,
        time_series_read_repository,
        thermal_output_identification_service=thermal_output_identification_service,
    )
    model_training_scheduler = ModelTrainingScheduler(model_training_runner)
    forecast_service = OpenMeteoForecastService(
        gateway=open_meteo,
        location=location,
        repository=forecast_repository,
        pv_tilt=settings.pv_tilt,
        pv_azimuth=settings.pv_azimuth,
        living_room_window_azimuth=settings.living_room_window_azimuth,
        poll_interval_seconds=settings.forecast_poll_interval_seconds,
    )
    forecast_scheduler = ForecastScheduler(
        forecast_service,
        interval_seconds=settings.forecast_poll_interval_seconds,
    )

    return AppContainer(
        settings=settings,
        database=database,
        home_assistant=gateway,
        open_meteo=open_meteo,
        history_import_repository=history_import_repository,
        history_import_service=history_import_service,
        weather_import_service=weather_import_service,
        historical_weather_repository=historical_weather_repository,
        historical_weather_import_service=historical_weather_import_service,
        telemetry_repository=telemetry_repository,
        time_series_read_repository=time_series_read_repository,
        identified_model_repository=identified_model_repository,
        identification_service=identification_service,
        model_training_service=model_training_service,
        prediction_service=prediction_service,
        mpc_planner=mpc_planner,
        backtesting_service=backtesting_service,
        telemetry_service=telemetry_service,
        telemetry_scheduler=telemetry_scheduler,
        historical_weather_scheduler=historical_weather_scheduler,
        model_training_scheduler=model_training_scheduler,
        forecast_repository=forecast_repository,
        forecast_service=forecast_service,
        forecast_scheduler=forecast_scheduler,
    )
