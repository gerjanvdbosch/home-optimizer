from app.logger import configure_logger
from app.settings import load_settings
from app.state_service import StateService
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.storage import JsonStorage


class Container:
    def __init__(self):
        settings = load_settings()

        configure_logger(settings.log_level)

        self.influx = InfluxDatabase(settings)
        self.resolver = InfluxSensorResolver(self.influx)

        self.state_service = StateService(
            influx=self.influx,
            resolver=self.resolver,
            # storage=MemoryStorage(),
            storage=JsonStorage(settings.data_path / "state.json"),
        )
