from app.settings import load_settings
from app.state_service import StateService
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver
from infrastructure.storage import JsonStorage


class Container:
    def __init__(self):
        settings = load_settings()

        self.influx = InfluxDatabase(settings)
        self.resolver = InfluxSensorResolver(self.influx)

        self.state_service = StateService(
            influx=self.influx,
            resolver=self.resolver,
            storage=JsonStorage(settings.data_path / "state.json"),
        )
