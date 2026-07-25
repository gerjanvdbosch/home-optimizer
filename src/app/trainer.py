from domain.models.config import TrainConfig
from domain.models.interface import Storage


class Trainer:
    def __init__(self, forecasters, storage: Storage):
        self.forecasters = forecasters
        self.storage = storage

    def train(self, config: TrainConfig):
        # dataset = loader.load(registration.dataset)
        #
        # features = generator.transform(dataset)
        #
        # registration.forecaster.fit(features)
        return

    # def backtest(self):
    #     return
    #
    # def tune(self):
    #     return
