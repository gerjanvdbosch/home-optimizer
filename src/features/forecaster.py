from pathlib import Path

import joblib


class SolarForecaster:
    def __init__(self):
        # self.forecaster =
        pass

    def fit(self):
        pass

    def backtest(self):
        pass

    def predict(self):
        pass

    def save(self, path: str | Path):
        joblib.dump(
            self,
            path,
        )

    def load(self, path: str | Path):
        loaded = joblib.load(path)
