from datetime import datetime, timedelta, timezone

from domain.models.config import AppConfig, TrainRequest
from domain.models.interface import Forecaster
from features.dataset import DatasetLoader


class Trainer:

    def __init__(self, loader: DatasetLoader, forecasters: list[Forecaster]):
        self.loader = loader
        self.forecasters = forecasters

    def train(self, config: AppConfig, request: TrainRequest):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=request.days)

        for forecaster in self.forecasters:
            dataset = forecaster.dataset(config)

            df = self.loader.load(dataset, start, end)

            print(df)

            # forecaster.fit(df)

    # def backtest(self, name, config):

    # dataset = model.forecaster.dataset(home_config)
    #
    # df = self.loader.load(
    #     dataset,
    #     start,
    #     end,
    # )
    #
    # return model.forecaster.backtest(df)


#         dhw = DhwForecaster(lags=96)
#
#         dhw.fit(
#             temperature=df["temp"],
#             exog=df[
#                 [
#                     "mode",
#                 ]
#             ],
#         )
#
#         dhw.tune(
#             temperature=df["temp"],
#             exog=df[
#                 [
#                     "mode",
#                 ]
#             ],
#         )
#
#         print(dhw.best_params)
#         print(dhw.best_score)
#
#         metrics, predictions = dhw.backtest(
#             temperature=df["temp"],
#             exog=df[
#                 [
#                     "mode",
#                 ]
#             ],
#         )
#
#         print(metrics)
#
#         future_index = pd.date_range(
#             start=df.index[-1] + pd.Timedelta("15min"),
#             periods=24,
#             freq="15min",
#         )
#
#         future_exog = pd.DataFrame(
#             {
#                 "mode": ["Uit"] * 24,
#             },
#             index=future_index,
#         )
#
#         future = dhw.predict(
#             steps=24,
#             exog=future_exog,
#         )
#
#         print(future)
