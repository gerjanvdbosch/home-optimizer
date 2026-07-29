from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.bootstrap import create_container
from domain.models.config import AppConfig, BacktestRequest, FitRequest, TuneRequest
from web.chart import backtest_chart, solar_forecast_chart

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

container = create_container()

templates = Jinja2Templates(
    directory=BASE_DIR / "templates",
)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    state = container.state_manager.load()
    backtest = container.backtest_repository.load()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "solar_chart": solar_forecast_chart(state),
            "backtest_chart": backtest_chart(backtest),
        },
    )


@app.get("/api/state")
async def state():
    return container.state_manager.load()


@app.post("/api/update")
async def update(config: AppConfig):
    container.config_repository.save(config)
    container.state_manager.update(config)

    return {"ok": True}


@app.post("/api/fit")
async def train(request: FitRequest):
    config = container.config_repository.load()
    container.forecasting.fit(config, request)

    return {"ok": True}


# @app.post("/api/predict")
# async def predict(request: FitRequest):
#     config = container.config_repository.load()
#     container.forecasting.train(config, request)
#
#     return {"ok": True}


@app.post("/api/backtest")
async def backtest(request: BacktestRequest):
    config = container.config_repository.load()

    return container.forecasting.backtest(config, request)


@app.post("/api/tune")
async def tune(request: TuneRequest):
    config = container.config_repository.load()

    study = container.forecasting.tune(config, request)

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
