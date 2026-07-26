from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.bootstrap import create_container
from domain.models.config import AppConfig, TrainRequest
from web.chart import solar_forecast_chart

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

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "solar_chart": solar_forecast_chart(state),
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


@app.post("/api/train")
async def train(request: TrainRequest):
    config = container.config_repository.load()
    container.trainer.train(config, request)

    return {"ok": True}


# @app.post("/api/backtest")
# async def backtest(request: TrainRequest):
#     container.trainer.train(config)
#
#     return {"ok": True}
#
#
# @app.post("/api/tune")
# async def backtest(request: TrainRequest):
#     container.trainer.tune(config)
#
#     return {"ok": True}
