import uuid
from contextlib import asynccontextmanager
from multiprocessing import Manager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.bootstrap import create_container
from app.worker import Worker
from domain.models.config import AppConfig, BacktestRequest, FitRequest, TuneRequest
from domain.models.worker import Job, JobType
from web.chart import backtest_chart, solar_forecast_chart

BASE_DIR = Path(__file__).resolve().parent


container = create_container()

manager = Manager()

worker = Worker(create_container, manager)


@asynccontextmanager
async def lifespan(app: FastAPI):

    worker.start()

    app.state.worker = worker

    yield

    worker.stop()
    manager.shutdown()


app = FastAPI(
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

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


@app.get("/api/status")
async def status():
    return list(worker.jobs.values())


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
    worker.submit(
        Job(
            id=uuid.uuid4().hex,
            type=JobType.FIT,
            request=request,
        )
    )

    return {"status": "queued"}


# @app.post("/api/predict")
# async def predict(request: FitRequest):
#     return {"ok": True}


@app.post("/api/backtest")
async def backtest(request: BacktestRequest):
    worker.submit(
        Job(
            id=uuid.uuid4().hex,
            type=JobType.BACKTEST,
            request=request,
        )
    )

    return {"status": "queued"}


@app.post("/api/tune")
async def tune(request: TuneRequest):
    worker.submit(
        Job(
            id=uuid.uuid4().hex,
            type=JobType.TUNE,
            request=request,
        )
    )

    return {"status": "queued"}
