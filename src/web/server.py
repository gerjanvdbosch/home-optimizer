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
from domain.models.config import BacktestParams, Config, FitParams, TuneParams
from domain.models.worker import Job, JobType
from web.chart import backtest_chart, solar_forecast_chart

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = create_container()

    app.state.manager = Manager()

    app.state.worker = Worker(
        create_container,
        app.state.manager,
    )

    app.state.worker.start()

    yield

    app.state.worker.stop()
    app.state.manager.shutdown()


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
    container = request.app.state.container

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
async def status(request: Request):
    return list(request.app.state.worker.jobs.values())


@app.get("/api/state")
async def state(request: Request):
    return request.app.state.container.state_manager.load()


@app.post("/api/update")
async def update(config: Config, request: Request):
    container = request.app.state.container

    container.config_repository.save(config)
    container.state_manager.update(config)
    container.backtest_repository.clear()

    return {"ok": True}


@app.post("/api/fit")
async def fit(request: Request, params: FitParams):
    job = Job(
        id=uuid.uuid4().hex,
        type=JobType.FIT,
        params=params,
    )

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


# @app.post("/api/predict")
# async def predict(request: Request, params: FitParams):
#     return {"ok": True}


@app.post("/api/backtest")
async def backtest(request: Request, params: BacktestParams):
    job = Job(
        id=uuid.uuid4().hex,
        type=JobType.BACKTEST,
        params=params,
    )

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/tune")
async def tune(request: Request, params: TuneParams):
    job = Job(
        id=uuid.uuid4().hex,
        type=JobType.TUNE,
        params=params,
    )

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }
