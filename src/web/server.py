from contextlib import asynccontextmanager
from multiprocessing import Manager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.bootstrap import create_container
from app.worker import Worker
from domain.models import (
    BacktestConfig,
    Config,
    FitConfig,
    Job,
    JobType,
    OptimizeConfig,
    PredictConfig,
    TuneConfig,
)
from web.chart import backtest_chart, dashboard_chart, solar_chart

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

    try:
        config = container.config_repository.load()
    except Exception:
        return templates.TemplateResponse(request=request, name="setup.html")

    state = container.state_manager.load()
    backtest = container.backtest_repository.load()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "dashboard_chart": dashboard_chart(state),
            "solar_chart": solar_chart(
                state,
                capacity=config.solar.capacity,
                efficiency=config.solar.efficiency,
            ),
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
async def update(request: Request, config: Config):
    job = Job(type=JobType.UPDATE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/fit")
async def fit(request: Request, config: FitConfig):
    job = Job(type=JobType.FIT, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/predict")
async def predict(request: Request, config: PredictConfig):
    job = Job(type=JobType.PREDICT, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/backtest")
async def backtest(request: Request, config: BacktestConfig):
    job = Job(type=JobType.BACKTEST, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/tune")
async def tune(request: Request, config: TuneConfig):
    job = Job(type=JobType.TUNE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }


@app.post("/api/optimize")
async def optimize(request: Request, config: OptimizeConfig):
    job = Job(type=JobType.OPTIMIZE, config=config)

    request.app.state.worker.submit(job)

    return {
        "job_id": job.id,
        "state": "queued",
    }
