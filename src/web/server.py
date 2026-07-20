from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.container import Container
from domain.models import TrainRequest, UpdateRequest
from web.charts import solar_forecast_chart

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

container = Container()

templates = Jinja2Templates(
    directory=BASE_DIR / "templates",
)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    state = container.state_service.load()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "solar_chart": solar_forecast_chart(state),
        },
    )


@app.get("/api/state")
async def state():
    return container.state_service.load()


@app.post("/api/update")
async def update(request: UpdateRequest):
    container.state_service.update(request)

    return {"ok": True}


@app.post("/api/train")
async def train(request: TrainRequest):
    container.training_service.train(request)

    return {"ok": True}
