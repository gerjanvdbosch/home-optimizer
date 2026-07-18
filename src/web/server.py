from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.settings import load_settings
from domain.models import UpdateRequest
from infrastructure.influx import InfluxDatabase

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

settings = load_settings()
db = InfluxDatabase(settings)

templates = Jinja2Templates(
    directory=BASE_DIR / "templates",
)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={},
    )


@app.post("/api/update")
async def update(request: UpdateRequest):
    entity, attribute = request.solar_forecast.p10

    return {"ok"}
