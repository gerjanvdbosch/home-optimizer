from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.settings import load_settings
from infrastructure.influx import InfluxDatabase

app = FastAPI()
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

settings = load_settings()
db = InfluxDatabase(settings)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={},
    )


@app.get("/api/update")
async def update():
    points = db.get_entity_values(measurement="W", entity_id="pv_output", limit=10)

    for point in points:
        print(point)

    return {"ok"}
