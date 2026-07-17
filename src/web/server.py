from fastapi import FastAPI

from app.settings import load_settings
from infrastructure.influx import InfluxDatabase

app = FastAPI()
settings = load_settings()
db = InfluxDatabase(settings)


@app.get("/")
async def dashboard():
    return {"ok"}


@app.get("/api/update")
async def update():
    points = db.get_entity_values(measurement="W", entity_id="pv_output", limit=10)

    for point in points:
        print(point)

    return {"ok"}
