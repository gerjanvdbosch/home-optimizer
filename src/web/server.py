from fastapi import FastAPI

app = FastAPI(
    title="Home Optimizer",
    version="0.11",
)

@app.get("/health")
async def health():
    return {"status": "ok"}
