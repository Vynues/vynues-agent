import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import ResearchRequest, ResearchResponse
from app.pipeline import run_vynues_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/research")
async def research(req: ResearchRequest):
    result = await run_vynues_pipeline(
        event_type=req.event_type,
        event_date=req.event_date,
        budget=req.budget,
        capacity=req.capacity,
        location=req.location,
        decor_style=req.decor_style,
        cuisine_type=req.cuisine_type,
        page=req.page,
    )
    return result
