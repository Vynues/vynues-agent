import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ResearchRequest, ResearchResponse
from app.pipeline import run_vynues_pipeline

app = FastAPI(
    title="Vynues API",
    description="Venue matching pipeline powered by the Yelp dataset and GPT-4o.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    try:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ResearchResponse(**result)
