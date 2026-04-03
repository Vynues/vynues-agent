import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import QueryRequest
from app.pipeline import run_vynues_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_query(query: str) -> dict:
    """Extract structured fields from a natural-language event planning query."""
    q = query.strip()

    # event_type: first token(s) before "in " or first comma
    event_type = "event"
    m = re.match(r'^([^,]+?)\s+in\s+', q, re.IGNORECASE)
    if m:
        event_type = m.group(1).strip()
    else:
        m = re.match(r'^([^,]+)', q)
        if m:
            event_type = m.group(1).strip()

    # location: after "in " until the next comma
    location = ""
    m = re.search(r'\bin\s+([^,]+)', q, re.IGNORECASE)
    if m:
        location = m.group(1).strip()

    # budget: dollar-sign notation ($$) or explicit "budget $..."
    budget = ""
    m = re.search(r'budget\s+(\$+)', q, re.IGNORECASE)
    if m:
        budget = m.group(1)
    else:
        m = re.search(r'(\$[\d,\-–]+(?:\s*[-–]\s*\$[\d,]+)?|\$+)', q)
        if m:
            budget = m.group(1)

    # capacity: "<number> guests"
    capacity = 100
    m = re.search(r'(\d+)\s+guests?', q, re.IGNORECASE)
    if m:
        capacity = int(m.group(1))

    # cuisine_type: "<word> cuisine"
    cuisine_type = ""
    m = re.search(r'([\w/]+)\s+cuisine', q, re.IGNORECASE)
    if m:
        cuisine_type = m.group(1).strip()

    # decor_style: "<words> style"
    decor_style = ""
    m = re.search(r'([\w/]+(?:\s+[\w/]+)?)\s+style', q, re.IGNORECASE)
    if m:
        decor_style = m.group(1).strip()

    return {
        "event_type": event_type,
        "location": location,
        "budget": budget,
        "capacity": capacity,
        "cuisine_type": cuisine_type,
        "decor_style": decor_style,
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/research")
async def research(req: QueryRequest):
    fields = parse_query(req.query)
    result = await run_vynues_pipeline(
        event_type=fields["event_type"],
        event_date="",
        budget=fields["budget"],
        capacity=fields["capacity"],
        location=fields["location"],
        decor_style=fields["decor_style"],
        cuisine_type=fields["cuisine_type"],
        page=req.page,
    )
    return result
