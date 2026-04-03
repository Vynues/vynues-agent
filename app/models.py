from pydantic import BaseModel, Field
from typing import List


# ── Request ───────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language event planning query")
    event_type: str = Field(default="event", description='e.g. "wedding reception", "birthday party"')
    event_date: str = Field(default="", description='YYYY-MM-DD, e.g. "2025-09-15"')
    budget: str = Field(default="", description='e.g. "$5,000–$15,000"')
    capacity: int = Field(default=100, description="Expected number of guests")
    location: str = Field(..., description='City name, e.g. "Philadelphia"')
    decor_style: str = Field(default="", description='e.g. "rustic", "modern", "classic"')
    cuisine_type: str = Field(default="", description='e.g. "Italian", "BBQ", "Mediterranean"')
    page: int = Field(default=1, ge=1, description="Page number (10 results per page)")


# ── Per-business result ───────────────────────────────────────────────────────

class VenueMatchResponse(BaseModel):
    business_name: str
    address: str
    category: str
    match_score: int
    price_range: str
    distance_miles: float
    phone: str
    website: str
    cuisine_type: str | None = None   # only present for Catering entries


# ── Response ──────────────────────────────────────────────────────────────────

class ResearchResponse(BaseModel):
    summary: str
    total_results: int
    current_page: int
    total_pages: int
    venues: List[VenueMatchResponse]
    catering: List[VenueMatchResponse]
    florists: List[VenueMatchResponse]
    bakeries: List[VenueMatchResponse]
