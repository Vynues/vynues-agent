import json
import math
import os
from typing import List

from agents import Agent, Runner, function_tool
from pydantic import BaseModel

MODEL = "gpt-4o"

# Resolve dataset path relative to this file so it works both locally and in
# production (Railway mounts the repo at its root).
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.environ.get(
    "YELP_DATASET_PATH",
    os.path.join(_HERE, "..", "Yelp JSON", "yelp_academic_dataset_business.json"),
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class ScoredBusiness(BaseModel):
    """Analyst output — includes yelp_stars and coordinates for pipeline use."""
    business_name: str
    address: str
    match_score: int
    reason: str
    category: str             # "Venues" | "Catering" | "Florists" | "Bakeries"
    yelp_stars: float
    price_range: str
    phone: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    cuisine_type: str = ""    # Catering only


class AnalystOutput(BaseModel):
    scored_businesses: List[ScoredBusiness]


class VenueMatch(BaseModel):
    """Final per-business result returned to the frontend."""
    business_name: str
    address: str
    category: str
    match_score: int
    price_range: str
    distance_miles: float = 0.0
    phone: str = "Not available"
    website: str = "Not available"
    cuisine_type: str = ""    # Catering only


class WriterOutput(BaseModel):
    venues: List[VenueMatch]
    catering: List[VenueMatch]
    florists: List[VenueMatch]
    bakeries: List[VenueMatch]


# ── Lookup tables ─────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "Venues": [
        "venues & event spaces", "event space", "banquet hall", "banquet halls",
        "wedding venue", "ballroom", "convention center", "country club",
        "reception hall", "social clubs", "hotels", "hotel",
    ],
    "Catering": [
        "caterers", "catering",
        "restaurants", "restaurant",
        "food trucks", "food truck",
    ],
    "Florists": [
        "florists", "florist",
        "floral designers",
        "flowers & gifts",
    ],
    "Bakeries": [
        "bakeries", "bakery",
        "cupcakes", "custom cakes",
        "desserts", "patisserie",
    ],
}

PRICE_LABELS = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}

CUISINE_KEYWORD_MAP: dict[str, list[str]] = {
    "american"      : ["american (traditional)", "american (new)", "burgers", "steakhouses", "diners"],
    "italian"       : ["italian", "pizza", "pasta"],
    "mexican"       : ["mexican", "tex-mex", "tacos", "burritos"],
    "chinese"       : ["chinese", "cantonese", "dim sum", "szechuan"],
    "japanese"      : ["japanese", "sushi", "ramen"],
    "indian"        : ["indian", "pakistani"],
    "mediterranean" : ["mediterranean", "greek", "middle eastern", "lebanese"],
    "french"        : ["french", "brasseries"],
    "thai"          : ["thai"],
    "bbq"           : ["barbeque", "bbq", "smokehouse"],
    "barbeque"      : ["barbeque", "bbq", "smokehouse"],
    "vegan"         : ["vegan", "vegetarian"],
    "vegetarian"    : ["vegan", "vegetarian"],
    "seafood"       : ["seafood", "fish & chips"],
    "korean"        : ["korean", "korean bbq"],
    "vietnamese"    : ["vietnamese", "pho"],
    "spanish"       : ["spanish", "tapas bars", "tapas/small plates"],
    "caribbean"     : ["caribbean", "cuban"],
    "southern"      : ["southern", "soul food", "cajun/creole"],
    "middle eastern": ["middle eastern", "lebanese", "halal"],
    "latin"         : ["latin american", "colombian", "peruvian"],
}

VALID_CATEGORIES = {"Venues", "Catering", "Florists", "Bakeries"}
VALID_PRICES     = {"$", "$$", "$$$", "$$$$"}

# Module-level flag set by search_yelp_businesses, read by run_vynues_pipeline.
# Not thread-safe for concurrent requests — acceptable for single-worker Railway deploy;
# move into a request-scoped context object if you scale to multiple workers.
_catering_cuisine_fallback: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cuisine_keywords(cuisine_type: str) -> list[str]:
    if not cuisine_type:
        return []
    ct = cuisine_type.lower().strip()
    if ct in CUISINE_KEYWORD_MAP:
        return CUISINE_KEYWORD_MAP[ct]
    for key, kws in CUISINE_KEYWORD_MAP.items():
        if key in ct or ct in key:
            return kws
    return [ct]


def detect_cuisine_type(cats_str: str) -> str:
    for cuisine, keywords in CUISINE_KEYWORD_MAP.items():
        if any(kw in cats_str for kw in keywords):
            return cuisine.title()
    return ""


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(a)), 1)


def clean_match(match: VenueMatch, coord_map: dict, city_lat: float, city_lng: float) -> VenueMatch:
    lat, lng = coord_map.get(match.business_name, (0.0, 0.0))
    if lat != 0.0 and lng != 0.0 and city_lat != 0.0:
        match.distance_miles = haversine_miles(city_lat, city_lng, lat, lng)
    else:
        match.distance_miles = 0.0
    if match.price_range not in VALID_PRICES:
        match.price_range = "Not available"
    if not match.phone or not match.phone.strip():
        match.phone = "Not available"
    match.website = "Not available"
    if match.category not in VALID_CATEGORIES:
        match.category = "Venues"
    return match


def build_summary(event_type: str, location: str, cuisine_type: str, catering_fallback: bool) -> str:
    base = f"Here are your venue matches for a {event_type} in {location}"
    if catering_fallback and cuisine_type:
        return (
            f"{base}. No exact {cuisine_type} catering match was found in {location}; "
            f"similar catering and restaurant options are shown instead."
        )
    return base + "."


# ── Yelp search tool ──────────────────────────────────────────────────────────

@function_tool
def search_yelp_businesses(location: str, cuisine_type: str = "", max_per_category: int = 25) -> str:
    """Search the Yelp dataset for event-relevant businesses near a given city.

    Filters businesses into four Vynues categories:
    - Venues (event spaces, banquet halls, hotels)
    - Catering (caterers, restaurants, food trucks)
    - Florists (florists, floral designers)
    - Bakeries (bakeries, cake shops)

    For Catering, results are filtered to match cuisine_type when provided.
    Falls back to all catering/restaurants if no cuisine match is found.

    Args:
        location: City name to search (e.g. "Philadelphia", "Nashville").
        cuisine_type: Cuisine preference for Catering (e.g. "Italian", "BBQ").
                      Leave empty to return all catering businesses.
        max_per_category: Maximum businesses to return per category (default 25).

    Returns:
        Formatted string with businesses grouped by category.
    """
    global _catering_cuisine_fallback
    _catering_cuisine_fallback = False

    cuisine_keywords = get_cuisine_keywords(cuisine_type)
    results: dict[str, list] = {cat: [] for cat in CATEGORY_KEYWORDS}
    catering_exact: list[dict] = []

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    biz = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if biz.get("is_open", 0) == 0:
                    continue

                city = (biz.get("city") or "").strip().lower()
                loc = location.strip().lower()
                if loc not in city and city not in loc:
                    continue

                cats_str = (biz.get("categories") or "").lower()

                matched = None
                for vcat, keywords in CATEGORY_KEYWORDS.items():
                    if any(kw in cats_str for kw in keywords):
                        matched = vcat
                        break
                if matched is None:
                    continue

                attrs = biz.get("attributes") or {}
                raw_price = str(attrs.get("RestaurantsPriceRange2", ""))
                price_display = PRICE_LABELS.get(raw_price, "Not available")

                biz_data = {
                    "name": biz.get("name", ""),
                    "address": (
                        f"{biz.get('address', '')}, "
                        f"{biz.get('city', '')}, "
                        f"{biz.get('state', '')} "
                        f"{biz.get('postal_code', '')}"
                    ).strip(", "),
                    "stars": biz.get("stars", 0.0),
                    "review_count": biz.get("review_count", 0),
                    "price_range": price_display,
                    "categories": biz.get("categories", ""),
                    "phone": biz.get("phone", ""),
                    "latitude": biz.get("latitude", 0.0),
                    "longitude": biz.get("longitude", 0.0),
                    "cuisine_type": detect_cuisine_type(cats_str) if matched == "Catering" else "",
                }

                if matched == "Catering" and cuisine_keywords:
                    if any(kw in cats_str for kw in cuisine_keywords):
                        if len(catering_exact) < max_per_category:
                            catering_exact.append(biz_data)
                    else:
                        if len(results["Catering"]) < max_per_category:
                            results["Catering"].append(biz_data)
                else:
                    if len(results[matched]) < max_per_category:
                        results[matched].append(biz_data)

    except FileNotFoundError:
        return (
            f"Error: Dataset not found at '{DATASET_PATH}'. "
            "Set the YELP_DATASET_PATH environment variable to the correct path."
        )

    if cuisine_keywords:
        if catering_exact:
            results["Catering"] = catering_exact
            _catering_cuisine_fallback = False
        else:
            _catering_cuisine_fallback = True

    lines: list[str] = []
    total = sum(len(v) for v in results.values())
    lines.append(f"Found {total} businesses in '{location}' across {len(CATEGORY_KEYWORDS)} categories.\n")

    for cat, businesses in results.items():
        if cat == "Catering" and cuisine_keywords:
            if _catering_cuisine_fallback:
                header = (
                    f"### Catering — no exact '{cuisine_type}' match found; "
                    f"showing general catering options ({len(businesses)} results)"
                )
            else:
                header = f"### Catering — '{cuisine_type}' cuisine ({len(businesses)} results)"
        else:
            header = f"### {cat} ({len(businesses)} results)"

        lines.append(header)
        if not businesses:
            lines.append("  No open businesses found in this category.")
        for i, b in enumerate(businesses, 1):
            entry = (
                f"{i}. {b['name']}\n"
                f"   Address      : {b['address']}\n"
                f"   Yelp Stars   : {b['stars']} ({b['review_count']} reviews)\n"
                f"   Price Range  : {b['price_range']}\n"
                f"   Phone        : {b['phone']}\n"
                f"   Coordinates  : {b['latitude']}, {b['longitude']}\n"
                f"   Yelp Category: {b['categories']}"
            )
            if cat == "Catering" and b["cuisine_type"]:
                entry += f"\n   Cuisine Type : {b['cuisine_type']}"
            lines.append(entry)
        lines.append("")

    return "\n".join(lines)


# ── Agents ────────────────────────────────────────────────────────────────────

analyst_agent = Agent(
    name="VynuesAnalyst",
    model=MODEL,
    instructions=(
        "You are a venue-matching analyst for Vynues, an event planning platform.\n"
        "Your job:\n"
        "1. Call search_yelp_businesses with:\n"
        "   - location    : the city from the user's location preference\n"
        "   - cuisine_type: the cuisine_type from the user's preferences (e.g. 'Italian', 'BBQ')\n"
        "   The tool will automatically filter Catering results to that cuisine type.\n"
        "   If no exact match is found the tool falls back to general catering and says so.\n"
        "2. Review every business returned across four categories:\n"
        "   Venues, Catering, Florists, Bakeries.\n"
        "3. Score each business 1–5 based on how well it matches the user's preferences:\n"
        "   - event_type  : Does this business suit this kind of event?\n"
        "   - budget      : Does the price range fit? ($=budget $$=moderate $$$=pricey $$$$=upscale)\n"
        "   - capacity    : For Venues only — would this venue hold this many guests?\n"
        "   - decor_style : For Venues — does the vibe match?\n"
        "   - cuisine_type: For Catering — the tool has already filtered by cuisine.\n"
        "     Score cuisine-matched businesses 4–5 if otherwise suitable.\n"
        "     If the tool fell back to general catering (noted in the output), score those 3.\n"
        "     Only score below 3 if a business has very low stars or is clearly unsuitable.\n"
        "4. Write a concise reason for each score (1 sentence).\n"
        "5. For each business, populate:\n"
        "   - phone     : Copy exactly from the search results. Use empty string if not listed.\n"
        "   - latitude / longitude: Copy exactly from the search results (0.0 if not listed).\n"
        "   - price_range: Copy exactly from the search results (e.g. '$', '$$', 'Not available').\n"
        "   - category  : Must be exactly one of: Venues, Catering, Florists, Bakeries.\n"
        "   - cuisine_type: For Catering businesses, copy the 'Cuisine Type' field from the search results.\n"
        "     If no 'Cuisine Type' line appears for a catering business, use an empty string.\n"
        "     For all non-Catering businesses, always use an empty string.\n"
        "6. Include ALL businesses in your output — the Writer applies score/stars filters.\n"
        "Score guide: 5=perfect match, 4=strong match, 3=decent match, 2=weak match, 1=poor match."
    ),
    tools=[search_yelp_businesses],
    output_type=AnalystOutput,
)

writer_agent = Agent(
    name="VynuesWriter",
    model=MODEL,
    instructions=(
        "You are the output formatter for Vynues, an event planning platform.\n"
        "Given analyst-scored businesses, return all qualifying results per category:\n"
        "  venues   : ALL Venues   with match_score >= 3 AND yelp_stars >= 3.0\n"
        "  catering : ALL Catering with match_score >= 3 AND yelp_stars >= 3.0\n"
        "  florists : ALL Florists with match_score >= 3 AND yelp_stars >= 3.0\n"
        "  bakeries : ALL Bakeries with match_score >= 3 AND yelp_stars >= 3.0\n"
        "DO NOT limit to top N — include EVERY business that meets both criteria.\n"
        "\n"
        "Each entry must include ONLY these fields:\n"
        "  business_name : string\n"
        "  address       : string\n"
        "  category      : exactly one of: Venues, Catering, Florists, Bakeries\n"
        "  match_score   : integer 3–5\n"
        "  price_range   : '$' | '$$' | '$$$' | '$$$$' | 'Not available'\n"
        "  distance_miles: always 0.0 — pipeline calculates this\n"
        "  phone         : string, use 'Not available' if empty\n"
        "  website       : always 'Not available' — not in this dataset\n"
        "  cuisine_type  : for Catering entries, copy from the cuisine_type field in the analyst data;\n"
        "                  use '' (empty string) for all non-Catering entries\n"
        "\n"
        "Output clean JSON only — no extra commentary, no extra fields."
    ),
    output_type=WriterOutput,
)


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run_vynues_pipeline(
    event_type: str,
    event_date: str,
    budget: str,
    capacity: int,
    location: str,
    decor_style: str,
    cuisine_type: str,
    page: int = 1,
    per_page: int = 10,
) -> dict:
    """Run the Vynues venue-matching pipeline and return paginated JSON."""
    global _catering_cuisine_fallback

    preferences_prompt = (
        f"User event preferences:\n"
        f"  event_type   : {event_type}\n"
        f"  event_date   : {event_date}\n"
        f"  budget       : {budget}\n"
        f"  capacity     : {capacity} guests\n"
        f"  location     : {location}\n"
        f"  decor_style  : {decor_style}\n"
        f"  cuisine_type : {cuisine_type}\n"
        f"\n"
        f"Search Yelp for businesses in '{location}' with cuisine_type='{cuisine_type}', "
        f"then score each one based on how well it matches these preferences."
    )

    # Step 1: Analyst
    analyst_result = await Runner.run(analyst_agent, preferences_prompt)
    analyst_output: AnalystOutput = analyst_result.final_output_as(AnalystOutput)
    scored_count = len(analyst_output.scored_businesses)
    catering_fallback = _catering_cuisine_fallback

    # Step 2: City centroid for distance calculation
    coord_map: dict[str, tuple[float, float]] = {
        b.business_name: (b.latitude, b.longitude)
        for b in analyst_output.scored_businesses
        if b.latitude != 0.0 and b.longitude != 0.0
    }
    valid_coords = list(coord_map.values())
    if valid_coords:
        city_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
        city_lng = sum(c[1] for c in valid_coords) / len(valid_coords)
    else:
        city_lat, city_lng = 0.0, 0.0

    # Step 3: Writer
    writer_prompt = (
        f"Original user preferences:\n"
        f"  event_type={event_type}, event_date={event_date}, budget={budget}, "
        f"capacity={capacity} guests, location={location}, "
        f"decor_style={decor_style}, cuisine_type={cuisine_type}\n\n"
        f"Analyst-scored businesses (all {scored_count}):\n"
        + "\n".join(
            f"- [{b.category}] {b.business_name} | {b.address} | "
            f"score={b.match_score}/5 | stars={b.yelp_stars} | price={b.price_range} | "
            f"phone={b.phone}"
            + (f" | cuisine_type={b.cuisine_type}" if b.category == "Catering" and b.cuisine_type else "")
            for b in analyst_output.scored_businesses
        )
    )
    writer_result = await Runner.run(writer_agent, writer_prompt)
    writer_output: WriterOutput = writer_result.final_output_as(WriterOutput)

    # Step 4: Collect, clean, filter
    all_matches: list[VenueMatch] = (
        writer_output.venues
        + writer_output.catering
        + writer_output.florists
        + writer_output.bakeries
    )
    all_matches = [clean_match(m, coord_map, city_lat, city_lng) for m in all_matches]
    all_matches = [m for m in all_matches if m.match_score >= 3]

    # Step 5: Sort per category then round-robin interleave
    by_category: dict[str, list[VenueMatch]] = {cat: [] for cat in VALID_CATEGORIES}
    for m in all_matches:
        by_category[m.category].append(m)
    for cat in by_category:
        by_category[cat].sort(key=lambda m: -m.match_score)

    ordered: list[VenueMatch] = []
    buckets = list(by_category.values())
    max_len = max((len(b) for b in buckets), default=0)
    for i in range(max_len):
        for bucket in buckets:
            if i < len(bucket):
                ordered.append(bucket[i])

    # Step 6: Paginate
    total_results = len(ordered)
    total_pages   = max(1, math.ceil(total_results / per_page))
    page          = max(1, min(page, total_pages))
    start         = (page - 1) * per_page
    page_matches  = ordered[start : start + per_page]

    def page_for(cat: str) -> list[dict]:
        records = [m.model_dump() for m in page_matches if m.category == cat]
        if cat != "Catering":
            for r in records:
                r.pop("cuisine_type", None)
        return records

    return {
        "summary"      : build_summary(event_type, location, cuisine_type, catering_fallback),
        "total_results": total_results,
        "current_page" : page,
        "total_pages"  : total_pages,
        "venues"  : page_for("Venues"),
        "catering": page_for("Catering"),
        "florists": page_for("Florists"),
        "bakeries": page_for("Bakeries"),
    }
