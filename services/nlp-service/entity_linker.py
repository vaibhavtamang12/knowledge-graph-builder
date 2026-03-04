"""
entity_linker.py – Entity Linking to Wikidata / Wikipedia
==========================================================

Enriches extracted named entities by linking them to external knowledge
bases and fetching metadata (description, industry, location, aliases, etc.)

Fixes in this version
---------------------
- Disambiguation pages are now resolved using label-specific hints
  (e.g. "Tesla" + ORG → tries "Tesla company", finds the car company)
- Field extraction is more robust: falls back to constructing the Wikipedia
  URL from the canonical title if content_urls is missing
- extract_html tags are stripped automatically
- entity label is passed through to the Wikipedia lookup so disambiguation
  can be resolved correctly

Public API
----------
    enrich_entities(entities: list[dict]) -> list[dict]
"""

from __future__ import annotations
import re
import time
import urllib.parse
import urllib.request
import json
from functools import lru_cache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIKIPEDIA_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKIPEDIA_SEARCH_API  = (
    "https://en.wikipedia.org/w/api.php"
    "?action=opensearch&search={query}&limit=5&namespace=0&format=json"
)
_WIKIDATA_SEARCH = (
    "https://www.wikidata.org/w/api.php"
    "?action=wbsearchentities&search={query}&language=en&format=json&limit=1"
)
_WIKIDATA_ENTITY = (
    "https://www.wikidata.org/w/api.php"
    "?action=wbgetentities&ids={qid}&languages=en&format=json"
    "&props=labels|aliases|descriptions|claims"
)

# Wikidata property IDs we care about
_WD_PROPS = {
    "P452":  "industry",
    "P17":   "country",
    "P131":  "location",
    "P571":  "inception_date",
    "P856":  "website",
    "P18":   "image",
    "P625":  "coordinates",
    "P569":  "date_of_birth",
    "P570":  "date_of_death",
    "P106":  "occupation",
}

# Entity labels worth linking (skip DATE, CARDINAL, ORDINAL, etc.)
_LINKABLE_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"}

# When a disambiguation page is hit, these suffixes are appended to the
# entity text and retried, ordered by priority per entity label
_DISAMBIGUATION_HINTS = {
    "ORG":     ["company", "corporation", "Inc", "organization", "brand"],
    "PERSON":  ["politician", "actor", "athlete", "musician", "author", "businessman"],
    "GPE":     ["city", "state", "country", "capital city"],
    "LOC":     ["river", "mountain", "region", "lake"],
    "PRODUCT": ["product", "software", "car", "model"],
}

_REQUEST_TIMEOUT  = 5    # seconds per HTTP call
_RATE_LIMIT_DELAY = 0.1  # seconds between outbound requests (be polite)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict | None:
    """Fetch *url* and parse JSON. Returns None on any error."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "KnowledgeGraphBuilder/1.0 (entity-linker)"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Wikipedia lookup
# ---------------------------------------------------------------------------

def _fetch_summary(title: str) -> dict | None:
    """Raw fetch of the Wikipedia REST summary endpoint for *title*."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    data = _get_json(_WIKIPEDIA_SUMMARY_API.format(title=encoded))
    time.sleep(_RATE_LIMIT_DELAY)
    return data


def _parse_summary(data: dict, confidence: float) -> dict:
    """
    Extract our fields from a Wikipedia REST summary response dict.
    Handles missing/None values gracefully.
    """
    # URL: prefer content_urls, fall back to constructing from title
    content_urls = data.get("content_urls") or {}
    desktop = content_urls.get("desktop") or {}
    url = desktop.get("page") or ""
    if not url:
        canonical = (data.get("title") or "").replace(" ", "_")
        if canonical:
            url = f"https://en.wikipedia.org/wiki/{canonical}"

    description = data.get("description") or ""
    extract = data.get("extract") or data.get("extract_html") or ""
    extract = re.sub(r"<[^>]+>", "", extract)[:500]

    return {
        "wikipedia_url":   url,
        "wikipedia_id":    data.get("title") or "",
        "description":     description,
        "extract":         extract,
        "link_confidence": confidence,
    }


def _resolve_disambiguation(entity_text: str, entity_label: str) -> dict:
    """
    When a direct Wikipedia lookup returns a disambiguation page, try
    label-specific suffixes to find the intended article.
    Falls back to opensearch if all suffix attempts fail.
    """
    hints = _DISAMBIGUATION_HINTS.get(entity_label, [""])

    for hint in hints:
        candidate = f"{entity_text} {hint}".strip()
        data = _fetch_summary(candidate)
        if data and data.get("type") != "disambiguation" and data.get("title"):
            return _parse_summary(data, confidence=0.75)

    # Last resort: opensearch → take the top result title
    query       = urllib.parse.quote(entity_text)
    search_data = _get_json(_WIKIPEDIA_SEARCH_API.format(query=query))
    time.sleep(_RATE_LIMIT_DELAY)

    if search_data and len(search_data) >= 2 and search_data[1]:
        top_title = search_data[1][0]
        data = _fetch_summary(top_title)
        if data and data.get("type") != "disambiguation" and data.get("title"):
            return _parse_summary(data, confidence=0.60)

    return {}


# Pages whose description suggests the wrong entity type for this label
_LABEL_MISMATCH_HINTS = {
    "ORG":    ["physicist", "inventor", "painter", "poet", "novelist",
               "philosopher", "musician", "athlete", "politician", "actor"],
    "PERSON": ["company", "corporation", "brand", "organization", "software",
               "city", "town", "municipality", "river", "mountain"],
    "GPE":    ["physicist", "inventor", "company", "corporation", "brand"],
}


def _is_wrong_article(data: dict, entity_label: str) -> bool:
    """Return True if the Wikipedia page looks like the wrong type for this label."""
    bad_keywords = _LABEL_MISMATCH_HINTS.get(entity_label, [])
    description  = (data.get("description") or "").lower()
    extract      = (data.get("extract") or "").lower()[:200]
    combined     = description + " " + extract
    return any(kw in combined for kw in bad_keywords)


@lru_cache(maxsize=512)
def _lookup_wikipedia(entity_text: str, entity_label: str = "ORG") -> dict:
    """
    Query Wikipedia for *entity_text*, handling:
    - disambiguation pages (e.g. "Austin")
    - wrong-article mismatches (e.g. "Tesla" ORG → car company, not physicist)

    Strategy:
    1. Direct lookup
    2. If disambiguation or wrong article → try label-specific suffix hints
    3. If no suffix works → opensearch top result
    """
    data = _fetch_summary(entity_text)

    if not data:
        return {}

    if data.get("type") == "disambiguation" or _is_wrong_article(data, entity_label):
        return _resolve_disambiguation(entity_text, entity_label)

    exact_match = (data.get("title") or "").lower() == entity_text.lower()
    confidence  = 0.90 if exact_match else 0.70
    return _parse_summary(data, confidence)


# ---------------------------------------------------------------------------
# Wikidata lookup
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _resolve_qid_label(qid: str) -> str:
    """
    Resolve a Wikidata QID (e.g. Q190117) to its English label
    (e.g. "automotive industry"). Returns the raw QID on failure.
    """
    url  = (
        "https://www.wikidata.org/w/api.php"
        f"?action=wbgetentities&ids={qid}&languages=en"
        "&props=labels&format=json"
    )
    data = _get_json(url)
    time.sleep(_RATE_LIMIT_DELAY)
    if not data:
        return qid
    entity = (data.get("entities") or {}).get(qid, {})
    label  = (entity.get("labels") or {}).get("en", {}).get("value")
    return label if label else qid


@lru_cache(maxsize=512)
def _lookup_wikidata(entity_text: str) -> dict:
    """
    Search Wikidata for *entity_text* and return structured metadata.
    Returns a dict with keys: wikidata_id, aliases, metadata.
    Empty dict on failure.
    """
    query       = urllib.parse.quote(entity_text)
    search_data = _get_json(_WIKIDATA_SEARCH.format(query=query))
    time.sleep(_RATE_LIMIT_DELAY)

    if not search_data:
        return {}

    results = search_data.get("search", [])
    if not results:
        return {}

    qid = results[0].get("id", "")
    if not qid:
        return {}

    entity_data = _get_json(_WIKIDATA_ENTITY.format(qid=qid))
    time.sleep(_RATE_LIMIT_DELAY)

    if not entity_data:
        return {"wikidata_id": qid, "aliases": [], "metadata": {}}

    wd_entity = (entity_data.get("entities") or {}).get(qid, {})

    aliases = [
        a["value"]
        for a in (wd_entity.get("aliases") or {}).get("en", [])
    ]

    claims   = wd_entity.get("claims") or {}
    metadata: dict = {}
    for prop_id, prop_name in _WD_PROPS.items():
        if prop_id not in claims:
            continue
        snak = claims[prop_id][0].get("mainsnak", {})
        dv   = snak.get("datavalue", {})
        val  = dv.get("value")
        if val is None:
            continue
        if isinstance(val, dict):
            if "id" in val:
                # Resolve QID to human-readable English label
                metadata[prop_name] = _resolve_qid_label(val["id"])
            elif "text" in val:
                metadata[prop_name] = val["text"]
            elif "latitude" in val:
                metadata[prop_name] = {"lat": val["latitude"], "lon": val["longitude"]}
        else:
            metadata[prop_name] = str(val)

    return {
        "wikidata_id": qid,
        "aliases":     aliases,
        "metadata":    metadata,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_entities(entities: list[dict]) -> list[dict]:
    """
    Enrich each entity in *entities* with Wikipedia / Wikidata data.
    Only entities whose label is in _LINKABLE_LABELS are looked up.
    Returns a new list of entity dicts (originals not mutated).
    """
    enriched = []
    for entity in entities:
        if entity.get("label") not in _LINKABLE_LABELS:
            enriched.append(entity)
            continue

        result = dict(entity)

        try:
            wiki  = _lookup_wikipedia(entity["text"], entity.get("label", "ORG"))
            wdata = _lookup_wikidata(entity["text"])

            result.update(wiki)
            result.update(wdata)

            if not result.get("link_confidence"):
                result["link_confidence"] = 0.50 if wdata else 0.0

        except Exception as exc:
            print(f"[entity_linker] Failed to enrich '{entity['text']}': {exc}")

        enriched.append(result)

    return enriched