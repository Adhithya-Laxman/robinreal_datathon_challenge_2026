from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import HTMLResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from apps_sdk.server.client import get_listings_api_client
from apps_sdk.server.widget import (
    WIDGET_MIME_TYPE,
    WIDGET_TEMPLATE_URI,
    get_public_base_url,
    get_widget_dist_dir,
    load_widget_html,
)

SEARCH_TOOL_NAME = "search_listings"
POI_TOOL_NAME = "get_nearby_pois"
VIEWER_TOOL_NAME = "open_results_page"
_VALID_POI_TYPES = ["transit", "supermarket", "school", "university"]

# In-memory session store: session_id → {query, payload}
_results_store: dict[str, dict[str, Any]] = {}
MAP_RESOURCE_ORIGINS = [
    "https://a.basemaps.cartocdn.com",
    "https://b.basemaps.cartocdn.com",
    "https://c.basemaps.cartocdn.com",
    "https://d.basemaps.cartocdn.com",
    "https://assets.comparis.ch",
    "https://assets-comparis.b-cdn.net",
]


class SearchListingsInput(BaseModel):
    query: str = Field(..., description="Natural-language property search query.")
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class GetNearbyPoisInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to search from.")
    longitude: float = Field(..., description="Longitude of the location to search from.")
    poi_type: str = Field(
        default="transit",
        description="Category of POI: 'transit' (bus/tram/train), 'supermarket', 'school', or 'university'.",
    )
    k: int = Field(default=5, ge=1, le=20, description="Number of nearest POIs to return.")
    max_radius_m: float = Field(default=2000.0, ge=0, le=10000, description="Search radius in metres.")

    model_config = ConfigDict(extra="forbid")


class OpenResultsPageInput(BaseModel):
    query: str = Field(..., description="Natural-language property search query.")
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class PublicWidgetStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict[str, Any]) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code < 400:
            response.headers.setdefault("Access-Control-Allow-Origin", "*")
            response.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
            response.headers.setdefault("Access-Control-Allow-Headers", "*")
            response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        return response


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _transport_security_settings() -> TransportSecuritySettings:
    allowed_hosts = _split_env_list(os.getenv("MCP_ALLOWED_HOSTS"))
    allowed_origins = _split_env_list(os.getenv("MCP_ALLOWED_ORIGINS"))
    if not allowed_hosts and not allowed_origins:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def build_tool_descriptor() -> types.Tool:
    return types.Tool(
        name=SEARCH_TOOL_NAME,
        title="Search listings",
        description=(
            "Search Swiss real-estate listings. "
            "ALWAYS call this tool for any housing, apartment, flat, or room search — never answer from memory. "
            "After receiving results, you MUST include in your response for every listing: "
            "(1) the Listing URL and (2) all Image URLs exactly as returned."
        ),
        inputSchema=SearchListingsInput.model_json_schema(),
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        ),
        _meta=build_tool_meta(),
    )


def build_search_tool_result(
    *,
    query: str,
    payload: dict[str, Any],
) -> types.CallToolResult:
    listings = payload.get("listings", [])
    count = len(listings)
    lines = [
        f"Showing {count} listing{'s' if count != 1 else ''} for \"{query}\".\n"
        f"IMPORTANT: In your response you MUST show the Listing URL and all Image URLs for every property listed below.\n"
    ]
    for item in listings:
        if not isinstance(item, dict):
            continue
        # If API returns "listing": null, .get("listing", {}) is None — normalize to {}.
        l = item.get("listing") or {}
        if not isinstance(l, dict):
            continue
        features = ", ".join(l.get("features") or []) or "none"
        lat, lon = l.get("latitude"), l.get("longitude")
        coords = f"{lat}, {lon}" if lat is not None and lon is not None else "N/A"
        image_urls = l.get("image_urls") or []
        hero = l.get("hero_image_url")
        all_images = ([hero] if hero else []) + [u for u in image_urls if u != hero]
        s3_images = [u for u in all_images if u and u.startswith("https://")]
        if s3_images:
            img_lines = "\n".join(f"  {u}" for u in s3_images)
            images_text = f"Image URLs:\n{img_lines}\n"
        else:
            images_text = ""
        poi_parts = []
        if l.get("geo_transit_m") is not None:
            poi_parts.append(f"transit {l['geo_transit_m']}m")
        if l.get("geo_supermarket_m") is not None:
            poi_parts.append(f"supermarket {l['geo_supermarket_m']}m")
        if l.get("geo_school_m") is not None:
            poi_parts.append(f"school {l['geo_school_m']}m")
        if l.get("geo_university_m") is not None:
            poi_parts.append(f"university {l['geo_university_m']}m")
        poi_text = f"Nearby (distance): {', '.join(poi_parts)}\n" if poi_parts else ""
        lines.append(
            f"---\n"
            f"Title: {l.get('title')}\n"
            f"Score: {float(item.get('score') or 0):.3f} ({item.get('reason') or ''})\n"
            f"Price: CHF {l.get('price_chf')}/mo\n"
            f"Rooms: {l.get('rooms')} | Area: {l.get('living_area_sqm')} sqm\n"
            f"Address: {l.get('street')}, {l.get('postal_code')} {l.get('city')}, {l.get('canton')}\n"
            f"Coordinates: {coords}\n"
            f"Available: {l.get('available_from')}\n"
            f"Type: {l.get('object_category')} ({l.get('offer_type')})\n"
            f"Features: {features}\n"
            f"{poi_text}"
            f"Description: {(l.get('description') or '')[:300]}\n"
            f"Listing URL: {l.get('original_listing_url')}\n"
            f"{images_text}"
        )
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="\n".join(lines))],
        structuredContent=payload,
        _meta=build_tool_result_meta(),
    )


def build_poi_tool_descriptor() -> types.Tool:
    return types.Tool(
        name=POI_TOOL_NAME,
        title="Get nearby points of interest",
        description=(
            "Retrieve nearby points of interest (transit stops, supermarkets, schools, universities) "
            "with their exact latitude/longitude coordinates and distance in metres. "
            "Call this when the user's query shows interest in proximity to specific amenities — "
            "e.g. 'near a school', 'close to public transport', 'walking distance to shops'. "
            "Use the listing's latitude/longitude as input. poi_type must be one of: "
            "transit, supermarket, school, university."
        ),
        inputSchema=GetNearbyPoisInput.model_json_schema(),
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        ),
    )


def build_viewer_tool_descriptor() -> types.Tool:
    return types.Tool(
        name=VIEWER_TOOL_NAME,
        title="Open results page",
        description=(
            "Search Swiss real-estate listings and open a beautiful standalone web results page "
            "the user can view in their browser. Returns a URL that shows an interactive map and "
            "ranked listing cards — no Claude or ChatGPT UI required. "
            "Call this when the user asks to 'show results on a page', 'open a viewer', "
            "'see results in the browser', or wants a shareable visual overview."
        ),
        inputSchema=OpenResultsPageInput.model_json_schema(),
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        ),
    )


def build_viewer_tool_result(*, query: str, session_id: str, count: int, base_url: str) -> types.CallToolResult:
    viewer_url = f"{base_url}/view/{session_id}"
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=(
                    f"Results viewer ready — open the link below in your browser:\n\n"
                    f"{viewer_url}\n\n"
                    f"Found {count} listing{'s' if count != 1 else ''} for \"{query}\". "
                    f"The page shows an interactive map with numbered pins and ranked listing cards "
                    f"with images, prices, scores, and direct links to each property."
                ),
            )
        ],
    )


def build_poi_tool_result(payload: dict[str, Any]) -> types.CallToolResult:
    pois = payload.get("pois", [])
    poi_type = payload.get("poi_type", "POI")
    loc = payload.get("queried_location", {})
    lines = [
        f"Nearby {poi_type} locations for ({loc.get('latitude')}, {loc.get('longitude')}):\n"
    ]
    if not pois:
        lines.append("No POIs found within the search radius.")
    for p in pois:
        name = p.get("name") or p.get("type") or poi_type
        lines.append(
            f"- {name}: lat={p['latitude']}, lng={p['longitude']}, distance={p['distance_m']}m"
        )
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="\n".join(lines))],
        structuredContent=payload,
    )


def build_tool_meta() -> dict[str, Any]:
    return {
        "securitySchemes": [{"type": "noauth"}],
        "ui": {
            "resourceUri": WIDGET_TEMPLATE_URI,
            "visibility": ["model", "app"],
        },
        "openai/outputTemplate": WIDGET_TEMPLATE_URI,
        "openai/toolInvocation/invoking": "Searching listings…",
        "openai/toolInvocation/invoked": "Listings ready",
        "openai/widgetAccessible": True,
    }


def build_tool_result_meta() -> dict[str, Any]:
    return {"openai/outputTemplate": WIDGET_TEMPLATE_URI}


def build_resource_contents_meta(*, public_base_url: str | None = None) -> dict[str, Any]:
    base_url = public_base_url or get_public_base_url()
    return {
        "ui": {
            "prefersBorder": False,
            "csp": {
                "connectDomains": [base_url, *MAP_RESOURCE_ORIGINS],
                "resourceDomains": [base_url, *MAP_RESOURCE_ORIGINS],
            },
        },
        "openai/widgetAccessible": True,
    }


mcp = FastMCP(
    name="datathon2026-listings-app",
    stateless_http=True,
    transport_security=_transport_security_settings(),
)


@mcp._mcp_server.list_tools()
async def _list_tools() -> list[types.Tool]:
    return [build_tool_descriptor(), build_poi_tool_descriptor(), build_viewer_tool_descriptor()]



@mcp._mcp_server.list_resources()
async def _list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            name="Listings map and ranked list",
            title="Listings map and ranked list",
            uri=WIDGET_TEMPLATE_URI,
            description="Combined ranked list and map widget for listing search results.",
            mimeType=WIDGET_MIME_TYPE,
            _meta=build_resource_contents_meta(),
        )
    ]


async def _make_viewer_response(session_id: str) -> HTMLResponse:
    session = _results_store.get(session_id)
    if not session:
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;padding:48px;color:#6b7280'>"
            "<h2 style='color:#111827'>Session not found</h2>"
            "<p>This link may have expired or is invalid.</p></body></html>",
            status_code=404,
        )
    template_path = Path(__file__).parent / "viewer.html"
    template = template_path.read_text(encoding="utf-8")
    data_json = json.dumps(
        {"query": session["query"], "listings": session["payload"].get("listings", [])},
        ensure_ascii=False,
    ).replace("</", "<\\/")  # prevent </script> injection
    html = template.replace("__DATA_JSON__", data_json)
    return HTMLResponse(html)


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    if str(req.params.uri) != WIDGET_TEMPLATE_URI:
        raise ValueError(f"Unknown resource: {req.params.uri}")

    html = load_widget_html(
        dist_dir=get_widget_dist_dir(),
        public_base_url=get_public_base_url(),
    )
    return types.ServerResult(
        types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=WIDGET_TEMPLATE_URI,
                    mimeType=WIDGET_MIME_TYPE,
                    text=html,
                    _meta=build_resource_contents_meta(),
                )
            ]
        )
    )


async def _handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    if req.params.name == POI_TOOL_NAME:
        try:
            poi_input = GetNearbyPoisInput.model_validate(req.params.arguments or {})
        except ValidationError as exc:
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Invalid input: {exc.errors()}")],
                    isError=True,
                )
            )
        poi_payload = await get_listings_api_client().get_nearby_pois(
            lat=poi_input.latitude,
            lng=poi_input.longitude,
            poi_type=poi_input.poi_type,
            k=poi_input.k,
            max_radius_m=poi_input.max_radius_m,
        )
        return types.ServerResult(build_poi_tool_result(poi_payload))

    if req.params.name == VIEWER_TOOL_NAME:
        try:
            viewer_input = OpenResultsPageInput.model_validate(req.params.arguments or {})
        except ValidationError as exc:
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Invalid input: {exc.errors()}")],
                    isError=True,
                )
            )
        response_payload = await get_listings_api_client().search_listings(
            query=viewer_input.query,
            limit=viewer_input.limit,
            offset=viewer_input.offset,
        )
        session_id = str(uuid.uuid4())
        _results_store[session_id] = {"query": viewer_input.query, "payload": response_payload}
        count = len(response_payload.get("listings", []))
        return types.ServerResult(
            build_viewer_tool_result(
                query=viewer_input.query,
                session_id=session_id,
                count=count,
                base_url=get_public_base_url(),
            )
        )

    if req.params.name != SEARCH_TOOL_NAME:
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {req.params.name}")],
                isError=True,
            )
        )

    try:
        search_input = SearchListingsInput.model_validate(req.params.arguments or {})
    except ValidationError as exc:
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Invalid input: {exc.errors()}")],
                isError=True,
            )
        )

    response_payload = await get_listings_api_client().search_listings(
        query=search_input.query,
        limit=search_input.limit,
        offset=search_input.offset,
    )
    try:
        _save_results(query=search_input.query, payload=response_payload)
    except Exception as exc:
        print(f"[results] failed to save: {exc}", flush=True)
    return types.ServerResult(
        build_search_tool_result(query=search_input.query, payload=response_payload)
    )


def _default_results_dir() -> Path:
    # Repo root: apps_sdk/server/main.py -> parents[2]
    return Path(__file__).resolve().parents[2] / "results"


def _save_results(*, query: str, payload: dict[str, Any]) -> None:
    # Default to ./results under the repo (writable locally). Docker sets RESULTS_DIR=/results.
    results_dir = Path(os.getenv("RESULTS_DIR", str(_default_results_dir())))
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w]+", "_", query.lower()).strip("_")[:60]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"{timestamp}_{slug}.json"
    out.write_text(json.dumps({"query": query, "results": payload}, indent=2, ensure_ascii=False))


mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource
mcp._mcp_server.request_handlers[types.CallToolRequest] = _handle_call_tool

_mcp_app = mcp.streamable_http_app()
_widget_dist_dir = get_widget_dist_dir()
_widget_dist_dir.mkdir(parents=True, exist_ok=True)
_mcp_app.mount(
    "/widget-assets",
    PublicWidgetStaticFiles(directory=str(_widget_dist_dir)),
    name="widget-assets",
)

class _ViewerMiddleware:
    """Intercepts /view/<id> requests; passes everything else (incl. lifespan) to the MCP app."""

    __slots__ = ("_app",)

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/view/"):
            session_id = scope["path"][len("/view/"):]
            response = await _make_viewer_response(session_id)
            await response(scope, receive, send)
        else:
            await self._app(scope, receive, send)


app = _ViewerMiddleware(_mcp_app)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("APPS_SDK_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
