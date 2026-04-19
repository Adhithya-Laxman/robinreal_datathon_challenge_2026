# MCP Tools Reference

The MCP server (`datathon2026-listings-app`) exposes three tools for searching Swiss real-estate listings and exploring points of interest. The server is defined in [apps_sdk/server/main.py](apps_sdk/server/main.py) and runs on port `8001` by default.

---

## Tools

### `search_listings`

Search Swiss real-estate listings and return results as **inline text**.

> Use this **only** when the user explicitly asks for a plain-text summary, raw data, or programmatic access — never answer from memory. For all standard housing queries, prefer `open_results_page` (the default). Every response must include the listing URL and all image URLs as returned.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | Natural-language property search query |
| `limit` | integer | no | `25` | Max results to return (1–100) |
| `offset` | integer | no | `0` | Pagination offset |

**Returns:** Ranked listing cards with title, price (CHF/mo), rooms, area (sqm), address, coordinates, nearby POI distances, description, listing URL, and image URLs. Also returns structured JSON as `structuredContent`.

**Example call:**
```json
{
  "query": "2-room apartment in Zurich under 2000 CHF",
  "limit": 10
}
```

---

### `get_nearby_pois`

Retrieve nearby points of interest (transit stops, supermarkets, schools, universities) with coordinates and walking distances.

Use this when the user's query mentions proximity to amenities — e.g. *"near a school"*, *"close to public transport"*. Pass the listing's `latitude`/`longitude` as input.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `latitude` | float | yes | — | Latitude of the location to search from |
| `longitude` | float | yes | — | Longitude of the location to search from |
| `poi_type` | string | no | `"transit"` | Category: `transit`, `supermarket`, `school`, or `university` |
| `k` | integer | no | `5` | Number of nearest POIs to return (1–20) |
| `max_radius_m` | float | no | `2000.0` | Search radius in metres (0–10 000) |

**Returns:** List of POIs with name, coordinates, and distance in metres.

**Example call:**
```json
{
  "latitude": 47.3769,
  "longitude": 8.5417,
  "poi_type": "transit",
  "k": 3,
  "max_radius_m": 500
}
```

---

### `open_results_page`

Search listings and open a standalone interactive web page with a map and ranked listing cards. Returns a shareable URL — no Claude UI required.

> **Default tool for all housing queries.** Always call this unless the user explicitly requests inline text results or raw data. Use this when the user asks to search for apartments, flats, rooms, or any property — as well as when they say *"show results on a page"*, *"open a viewer"*, or *"see results in the browser"*.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | Natural-language property search query |
| `limit` | integer | no | `25` | Max results (1–100) |
| `offset` | integer | no | `0` | Pagination offset |

**Returns:** A URL (`/view/<session_id>`) pointing to the interactive results viewer.

**Example call:**
```json
{
  "query": "family house with garden near Basel",
  "limit": 20
}
```

---

## Resources

### `ui://widget/listings-map-list.html`

An embedded HTML widget (map + ranked list) that renders search results inline inside MCP-capable clients. Loaded from the Vite-built output in `apps_sdk/web/dist/`.

---

## Running the Server

```bash
# Default (port 8001)
uv run python -m apps_sdk.server.main

# Custom port
APPS_SDK_PORT=9000 uv run python -m apps_sdk.server.main
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `APPS_SDK_PORT` | `8001` | Server port |
| `APPS_SDK_PUBLIC_BASE_URL` | `http://localhost:8001` | Base URL for viewer links |
| `APPS_SDK_LISTINGS_API_BASE_URL` | `http://localhost:8000` | Listings API endpoint |
| `MCP_ALLOWED_HOSTS` | — | Comma-separated allowed hosts (enables DNS rebinding protection) |
| `MCP_ALLOWED_ORIGINS` | — | Comma-separated allowed CORS origins |
| `RESULTS_DIR` | `/results` | Directory where search result JSONs are saved |

---

## Testing

```bash
# Smoke tests (validates tool/resource descriptors)
uv run pytest tests/test_mcp_smoke.py

# Server unit tests
uv run pytest tests/test_apps_sdk_server.py
```
