# RobinReal listing search — Datathon 2026

Hybrid retrieval + ranking for Swiss rental listings: **SQLite** + **BM25** + **dense embeddings** (Bedrock Cohere or local fastembed) + optional **SigLIP2 VLM** shards + **geo / POI** features. Natural-language queries go through **Claude** (Bedrock or Anthropic API) for structured understanding; an **MCP** server exposes the same API to ChatGPT / Claude-style clients with a **Vite + React** map/list widget.

This repo started from the official challenge harness; the participant pipeline lives under `app/participant/` (notably `query_understanding.py`, `unified_ranker.py`, `hard_fact_extraction.py`).

---

## Prerequisites

| Tool | Notes |
|------|--------|
| **Python 3.12+** | Required (`pyproject.toml`). |
| **[uv](https://docs.astral.sh/uv/)** | Dependency install and `uv run` commands. |
| **Node.js ≥ 18** | Only for building the MCP widget (`apps_sdk/web`). |
| **Docker & Docker Compose** | Optional; recommended for S3 bootstrap + reproducible runtime. |
| **NVIDIA GPU + nvidia-container-toolkit** | Optional; for `docker-compose.gpu.yml` (fastembed-gpu in the API container). |
| **AWS credentials** | Optional locally; required to pull artifacts from S3 and/or use Bedrock / S3 image helpers. |

---

## 1. Clone and install (Python)

```bash
git clone <your-repo-url>
cd robinreal_datathon_challenge_2026

uv sync --dev
```

Optional fastembed (CPU) for local embeddings when not using Bedrock Cohere:

```bash
uv sync --dev --extra cpu
```

GPU extra (only if you have CUDA and want `fastembed-gpu` locally—usually use Docker GPU override instead):

```bash
uv sync --dev --extra gpu
```

---

## 2. Environment variables

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Important groups (see comments inside `.env.example`):

- **Bedrock** — `BEDROCK_AWS_*`, model IDs for Sonnet (query understanding), Cohere embeddings, Haiku (reserved in config).
- **Anthropic API** — `ANTHROPIC_API_KEY` when Bedrock is blocked or for `hard_fact_extraction` (Haiku JSON) + Sonnet fallback path.
- **AWS S3** — `ARTIFACTS_S3_*`, `AWS_*` for bootstrap and listing image enrichment.
- **Local embeddings** — `USE_LOCAL_EMBEDDINGS`, `LOCAL_EMBEDDING_MODEL`, `HF_HOME` / Hugging Face cache for SigLIP text tower if you use VLM.

Docker Compose **loads `.env` automatically** from the project root.

---

## 3. Data and artifacts

The ranking API expects a populated **`listings.db`** (and optionally **BM25 / geo / VLM shards** under paths used by `unified_ranker`).

### Option A — Docker Compose (recommended)

On `docker compose up`, the **`bootstrap`** service runs `scripts/bootstrap_from_s3.sh` once and syncs from `ARTIFACTS_S3_BUCKET` (default in `.env.example`) into the volume:

- `/data/listings.db`, `bm25.pkl`, `geo/…`
- `/app/features_vlm/siglip2/shard_*.npz`

You need valid **`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`** (and usually `AWS_SESSION_TOKEN` for workshop STS) in `.env`.

Optional:

- `BOOTSTRAP_FORCE=1` — re-sync even if files exist.
- `BOOTSTRAP_INCLUDE_IMAGES=1` — also sync raw images (large).

### Option B — Local `uvicorn` without Docker

Point the app at a database you already have:

```bash
export LISTINGS_DB_PATH="$PWD/data/listings.db"
```

Ensure `data/` contains `listings.db` (and that `features_vlm/siglip2/` exists relative to the repo if you use VLM). You can copy artifacts from a teammate or run bootstrap **inside** a one-off container, or use `aws s3 sync` manually to mirror what `bootstrap_from_s3.sh` does.

### Challenge CSVs (harness)

If you still use the organizer’s **`raw_data.zip`** under `raw_data/`, the harness can bootstrap SQLite from CSV on startup in some configurations; the **team pipeline** for this project assumes **prebuilt artifacts** (typically from S3) for evaluation-scale search.

---

## 4. Run the FastAPI API (local)

```bash
# Optional: Hugging Face cache location (SigLIP text encoder on first VLM use)
export HF_HOME="$PWD/.hf_cache"

uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Smoke checks:

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/listings \
  -H 'Content-Type: application/json' \
  -d '{"query":"2 rooms Zurich under 2500","limit":5}'
```

Structured filter-only search (bypasses NL query pipeline):

```bash
curl -s -X POST http://127.0.0.1:8000/listings/search/filter \
  -H 'Content-Type: application/json' \
  -d '{"hard_filters":{"city":["Zürich"],"max_price":3000,"limit":5}}'
```

POI helper (used by the MCP tool `get_nearby_pois`):

```bash
curl -s "http://127.0.0.1:8000/poi/nearby?lat=47.37&lng=8.54&poi_type=transit&k=3"
```

Default DB path without env: **`data/listings.db`** under the repo (`app/config.py`).

---

## 5. Build the MCP widget (Vite)

Required before the MCP server can serve `ReadResource` HTML (manifest under `apps_sdk/web/dist/.vite/manifest.json`).

```bash
cd apps_sdk/web
npm ci
npm run build
cd ../..
```

---

## 6. Run the MCP server (local)

The MCP app proxies search to the API; it does **not** reimplement ranking.

**Terminal 1 — API**

```bash
export HF_HOME="$PWD/.hf_cache"   # optional
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — MCP**

```bash
export APPS_SDK_LISTINGS_API_BASE_URL=http://127.0.0.1:8000
export APPS_SDK_PUBLIC_BASE_URL=http://127.0.0.1:8001
export LISTINGS_IMAGE_PUBLIC_BASE_URL=http://127.0.0.1:8000

cd apps_sdk/web && npm run build && cd ../..
uv run uvicorn apps_sdk.server.main:app --reload --host 127.0.0.1 --port 8001
```

Endpoints:

- MCP (streamable HTTP): **`http://127.0.0.1:8001/mcp`**
- Optional HTML viewer (tool `open_results_page`): **`http://127.0.0.1:8001/view/<session_id>`**

Saved JSON traces from MCP search (when enabled) go under **`results/`** in the repo by default (`RESULTS_DIR`; see `apps_sdk/server/main.py`).

---

## 7. Public URL (tunnel) for ChatGPT / remote MCP

Expose **port 8001** (MCP). Examples:

**Cloudflare quick tunnel**

```bash
npx --yes cloudflared tunnel --url http://127.0.0.1:8001
```

**ngrok**

```bash
ngrok http 8001
```

Set the **HTTPS origin** the tunnel prints (no trailing slash):

```bash
export APPS_SDK_PUBLIC_BASE_URL="https://YOUR-SUBDOMAIN.trycloudflare.com"
# or https://xxxx.ngrok-free.app
export LISTINGS_IMAGE_PUBLIC_BASE_URL="http://127.0.0.1:8000"
```

If **browsers** must load listing images via your API through a second tunnel, also expose **8000** and set `LISTINGS_IMAGE_PUBLIC_BASE_URL` to that HTTPS API origin.

Restart the MCP server after changing env vars.

Optional hardening (can cause `421` if misconfigured):

```bash
export MCP_ALLOWED_HOSTS=your-tunnel-hostname
export MCP_ALLOWED_ORIGINS=https://your-tunnel-hostname
```

Register in the client:

```text
https://YOUR-TUNNEL-HOST/mcp
```

---

## 8. Docker Compose — full stack

From the repo root (with `.env` filled for AWS / Bedrock as needed):

```bash
docker compose up --build
```

Services:

| Service | Port | Role |
|---------|------|------|
| `bootstrap` | — | One-shot S3 sync into volume |
| `api` | **8000** | FastAPI |
| `mcp` | **8001** | MCP + static widget assets |

**GPU** (host must have NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

This switches the API image to **`fastembed-gpu`** and attaches GPUs to the **`api`** service only.

Inside Compose, MCP uses `APPS_SDK_LISTINGS_API_BASE_URL=http://api:8000`. For remote widgets, override **`APPS_SDK_PUBLIC_BASE_URL`** in `.env` to your tunnel HTTPS origin before starting `mcp`.

---

## 9. MCP protocol smoke test

With API + MCP running and the widget built:

```bash
uv run python scripts/mcp_smoke.py --url http://127.0.0.1:8001/mcp
```

---

## 10. Tests

```bash
uv run pytest tests -q
```

---

## 11. Key directories

```text
app/
  api/routes/listings.py       # POST /listings, /listings/search/filter, GET /poi/nearby
  participant/
    query_understanding.py     # Claude Sonnet + tool use (Bedrock / Anthropic)
    hard_fact_extraction.py    # Claude Haiku JSON hard filters (Anthropic API)
    unified_ranker.py          # Hybrid ranking + VLM / geo / fusion
  core/                        # SQLite filters, S3 helpers
apps_sdk/
  server/main.py               # MCP tools + widget resource + /view/* viewer
  web/                         # Vite React widget → dist/
scripts/
  bootstrap_from_s3.sh         # Artifact sync (Docker bootstrap)
data/                          # Default local DB path: data/listings.db
results/                     # Optional MCP JSON dumps (local dev)
.env.example                 # Copy to .env
```

---

## 12. Troubleshooting

| Issue | What to check |
|--------|----------------|
| `Widget manifest not found` | Run `npm run build` in `apps_sdk/web`. |
| `404` / empty search | `LISTINGS_DB_PATH` valid and file exists; run bootstrap or sync S3. |
| Bedrock `AccessDenied` | Workshop policy; use `ANTHROPIC_API_KEY` or `FORCE_BEDROCK_FALLBACK=1` per `.env.example`. |
| VLM slow / download on first query | Set `HF_HOME` to a persistent directory with space. |
| MCP works locally but widget broken remotely | `APPS_SDK_PUBLIC_BASE_URL` must be the **tunnel HTTPS** origin, not `localhost`. |
| `Permission denied` writing results | Default `RESULTS_DIR` is repo `results/` for local MCP; Docker uses `./results:/results`. |

---

## 13. Side-challenge notes (AWS / Anthropic)

Short write-ups for organizers:

- **`aws.txt`** — S3 artifacts, Bedrock, CLI/bootstrap.
- **`claude.txt`** — Claude Sonnet / Haiku usage and MCP.

---

## 14. Technical write-up

See `technical_writeup.pdf for a fuller architecture and pipeline description. The original challenge harness emphasized CSV import stubs; this repo’s main search path is the **unified ranker** + artifacts from S3.
