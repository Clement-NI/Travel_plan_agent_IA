# Travel Agent — Usage & Architecture

A French-Europe travel CLI agent. Plug in any LLM (Claude / Ollama / Gemini),
get real-time flights (Duffel), trains (SNCF Open Data), and buses (FlixBus).

> If you're new, jump to **[Quick start](#quick-start)**.
> If you want the design tour, see **[Architecture](#architecture)**.

---

## What it does

| Ask | What happens | How long |
|---|---|---|
| `"TGV from Paris to Bordeaux on 2026-05-25"` | Regex fast-path → SNCF API → real TGV schedule | ~3-5 s |
| `"Flights from Paris to Lyon tomorrow"` | Regex fast-path → Duffel API → real offers with prices | ~5-8 s |
| `"Plan a trip from Paris to Barcelona on May 25"` | Regex fast-path → flights + trains + buses in parallel | ~10-15 s |
| `"Compare TGV vs flight to Marseille, which is faster?"` | LLM agent (reasoning required) → loads `travel` skill → calls tools | ~30-60 s |
| `"What's 17 × 23?"` | Direct tool call (`calculator`) | ~2-3 s |
| `"hi"` | No tool — just chat | <1 s |

Real data, every time. No mocked responses, no scraping. SNCF data covers all
~3000 French stations plus cross-border services (Paris ↔ Barcelona, Brussels,
Geneva, Frankfurt, Milan, Turin).

---

## Quick start

### 1. Prerequisites

- Python 3.12 (the one at `C:\Users\<you>\AppData\Local\Programs\Python\Python312`)
- `py` launcher on PATH (ships with Python on Windows)
- `git` (to clone)
- `uv` installed for Python (`py -m pip install --user uv`) — needed for the
  `flights-mcp` and `flixbus-mcp` vendor servers
- An Ollama install if you want free inference (default model:
  `gpt-oss:20b-cloud`, requires `ollama signin`)

### 2. Install Python deps

```powershell
cd "D:\Agent AI"
py -m pip install -r requirements-langchain.txt
```

The two vendor MCPs (`flights-mcp`, `flixbus-mcp`) each have their own venv —
already set up if you cloned this project fresh; otherwise:

```powershell
cd MCP\flights-mcp  ; py -m uv sync ; cd ..\..
cd MCP\flixbus-mcp  ; py -m uv sync ; cd ..\..
```

### 3. Configure `.env`

Copy `.env.example` to `.env` and fill in the keys you have. Minimum to
actually return data:

```ini
# at least one LLM provider:
ANTHROPIC_API_KEY=sk-ant-...            # paid, most reliable
# OR run Ollama Cloud (free) — needs `ollama signin` in your shell

# transport APIs (free tiers):
DUFFEL_API_KEY_LIVE=duffel_test_...     # https://duffel.com — test key works
SNCF_API_TOKEN=...                       # https://numerique.sncf.com/startup/api
RAPID_API_KEY=...                        # https://rapidapi.com — subscribe to FlixBus2

# observability:
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...            # https://smith.langchain.com
LANGSMITH_PROJECT=travel-agent
```

### 4. Run

```powershell
# One-shot
py ask.py "TGV from Paris to Bordeaux on 2026-05-25"

# Multi-turn chat
py chat.py

# Web UI (browser at http://localhost:8501)
py -m streamlit run app.py

# Free / paid switch
py ask.py --ollama "..."                    # Ollama Cloud, free
py ask.py "..."                              # Claude Sonnet 4.5, paid

# Model override
py chat.py --ollama --model gpt-oss:120b-cloud
py chat.py --model claude-haiku-4-5
```

That's it. Every run also uploads a trace to LangSmith (sign in to
[smith.langchain.com](https://smith.langchain.com), open the `travel-agent`
project).

### Web UI

`py -m streamlit run app.py` opens a browser chat interface at
`http://localhost:8501`. The sidebar lets you switch provider
(Anthropic / Ollama / Gemini) and model on the fly, with a button to
reset the conversation. The same regex fast-path runs in the UI, so
"plan a trip" queries finish in ~5-10s without the LLM. Stop the
server with Ctrl-C in the terminal.

---

## Architecture

Three independent layers + the entry points wrap them:

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRY POINTS — ask.py (one-shot), chat.py (REPL)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
┌──────────────────────┐              ┌──────────────────────────┐
│ REGEX FAST-PATH      │              │ LLM AGENT                │
│ MCP/skills.py        │              │ LangChain create_agent   │
│                      │              │                          │
│ Parses "X to Y on    │              │ Reads:                   │
│ DATE", dispatches    │              │  • SYSTEM_PROMPT (2 KB)  │
│ MCP tools in         │              │  • skills/travel.md      │
│ parallel.            │              │    (via load_skill tool, │
│                      │              │     loaded on demand)    │
│ ~3-10 s, no LLM.     │              │                          │
└────────┬─────────────┘              └────────┬─────────────────┘
         │                                     │
         └──────────────┬──────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │ MCP TOOL LAYER (MCP/Tools.py)     │
        │                                   │
        │ Generic:  calculator,             │
        │           current_time, file_read,│
        │           web_search, web_fetch,  │
        │           load_skill              │
        │                                   │
        │ From MCP servers (via stdio):     │
        │   search_flights, get_offer_…     │
        │   plan_train_journey, find_…      │
        │   search_locations, search_trips, │
        │   brave_web_search, …             │
        │                                   │
        │ Wrapped with: params-unwrap,      │
        │   date-format coercion, output    │
        │   compression, future-date check  │
        └───────────────┬───────────────────┘
                        │ stdio JSON-RPC
        ┌───────────────┴───────────────────┐
        ▼               ▼                   ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ flights-mcp  │ │ sncf-mcp     │ │ flixbus-mcp      │
│ (vendor)     │ │ (ours, 277L) │ │ (vendor)         │
└──────┬───────┘ └──────┬───────┘ └─────────┬────────┘
       │ HTTPS         │ HTTPS              │ HTTPS
       ▼               ▼                    ▼
   api.duffel       api.sncf.com      flixbus2.p.
   .com             /coverage/sncf    rapidapi.com
```

### The two-tier skill system

A typical user query goes through this decision:

```
user types "TGV from Paris to Bordeaux on 2026-05-25"
            │
            ▼
   MCP/skills.py::parse_query()  ←── regex match: train mode, from=Paris,
            │                          to=Bordeaux, date=2026-05-25
            │
       matched? ──── yes ──→ skills.run_train()
            │                  └─► plan_train_journey(...) → SNCF API
            │                  └─► format + return → exit (no LLM)
            │
           no
            ▼
   LangChain agent (LLM-driven)
            │
            ├─ reads slim system prompt (2 KB)
            │
            ├─ if travel-related, calls load_skill("travel")
            │    └─► reads skills/travel.md (5 KB of detailed rules)
            │
            └─ then makes tool calls per the loaded skill
```

The regex fast-path catches the common shape "X to Y" (with optional date)
across all four modes (`flight` / `train` / `bus` / `plan`). For anything
ambiguous ("compare X vs Y", "best way to get there avoiding airports"), the
LLM takes over with the skill loaded into context.

### Layer breakdown

| File | Lines | Role |
|---|---|---|
| `ask.py` | 135 | Single-shot entrypoint — try skill, fall back to LLM, print, exit |
| `chat.py` | 144 | Multi-turn REPL — same engine, `while input()` loop |
| `MCP/skills.py` | 417 | Regex fast-path: `parse_query`, `run_train/flight/bus/plan` |
| `MCP/Tools.py` | 223 | Generic tools + `load_skill`, splat-imports MCP tools |
| `MCP/mcp_servers.py` | 542 | Spawns 4 MCP server subprocesses; param/date/output adapters |
| `MCP/sncf-mcp/sncf_server.py` | 277 | Our SNCF MCP — `plan_train_journey`, `find_train_station` |
| `configuration/langchain_system_prompt.py` | 45 | Slim system prompt (~2 KB) |
| `skills/travel.md` | — | On-demand travel skill (~5 KB) loaded via `load_skill` |
| `CLI/cli.py` | 164 | Shared helpers (`build_agent`, `_extract_text`, `_fmt_args`) |
| `eval/langsmith_eval.py` | ~210 | Push `cases.json` to LangSmith, run experiments, score |
| `eval/cases.json` | 11 cases | Eval suite (calc, no-tool, transport single/multi/cross-border) |
| `agent.py` | ~60 | `load_dotenv()` helper used by all entrypoints |

---

## Configuration reference

All env vars live in `.env` (template in `.env.example`):

### Required for travel queries

| Var | Where to get it |
|---|---|
| `SNCF_API_TOKEN` | <https://numerique.sncf.com/startup/api> (free) |
| `DUFFEL_API_KEY_LIVE` | <https://duffel.com> (test or live key) |
| `RAPID_API_KEY` | <https://rapidapi.com> → subscribe to `flixbus2` |

### LLM providers (at least one)

| Var | Provider |
|---|---|
| `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> (paid) |
| `GOOGLE_API_KEY` | <https://aistudio.google.com/apikey> (free, daily quota) |
| (none needed for Ollama) | `ollama signin` is sufficient for cloud models |

### Optional

| Var | Effect |
|---|---|
| `BRAVE_API_KEY` | Adds `brave_web_search` / `brave_local_search` to TOOLS |
| `GOOGLE_MAPS_API_KEY` | Adds Google Maps tools |
| `LANGSMITH_API_KEY` | Enables auto-tracing + the eval runner |
| `LANGSMITH_PROJECT` | Project name in LangSmith UI (default: `travel-agent`) |
| `FLIGHTS_MCP_PATH` / `SNCF_MCP_PATH` / `FLIXBUS_MCP_PATH` | Override MCP server location (default: `<project>/MCP/<name>`) |
| `OLLAMA_HOST` | Default `http://localhost:11434` |
| `MCP_PYTHON` | Override the Python interpreter that launches MCP servers |

---

## Evaluation

```powershell
# Full suite (11 cases)
py eval/langsmith_eval.py --provider ollama
py eval/langsmith_eval.py --provider anthropic   # uses Claude (paid)
py eval/langsmith_eval.py --filter transport     # only transport cases
```

Output:
- **Local:** per-case `PASS/FAIL` summary in stdout
- **Remote:** full traces + scores at <https://smith.langchain.com> → project `travel-agent`

`eval/cases.json` covers:
- **generic:** `calc_basic`, `calc_compound`, `time_tokyo`, `file_read`, `no_tool_needed`
- **transport:** train/flight/bus single-mode, multi-mode plan, cross-border (Paris↔Barcelona), skill-loading regression

Each case is scored on `expected_tool`, `expected_tools_any`, `no_tool`,
`answer_regex` — and a roll-up `pass` flag.

---

## Project layout

```
D:\Agent AI\
├── .env / .env.example          # all keys + config
├── USAGE.md                     # this file
├── requirements.txt
├── requirements-langchain.txt
│
├── ask.py                       # one-shot CLI
├── chat.py                      # multi-turn CLI
├── agent.py                     # load_dotenv helper (+ misc)
│
├── CLI/
│   └── cli.py                   # build_agent, _extract_text helpers
│
├── configuration/
│   └── langchain_system_prompt.py
│
├── skills/
│   ├── README.md
│   └── travel.md                # loaded on demand by load_skill tool
│
├── MCP/
│   ├── Tools.py                 # built-in tools + load_skill
│   ├── mcp_servers.py           # subprocess + adapter logic
│   ├── skills.py                # regex fast-path router
│   ├── sncf-mcp/
│   │   └── sncf_server.py       # OUR SNCF Open Data wrapper
│   ├── flights-mcp/             # vendor clone (ravinahp/flights-mcp)
│   └── flixbus-mcp/             # vendor clone (vlad-ds/flixbus-mcp)
│
└── eval/
    ├── cases.json               # eval suite (11 cases)
    └── langsmith_eval.py        # dataset upload + scored experiment
```

---

## Adding new capabilities

### New MCP server (recommended path for new domains)

1. Drop the server's source in `MCP/<your-mcp>/`
2. Add a block in `MCP/mcp_servers.py::_enabled_servers()` that returns the
   stdio command + env vars when the relevant API key is present
3. Done — `MCP/Tools.py` splat-imports all loaded MCP tools automatically

### New skill (for LLM domain expertise without a dedicated tool)

1. Drop `skills/<your-skill>.md` with the focused instructions
2. The LLM calls `load_skill(name='your-skill')` when needed — no code change
3. Mention the skill name in the base system prompt so the LLM knows when to load it

### New regex fast-path

Edit `MCP/skills.py` — add to `_MODE_KEYWORDS` and add a `run_<mode>()`
handler. The regex shape "X to Y on DATE" already covers a lot of intents;
just add the dispatch.

---

## Known issues / limitations

| Issue | Why | Status |
|---|---|---|
| FlixBus free RapidAPI tier returns empty for many routes | Their plan limit, not our code | Skip / pay for higher tier |
| SNCF Open Data has ~3-week forward window | Upstream API limit | Tool returns "dataset doesn't extend that far" gracefully |
| First MCP load takes 5-10s | 4 stdio subprocesses spawn | Cold-start cost; per-query cost is much lower |
| `langgraph dev` doesn't work | Our adapter uses `asyncio.run()`, conflicts with uvicorn loop | Use `ask.py`/`chat.py` instead |
| Ollama Cloud occasionally returns "I apologize, I can't access" despite tool returning data | Model flakiness in long multi-turn flows | Retry usually fixes it; the regex fast-path avoids the LLM entirely for common shapes |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[mcp] failed to load MCP tools` | Stdio race on cold-start | Retry — our adapter has 3-attempt fallback |
| `Failed to parse JSONRPC message` red wall | An MCP server printed banner to stdout instead of stderr | We patched flixbus-mcp already; if a new MCP misbehaves, fix its `print()` to use `file=sys.stderr` |
| `LANGSMITH_API_KEY not set` from `eval/langsmith_eval.py` | Self-explanatory | Set it in `.env` |
| Past-date error from `search_flights` | Model used wrong year | Our adapter intercepts and returns a self-explanatory error — model retries with correct year |
| FlixBus returns HTTP 471 | Date in wrong format | Our adapter auto-converts ISO `YYYY-MM-DD` → `DD.MM.YYYY` |
| Skill regex doesn't catch a shape | Free-text doesn't match the pattern | LLM fallback handles it; or add the pattern to `MCP/skills.py` |

---

## Credits & licenses

- **Our code** (`sncf-mcp`, `skills.py`, the agents and adapters): MIT
- **flights-mcp** (vendor): <https://github.com/ravinahp/flights-mcp> (MIT)
- **flixbus-mcp** (vendor): <https://github.com/vlad-ds/flixbus-mcp>
- **APIs:** SNCF Open Data (Navitia), Duffel, FlixBus (via RapidAPI), Brave Search
