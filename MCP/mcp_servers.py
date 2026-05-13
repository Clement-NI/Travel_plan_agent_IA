"""External MCP server integration.

We connect to MCP servers (Model Context Protocol) that someone else built
and maintains, and expose their tools to our LangGraph agent — same shape
as our local @tool functions.

Each MCP server runs as a subprocess (Node via npx, Python module, or `uv
run`) and speaks JSON-RPC over stdio. `MultiServerMCPClient` manages the
lifecycles and returns LangChain-compatible BaseTool instances.

Config rule: each server entry is conditionally included based on whether
its required env vars are present, so a missing API key or path just skips
that server instead of crashing the agent at startup.

### Travel stack (the 3 transport MCPs)

We delegate ALL transport queries to three external MCPs, each cloned
locally and pointed at via env vars:

  • `ravinahp/flights-mcp`  — Duffel-backed flight search
      env: FLIGHTS_MCP_PATH (clone dir), DUFFEL_API_KEY_LIVE
  • `Kryzo/mcp-sncf`         — SNCF/Navitia train journeys + station info
      env: SNCF_MCP_PATH (clone dir), SNCF_API_TOKEN
  • `vlad-ds/flixbus-mcp`   — FlixBus intercity buses (RapidAPI)
      env: FLIXBUS_MCP_PATH (clone dir), RAPID_API_KEY

If a server's path or key is missing, it's silently skipped — the rest of
the agent still works.

Public function: `get_mcp_tools()` — `MCP/Tools.py` splats this into TOOLS.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import site
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.tools import BaseTool, StructuredTool


# Default location of the 3 transport MCP servers — they ship inside this
# repo, parallel to mcp_servers.py. Env vars (`FLIGHTS_MCP_PATH`,
# `SNCF_MCP_PATH`, `FLIXBUS_MCP_PATH`) override this, but they're
# optional now that everything lives in the project.
_DEFAULT_MCP_BASE = Path(__file__).parent


def _enabled_servers() -> Dict[str, Dict[str, Any]]:
    """Return the subset of MCP server configs whose env vars + paths are satisfied."""
    servers: Dict[str, Dict[str, Any]] = {}

    # ---- Brave Search (Node, free 2000/mo, sign up: https://brave.com/search/api/) ----
    brave_key = os.environ.get("BRAVE_API_KEY")
    if brave_key:
        servers["brave_search"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": brave_key},
            "transport": "stdio",
        }

    # ---- Google Maps (Node, free $200/mo credit, https://developers.google.com/maps) ----
    maps_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if maps_key:
        servers["google_maps"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-google-maps"],
            "env": {"GOOGLE_MAPS_API_KEY": maps_key},
            "transport": "stdio",
        }

    # ---- Flights MCP (Duffel-backed) — github.com/ravinahp/flights-mcp ----
    # Tools: search_flights, get_offer_details, search_multi_city
    #
    # We invoke the venv's python.exe directly rather than going through
    # `uv run`. Reasons:
    #   1. `uv run` adds 5-15s of project-resolution overhead on every
    #      spawn, which exceeded the MCP initialize timeout on slower
    #      runs and showed up as `TaskGroup` failures during loading.
    #   2. The upstream `flights-mcp` console script wraps the sync
    #      `server.main()` in `asyncio.run()`, which crashes on shutdown.
    #      `python -m flights.server` runs server.py's own __main__ guard
    #      and avoids that wrapper.
    flights_path = os.environ.get("FLIGHTS_MCP_PATH") or str(_DEFAULT_MCP_BASE / "flights-mcp")
    duffel_key = os.environ.get("DUFFEL_API_KEY_LIVE")
    if flights_path and duffel_key and Path(flights_path).is_dir():
        # Prefer the vendor's own .venv (local dev path). On Streamlit
        # Cloud / Docker / any PaaS, that .venv doesn't exist — we fall
        # back to the main interpreter (`sys.executable`) and rely on
        # the top-level requirements.txt to provide flights-mcp's deps
        # (httpx, pydantic, mcp).
        venv_python = Path(flights_path) / ".venv" / "Scripts" / "python.exe"
        if not venv_python.is_file():
            venv_python = Path(flights_path) / ".venv" / "bin" / "python"
        command = str(venv_python) if venv_python.is_file() else sys.executable
        servers["flights"] = {
            "command": command,
            "args": ["-m", "flights.server"],
            "cwd": flights_path,
            "env": {**os.environ.copy(), "DUFFEL_API_KEY_LIVE": duffel_key,
                    "PYTHONPATH": str(Path(flights_path) / "src")},
            "transport": "stdio",
        }

    # ---- SNCF MCP (our own minimal Navitia client) ----
    # Server lives at D:\MCP-servers\sncf-mcp\sncf_server.py (parallel to
    # the other 3 MCP servers). Tools: plan_train_journey,
    # find_train_station.
    #
    # We replaced `Kryzo/mcp-sncf` after discovering it (a) used CSV row
    # numbers as SNCF station IDs (so any city not in its 11-entry
    # hardcoded list returned `stop_area:SNCF:<row_number>` → 404), and
    # (b) had several mislabeled IDs in that hardcoded list (Lyon
    # Part-Dieu/Perrache swapped, Toulouse→Montpellier, etc.). Our server
    # talks to SNCF's /places API directly, so any of the ~3000 French
    # stations plus cross-border destinations (Barcelona, Brussels,
    # Geneva, Frankfurt) resolve dynamically — no static list.
    sncf_path = os.environ.get("SNCF_MCP_PATH") or str(_DEFAULT_MCP_BASE / "sncf-mcp")
    sncf_key = os.environ.get("SNCF_API_TOKEN")
    if sncf_path and sncf_key:
        sncf_script = Path(sncf_path) / "sncf_server.py"
        if sncf_script.is_file():
            servers["sncf"] = {
                "command": _python_cmd(),
                "args": [str(sncf_script)],
                "cwd": str(sncf_path),
                "env": {**os.environ.copy(), "SNCF_API_TOKEN": sncf_key},
                "transport": "stdio",
            }

    # ---- FlixBus MCP (RapidAPI) — github.com/vlad-ds/flixbus-mcp ----
    # Tools: search_locations, search_trips, get_station_timetable
    # Same `uv run` bypass as flights-mcp above — call venv python directly.
    flixbus_path = os.environ.get("FLIXBUS_MCP_PATH") or str(_DEFAULT_MCP_BASE / "flixbus-mcp")
    rapid_key = os.environ.get("RAPID_API_KEY")
    if flixbus_path and rapid_key and Path(flixbus_path).is_dir():
        # Same vendor-venv-or-main-interpreter fallback as flights-mcp above.
        venv_python = Path(flixbus_path) / ".venv" / "Scripts" / "python.exe"
        if not venv_python.is_file():
            venv_python = Path(flixbus_path) / ".venv" / "bin" / "python"
        command = str(venv_python) if venv_python.is_file() else sys.executable
        servers["flixbus"] = {
            "command": command,
            "args": ["server.py"],
            "cwd": flixbus_path,
            "env": {**os.environ.copy(), "RAPID_API_KEY": rapid_key},
            "transport": "stdio",
        }

    return servers


# ---------- launcher helpers ----------

def _uv_invocation() -> Tuple[str, List[str]]:
    """Return `(command, prefix_args)` to invoke `uv`.

    Order:
      1. Explicit override via `UV_EXE` env var.
      2. `uv` on PATH (the normal case — `shutil.which`).
      3. `uv.exe` in the current Python's pip-`--user` Scripts dir
         (Windows location when uv was installed via `pip install --user uv`).
      4. Fallback: `<sys.executable> -m uv`. Works as long as uv is importable
         by the current interpreter, even if the .exe shim isn't on PATH.

    Returning a (cmd, prefix) tuple lets the fallback prepend `-m uv` to the
    args list; the on-PATH case returns an empty prefix.
    """
    override = os.environ.get("UV_EXE")
    if override and Path(override).is_file():
        return override, []

    on_path = shutil.which("uv")
    if on_path:
        return on_path, []

    # pip --user install on Windows lands here; on Linux/macOS it'd be
    # site.getuserbase() + "/bin/uv". `USER_SITE` gives a sibling dir we can
    # walk up from to find Scripts/.
    user_site = site.getusersitepackages()
    if user_site:
        for candidate in (
            Path(user_site).parent / "Scripts" / "uv.exe",  # Windows --user
            Path(user_site).parent / "bin" / "uv",          # POSIX --user
        ):
            if candidate.is_file():
                return str(candidate), []

    # Last resort: `python -m uv`. Slower (extra interpreter spin-up) but
    # robust — works whenever `import uv` succeeds.
    return sys.executable, ["-m", "uv"]


def _python_cmd() -> str:
    """Resolve a Python interpreter for child MCP servers.

    Order: explicit override (`MCP_PYTHON`), then the current interpreter, then
    `py` on Windows / `python3` elsewhere. We avoid `sys.executable` directly
    because on Windows that's often a venv-scoped python that won't have the
    target server's deps; users typically install deps into the system py.
    """
    override = os.environ.get("MCP_PYTHON")
    if override:
        return override
    return shutil.which("py") or shutil.which("python3") or shutil.which("python") or "python"


def get_mcp_tools() -> List[BaseTool]:
    """Connect to all configured MCP servers and return their tools.

    Called at module load time from `MCP.Tools` so the agent sees these
    tools as if they were local @tool functions. If no MCP servers are
    configured (no API keys / paths), returns an empty list.

    SEQUENTIAL spawn (not concurrent): the upstream
    `MultiServerMCPClient.get_tools()` uses `asyncio.gather()` to spawn all
    stdio subprocesses at once, which on Windows races so hard the
    initialize handshakes drop and the adapter's own cleanup code throws
    `UnboundLocalError`. We call `load_mcp_tools` once per server, each
    inside its own `asyncio.run`, so each subprocess gets a fresh,
    uncontested event loop. Slower by a few seconds at startup, but
    reliable. Per-server failures are tolerated — the rest still load.
    """
    config = _enabled_servers()
    if not config:
        return []

    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError:
        print("[mcp] langchain-mcp-adapters not installed; skipping MCP tools.")
        return []

    async def _load_one(conn: Dict[str, Any]) -> List[BaseTool]:
        # Pass `connection=` (not a live session) so each returned tool spawns
        # a fresh subprocess per invocation — same per-call lifecycle that
        # MultiServerMCPClient uses, but driven sequentially per server.
        return await load_mcp_tools(session=None, connection=conn)

    all_tools: List[BaseTool] = []
    for name, conn in config.items():
        last_exc: Optional[Exception] = None
        for attempt in range(1, 3):  # 2 tries per server
            try:
                server_tools = asyncio.run(_load_one(conn))
                all_tools.extend(server_tools)
                if attempt > 1:
                    print(f"[mcp] {name}: recovered on attempt {attempt}")
                break
            except Exception as exc:
                last_exc = exc
                kind = type(exc).__name__
                if attempt < 2:
                    # Brief settle period before retrying just this one.
                    import time as _t
                    _t.sleep(0.5)
                else:
                    print(f"[mcp] {name} failed: {kind}: {str(exc)[:100]}")
        _ = last_exc

    if not all_tools:
        print("[mcp] no MCP tools loaded — agent will run with built-in tools only.")
        return []

    # MCP adapter v0.2.x returns structured [{'type':'text','text':...}]
    # content. Many LLMs (Gemini flash, smaller Ollama models) don't
    # interpret that as the actual tool output — they see the JSON-shape
    # wrapper and skip the data. Flatten to plain string here.
    all_tools = [_flatten_output(t) for t in all_tools]
    print(f"[mcp] loaded {len(all_tools)} tool(s) from {len(config)} server(s): "
          f"{', '.join(t.name for t in all_tools)}")
    return all_tools


def _flatten_mcp_content(content) -> str:
    """Coerce an MCP-style structured content blob into a flat string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _check_future_flight_date(kwargs: dict) -> Optional[str]:
    """Return an error string if `departure_date` is in the past, else None.

    Smaller LLMs (qwen, llama) often call `current_time` correctly but then
    emit a `departure_date` with a stale year (`2024-…` / `2025-…`) anyway —
    a pattern baked in from training data. Duffel responds with HTTP 422,
    which the MCP adapter wraps as a bare ToolException; the LLM has no way
    to tell what went wrong and falls back to web_search.

    Catching it here lets us return a specific, actionable message:
        "Error: departure_date 2025-05-20 is in the past. Today is 2026-05-11.
         Use the year from current_time and retry."

    The LLM sees this as a tool response, not an exception, and will re-call
    with a corrected year on its next turn.
    """
    from datetime import date as _date
    dep = kwargs.get("departure_date")
    if not isinstance(dep, str):
        return None
    try:
        dep_d = _date.fromisoformat(dep)
    except ValueError:
        return (f"Error: departure_date {dep!r} is not in YYYY-MM-DD format. "
                f"Retry with an ISO date like '{_date.today().isoformat()}'.")
    today = _date.today()
    if dep_d < today:
        return (
            f"Error: departure_date {dep} is in the past. Today is "
            f"{today.isoformat()}. You probably used the wrong year — "
            f"check the year from current_time and retry with a date "
            f"on or after {today.isoformat()}."
        )
    return None


def _summarize_flight_offers(raw: str) -> str:
    """Compress flights-mcp's verbose JSON into a one-line-per-offer table.

    flights-mcp returns a ~30-40 KB JSON blob with deeply nested offers,
    slices, connections, baggage allowances, etc. LLMs (especially smaller
    ones like gemini-2.5-flash) tend to refuse to summarize this much
    structured data — they hallucinate a generic "tool failed" message.

    We reduce each offer to one line: price, departure→arrival times,
    duration, carrier, stops. The full JSON is still available if the LLM
    asks for details on a specific `offer_id` via `get_offer_details`.

    If parsing fails (schema drift), fall back to the original string so
    nothing is silently lost.
    """
    import json
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    offers = data.get("offers") if isinstance(data, dict) else None
    if not isinstance(offers, list) or not offers:
        # Could be an error dict ({"error": ...}) — pass through.
        return raw

    lines = [f"Found {len(offers)} flight offer(s):"]
    for o in offers:
        price = (o.get("price") or {})
        amount = price.get("amount", "?")
        currency = price.get("currency", "")
        slices = o.get("slices") or []
        if not slices:
            continue
        # Multi-slice (round-trip / multi-city) → render each leg.
        for i, sl in enumerate(slices):
            origin = sl.get("origin", "?")
            destination = sl.get("destination", "?")
            dep = (sl.get("departure") or "")[:16].replace("T", " ")
            arr = (sl.get("arrival") or "")[:16].replace("T", " ")
            duration = _humanize_iso_duration(sl.get("duration", ""))
            carrier = sl.get("carrier", "?")
            stops = sl.get("stops_description", "?")
            leg_label = f"leg {i+1}/{len(slices)}: " if len(slices) > 1 else ""
            price_label = f"{currency}{amount}" if i == 0 else " " * (len(currency) + len(amount))
            lines.append(
                f"  {price_label:>10s}  {leg_label}{origin}→{destination}  "
                f"{dep} → {arr}  ({duration})  {carrier}  [{stops}]  id={o.get('offer_id','')}"
            )
    return "\n".join(lines)


def _humanize_iso_duration(s: str) -> str:
    """`PT1H43M` → `1h43m`. `PT7H5M` → `7h05m`. Pass anything else through."""
    if not s.startswith("PT"):
        return s
    body = s[2:]
    h = m = 0
    if "H" in body:
        h_part, _, body = body.partition("H")
        try:
            h = int(h_part)
        except ValueError:
            return s
    if "M" in body:
        m_part, _, _ = body.partition("M")
        try:
            m = int(m_part)
        except ValueError:
            return s
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _to_european_date(s: str) -> str:
    """Convert `YYYY-MM-DD` → `DD.MM.YYYY`. Pass anything else through unchanged.

    FlixBus's RapidAPI endpoint rejects ISO dates with HTTP 471; our agent
    (and any reasonable LLM caller) emits ISO. This single-purpose conversion
    sits in the adapter so the upstream MCP doesn't have to change.
    """
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return f"{s[8:10]}.{s[5:7]}.{s[0:4]}"
    return s


def _needs_params_wrapper(tool: BaseTool) -> bool:
    """True if this MCP tool's top-level schema is `{params: <model>}`.

    flights-mcp (Duffel) wraps every tool's args in a single `params` Pydantic
    model — its tools have `properties=['params']` / `required=['params']`. The
    LLM consistently forgets this wrapper and calls them with flat kwargs,
    which fails MCP-side validation. We detect the pattern from the schema
    and auto-wrap at call time so the LLM never has to know.

    SNCF and FlixBus tools take flat args, so this returns False for them.
    """
    try:
        schema = tool.args_schema
        if not isinstance(schema, dict) and hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        if not isinstance(schema, dict):
            return False
        props = list((schema.get("properties") or {}).keys())
        required = schema.get("required") or []
        return props == ["params"] and required == ["params"]
    except Exception:
        return False


def _flatten_output(tool: BaseTool) -> BaseTool:
    """Wrap a tool so its returned content is always a plain string.

    Two transforms:
      1. Sync bridge: MCP-adapter tools are async-only — they don't support
         sync `.invoke()`. The sync `func` bridges via `asyncio.run()` to
         call the async version. Works when called from sync contexts (most
         LangGraph agent paths).
      2. Auto-wrap: tools whose schema is `{params: <model>}` (flights-mcp's
         pattern) get their flat kwargs re-wrapped under `params` before the
         underlying MCP call, so the LLM can keep using flat args.
    """
    orig = tool
    wrap_params = _needs_params_wrapper(orig)
    # flixbus-mcp's /trips and /schedule endpoints require date in DD.MM.YYYY
    # format (verified by hitting RapidAPI directly — wrong format returns
    # `471 Invalid date (should be DD.MM.YYYY)`). The LLM naturally emits ISO,
    # so we silently rewrite it for the two affected tools.
    fix_date_format = orig.name in {"search_trips", "get_station_timetable"}

    def _prepare(kwargs: dict) -> dict:
        if fix_date_format and isinstance(kwargs.get("date"), str):
            kwargs = {**kwargs, "date": _to_european_date(kwargs["date"])}
        # If caller already passed `params`, trust it. Otherwise rewrap.
        if wrap_params and "params" not in kwargs:
            return {"params": kwargs}
        return kwargs

    # flights-mcp returns ~38 KB of nested JSON per search — most LLMs (Gemini
    # Flash especially) get overwhelmed and refuse to summarize. Compress to
    # a one-line-per-offer table that the LLM can paste through verbatim.
    compress_output = orig.name == "search_flights"
    validate_flight_date = orig.name == "search_flights"

    async def _async(**kwargs):
        # Pre-flight check: past dates bounce off Duffel with HTTP 422 which
        # the MCP adapter wraps as a generic ToolException — opaque enough
        # that LLMs give up instead of retrying with a corrected year. Catch
        # it here and return a self-explanatory message so the LLM can fix
        # the call (this triggers a lot with smaller models that ignore
        # current_time's year and emit '2025-…' out of training-data habit).
        if validate_flight_date:
            err = _check_future_flight_date(kwargs)
            if err:
                return err

        raw = _flatten_mcp_content(await orig.ainvoke(_prepare(kwargs)))
        if compress_output:
            raw = _summarize_flight_offers(raw)
        return raw

    def _sync(**kwargs):
        # Bridge sync → async. asyncio.run creates a fresh event loop.
        # Safe when called from sync code (LangGraph create_agent's sync path).
        return asyncio.run(_async(**kwargs))

    # If we're auto-wrapping, also flatten the schema the LLM sees so it
    # advertises the actual flight-search fields (origin, destination, ...)
    # rather than just a single opaque `params` argument.
    schema = orig.args_schema
    if wrap_params and isinstance(schema, dict):
        inner_schema = _inline_params_schema(schema)
        if inner_schema is not None:
            schema = inner_schema

    return StructuredTool(
        name=orig.name,
        description=orig.description,
        args_schema=schema,
        func=_sync,
        coroutine=_async,
    )


def _inline_params_schema(schema: dict) -> dict | None:
    """Inline the inner `params` model into the top-level schema.

    Turns `{properties: {params: {$ref: '#/$defs/FlightSearchParams'}},
    required: [params], $defs: {FlightSearchParams: {...}}}` into the
    referenced model's own properties/required, so the LLM sees the real
    field names instead of an opaque `params` blob. Returns None if the
    schema doesn't match the expected shape (caller falls back to original).
    """
    try:
        params_prop = (schema.get("properties") or {}).get("params") or {}
        ref = params_prop.get("$ref", "")
        # Expect "#/$defs/<ModelName>"
        if not ref.startswith("#/$defs/"):
            return None
        model_name = ref.split("/")[-1]
        defs = schema.get("$defs") or {}
        inner = defs.get(model_name)
        if not isinstance(inner, dict) or "properties" not in inner:
            return None
        # Build a new top-level schema using the inner model's fields.
        # Preserve $defs so nested refs (e.g. FlightSegment) still resolve.
        out = {
            "type": "object",
            "properties": inner["properties"],
            "required": inner.get("required", []),
            "$defs": defs,
        }
        if "title" in inner:
            out["title"] = inner["title"]
        return out
    except Exception:
        return None
