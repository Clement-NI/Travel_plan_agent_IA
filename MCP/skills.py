"""Skill-based fast path for common travel queries.

The LLM round-trip on Ollama Cloud is 5-20 seconds per turn. A "plan a
trip" query goes through 3-4 turns plus all the tool latency, ending up
around 30-50 seconds. For queries that match a well-known shape, we can
skip the LLM entirely:

  1. parse the user's free-text input with regex
  2. resolve dates + IATA codes deterministically
  3. invoke the MCP tools directly (`from MCP.Tools import TOOLS`)
  4. format the result and return

Total time: dominated by MCP subprocess spawn (~3s) + one API call per
tool (~2s) = ~5-8s for a single-mode query, ~10s for a "plan" query with
3 modes in parallel.

If the input doesn't match any known shape, `handle()` returns None and
the caller falls back to the full LLM agent path.
"""
from __future__ import annotations

import calendar
import difflib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as _date, timedelta
from typing import Optional


# Common single-word date phrases that users might typo. We fuzzy-match
# trailing words in the destination against this list with a tight
# similarity threshold — catches "tomorrw", "tomorrrow", "tommorow",
# "wednsday", etc., without hijacking legitimate city names that happen
# to vaguely resemble these.
_DATE_WORDS = (
    "today", "tomorrow", "yesterday",
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
)


def _correct_date_typo(word: str) -> Optional[str]:
    """Return the closest match in `_DATE_WORDS` if `word` is a plausible
    typo (similarity >= 0.78), else None.

    The 0.78 threshold catches one-character drops/swaps ("tomorrw",
    "tommorow", "wednsday") while staying conservative enough not to
    misfire on station names like "Tomelloso" or city names like
    "Toronto".
    """
    if not word or word.lower() in _DATE_WORDS:
        return None
    matches = difflib.get_close_matches(word.lower(), _DATE_WORDS, n=1, cutoff=0.78)
    return matches[0] if matches else None


# ---------- city → IATA mapping (mirror of system prompt) ----------

IATA: dict[str, str] = {
    "paris": "CDG", "lyon": "LYS", "marseille": "MRS", "toulouse": "TLS",
    "nice": "NCE", "bordeaux": "BOD", "nantes": "NTE", "strasbourg": "SXB",
    "berlin": "BER", "madrid": "MAD", "barcelona": "BCN", "rome": "FCO",
    "milan": "MXP", "naples": "NAP", "london": "LHR", "amsterdam": "AMS",
    "frankfurt": "FRA", "munich": "MUC", "brussels": "BRU", "vienna": "VIE",
    "zurich": "ZRH", "geneva": "GVA", "lisbon": "LIS", "athens": "ATH",
    "copenhagen": "CPH", "stockholm": "ARN", "oslo": "OSL", "helsinki": "HEL",
    "dublin": "DUB", "warsaw": "WAW", "prague": "PRG", "budapest": "BUD",
    "istanbul": "IST",
}


# ---------- date phrase → ISO date ----------

_WEEKDAY = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTH = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTH.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})  # jan/feb/...


def parse_date(phrase: str, *, today: Optional[_date] = None) -> Optional[_date]:
    """Convert a free-text date phrase to a `date`. Returns None on failure.

    Handled shapes (case-insensitive):
      • "2026-05-20"                      ISO date
      • "today" / "tomorrow"
      • "next monday" / "this friday"     next/this <weekday>
      • "in 3 days" / "in 2 weeks"
      • "may 20" / "20 may" / "may 20 2026" / "20 may 2026"
      • empty / None                      → today
    """
    today = today or _date.today()
    if not phrase or not phrase.strip():
        return today
    s = phrase.strip().lower()

    # ISO date
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if iso_match:
        try:
            return _date(int(iso_match.group(1)),
                         int(iso_match.group(2)),
                         int(iso_match.group(3)))
        except ValueError:
            return None

    if s == "today":
        return today
    if s == "tomorrow":
        return today + timedelta(days=1)
    if s in ("day after tomorrow", "the day after tomorrow"):
        return today + timedelta(days=2)

    # next/this <weekday>
    m = re.fullmatch(r"(next|this|on|this coming|coming)\s+(\w+)", s)
    if m and m.group(2) in _WEEKDAY:
        target = _WEEKDAY[m.group(2)]
        delta = (target - today.weekday()) % 7
        if delta == 0 or m.group(1) == "next":
            delta = delta or 7
        return today + timedelta(days=delta)

    # bare weekday name → next occurrence
    if s in _WEEKDAY:
        delta = (_WEEKDAY[s] - today.weekday()) % 7
        delta = delta or 7
        return today + timedelta(days=delta)

    # "in N days/weeks"
    m = re.fullmatch(r"in\s+(\d+)\s+(day|days|week|weeks)", s)
    if m:
        n = int(m.group(1))
        return today + timedelta(days=n * (7 if "week" in m.group(2) else 1))

    # "may 20" / "may 20 2026" / "20 may" / "20 may 2026"
    for pat in (
        r"([a-z]+)\s+(\d{1,2})(?:\s+(\d{4}))?",   # may 20 [2026]
        r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?",   # 20 may [2026]
    ):
        m = re.fullmatch(pat, s)
        if m:
            groups = m.groups()
            if groups[0].isdigit():
                day, month_name, year = int(groups[0]), groups[1], groups[2]
            else:
                month_name, day, year = groups[0], int(groups[1]), groups[2]
            if month_name not in _MONTH:
                continue
            month = _MONTH[month_name]
            year_int = int(year) if year else today.year
            try:
                d = _date(year_int, month, day)
            except ValueError:
                return None
            # If month/day already passed THIS year and the user didn't
            # explicitly give a year, bump to next year.
            if not year and d < today:
                d = _date(year_int + 1, month, day)
            return d

    return None


# ---------- query → (mode, from_city, to_city, date) ----------

# Mode keywords. Order matters — more specific first.
_MODE_KEYWORDS = {
    "train":  r"\b(tgv|train|trains|rail|sncf|ouigo|intercit[ée]s|ter)\b",
    "flight": r"\b(flight|flights|fly|flying|plane|airline|airfare|airplane)\b",
    "bus":    r"\b(bus|buses|coach|flixbus)\b",
    "plan":   r"\b(plan|planning|itinerary|trip|options|alternatives|"
              r"all options|best way|how (?:to|do i) get)\b",
}


def _detect_mode(text: str) -> str:
    """Pick the most specific transport mode the query asks about.

    Single explicit mode wins. Multiple mentioned → 'plan'. None → 'plan'
    (default for bare "X to Y").
    """
    s = text.lower()
    hits = [m for m, pat in _MODE_KEYWORDS.items() if re.search(pat, s) and m != "plan"]
    if len(hits) == 1:
        return hits[0]
    if len(hits) >= 2:
        return "plan"
    if re.search(_MODE_KEYWORDS["plan"], s):
        return "plan"
    return "plan"  # bare "X to Y" defaults to plan


# Match "<stuff> from CITY to CITY [on/for DATE]" with non-greedy capture.
# `to` allows digits / hyphens / dots so a bare-trailing date phrase like
# "2026-05-25" or "May 25" gets captured into `to`; `_split_trailing_date`
# peels it off afterwards.
_ROUTE_RE = re.compile(
    r"\bfrom\s+(?P<from>[A-Za-zÀ-ÿ\-'\s]+?)\s+to\s+(?P<to>[A-Za-zÀ-ÿ\d\-'\.\s]+?)"
    r"(?:\s+(?:on|for|at)\s+(?P<date>.+?))?\s*\??\s*$",
    re.IGNORECASE,
)
# Also accept "CITY to CITY" without "from" prefix.
_ROUTE_RE_BARE = re.compile(
    r"^(?:[A-Za-zÀ-ÿ\-'\s]*?\b)??(?P<from>[A-Z][A-Za-zÀ-ÿ\-']+(?:\s+[A-Z][A-Za-zÀ-ÿ\-']+)*)"
    r"\s+to\s+(?P<to>[A-Z][A-Za-zÀ-ÿ\-']+(?:\s+[A-Z\d][A-Za-zÀ-ÿ\d\-'\.]+)*)"
    r"(?:\s+(?:on|for|at)\s+(?P<date>.+?))?\s*\??\s*$"
)


def _split_trailing_date(text: str, today: _date) -> tuple[str, str]:
    """Peel a trailing date phrase off `text`.

    Users often write "Paris to Lyon tomorrow" / "Paris to Lyon next
    Monday" / "Paris to Lyon 2026-05-25" without an "on"/"for"/"at"
    separator. Our `_ROUTE_RE` regex doesn't capture those date phrases,
    so they end up glued onto the `to` city ("Lyon tomorrow").

    This helper tries progressively shorter suffixes (longest first, to
    prefer "next Monday" over the bare "Monday") and returns
    (city_without_date, date_phrase) once `parse_date` accepts one.

    Returns (text, "") unchanged if no trailing date phrase is found.
    """
    parts = text.split()
    if not parts:
        return text, ""
    # Try suffix lengths from longest to shortest (max 4 words —
    # "in 3 days" / "next Monday" / "may 25 2026" all fit in 3).
    for n in range(min(4, len(parts)), 0, -1):
        candidate = " ".join(parts[-n:])
        parsed = parse_date(candidate, today=today)
        if parsed is not None and parsed >= today:
            remaining = " ".join(parts[:-n]).strip()
            if remaining:
                return remaining, candidate

    # Typo recovery: trailing single word might be a misspelled date
    # ("tomorrw" → "tomorrow", "wednsday" → "wednesday"). If we can
    # confidently auto-correct it, peel + return the corrected form.
    if len(parts) >= 2:
        corrected = _correct_date_typo(parts[-1])
        if corrected:
            # "next mondy" / "this mondy" — also peel the modifier word
            if len(parts) >= 3 and parts[-2].lower() in ("next", "this", "coming"):
                combined = f"{parts[-2].lower()} {corrected}"
                if parse_date(combined, today=today) is not None:
                    remaining = " ".join(parts[:-2]).strip()
                    return remaining, combined
            # plain "tomorrw" / "wednsday" — peel just the typo
            if parse_date(corrected, today=today) is not None:
                remaining = " ".join(parts[:-1]).strip()
                return remaining, corrected

    return text, ""


def parse_query(text: str) -> Optional[dict]:
    """Extract (mode, from_city, to_city, date) from a free-text query.

    Returns dict or None if the query doesn't look like an O/D travel
    request. The caller treats None as "fall back to LLM".
    """
    s = text.strip()
    if not s:
        return None

    m = _ROUTE_RE.search(s)
    if not m:
        m = _ROUTE_RE_BARE.search(s)
    if not m:
        return None

    from_city = m.group("from").strip()
    to_city = m.group("to").strip()
    date_phrase = (m.group("date") or "").strip()

    # Clean: drop trailing punctuation, leading articles
    from_city = re.sub(r"^(?:the\s+)", "", from_city, flags=re.IGNORECASE)
    to_city = re.sub(r"^(?:the\s+)", "", to_city, flags=re.IGNORECASE)

    today = _date.today()

    # If the regex didn't capture an explicit date (no "on/for/at"
    # separator), try to peel a trailing date phrase off `to_city`.
    # Handles "Paris to Lyon tomorrow", "Paris to Lyon next Monday",
    # "Paris to Lyon 2026-05-25", etc.
    if not date_phrase:
        new_to, trailing = _split_trailing_date(to_city, today)
        if trailing:
            to_city = new_to
            date_phrase = trailing

    parsed_date = parse_date(date_phrase, today=today)
    if parsed_date is None:
        return None  # unrecognized date phrase → punt to LLM
    if parsed_date < today:
        return None  # past date → punt to LLM with rejection

    return {
        "mode":  _detect_mode(s),
        "from":  from_city,
        "to":    to_city,
        "date":  parsed_date,
    }


# ---------- mode handlers (call MCP tools directly) ----------

def _iata(city: str) -> Optional[str]:
    return IATA.get(city.lower().strip())


def _get_tool(name: str):
    """Lazy lookup so importing skills.py doesn't trigger MCP loading."""
    from MCP.Tools import TOOLS
    for t in TOOLS:
        if t.name == name:
            return t
    return None


def run_train(from_city: str, to_city: str, date: _date) -> str:
    tool = _get_tool("plan_train_journey")
    if tool is None:
        return "[skill error: plan_train_journey tool not loaded]"
    return tool.invoke({
        "from_city": from_city,
        "to_city": to_city,
        "departure_date": date.isoformat(),
    })


def run_flight(from_city: str, to_city: str, date: _date) -> str:
    origin = _iata(from_city)
    destination = _iata(to_city)
    if not origin:
        return f"[skill: no IATA code known for {from_city!r}; falling back to LLM]"
    if not destination:
        return f"[skill: no IATA code known for {to_city!r}; falling back to LLM]"
    tool = _get_tool("search_flights")
    if tool is None:
        return "[skill error: search_flights tool not loaded]"
    return tool.invoke({
        "type": "one_way",
        "origin": origin,
        "destination": destination,
        "departure_date": date.isoformat(),
    })


def run_bus(from_city: str, to_city: str, date: _date) -> str:
    """Resolve location IDs sequentially (search_trips depends on them),
    then call search_trips."""
    locate = _get_tool("search_locations")
    trips = _get_tool("search_trips")
    if locate is None or trips is None:
        return "[skill error: FlixBus tools not loaded]"
    import json

    def _pick_id(raw: str, city: str) -> Optional[str]:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, list) or not data:
            return None

        # Multi-stage filter — relax progressively until we find something.
        city_lc = city.lower()

        def _not_airport(r):
            return "airport" not in (r.get("name") or "").lower()

        # Tier 1: non-airport stop where city matches
        tier1 = [r for r in data if _not_airport(r)
                 and city_lc in (r.get("city") or "").lower()]
        # Tier 2: any stop where city matches (allows train stations)
        tier2 = [r for r in data if city_lc in (r.get("city") or "").lower()]
        # Tier 3: non-airport stop, any city (FlixBus returns near-matches)
        tier3 = [r for r in data if _not_airport(r)]
        # Tier 4: anything FlixBus gave us
        candidates = tier1 or tier2 or tier3 or list(data)

        candidates.sort(key=lambda r: -(r.get("importance") or 0))
        return candidates[0].get("id") if candidates else None

    paris_raw = locate.invoke({"query": from_city})
    dest_raw  = locate.invoke({"query": to_city})
    from_id = _pick_id(paris_raw, from_city)
    to_id   = _pick_id(dest_raw,  to_city)
    if not from_id or not to_id:
        return (f"[skill: FlixBus didn't return resolvable stops for "
                f"{from_city} → {to_city}; falling back to LLM]")
    return trips.invoke({
        "from_id": from_id,
        "to_id": to_id,
        "date": date.isoformat(),
        "adult": 1,
    })


def run_plan(from_city: str, to_city: str, date: _date) -> str:
    """Fire all three modes in parallel via a thread pool."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            "flight": pool.submit(run_flight, from_city, to_city, date),
            "train":  pool.submit(run_train,  from_city, to_city, date),
            "bus":    pool.submit(run_bus,    from_city, to_city, date),
        }
        results: dict[str, str] = {}
        for mode, fut in futures.items():
            try:
                results[mode] = fut.result(timeout=120)
            except Exception as exc:
                results[mode] = f"[skill error in {mode}: {type(exc).__name__}: {exc}]"

    parts = [
        f"### ✈ Flights\n{results['flight']}",
        f"### 🚆 Trains\n{results['train']}",
        f"### 🚌 Buses\n{results['bus']}",
    ]
    return "\n\n".join(parts)


# ---------- public entry point ----------

def handle(text: str) -> Optional[str]:
    """Try to answer the query without invoking the LLM.

    Returns the formatted reply on success, or None if the query doesn't
    match a known shape (caller should fall back to the LLM agent).
    """
    parsed = parse_query(text)
    if parsed is None:
        return None

    mode = parsed["mode"]
    from_city = parsed["from"]
    to_city = parsed["to"]
    date = parsed["date"]

    header = f"[skill: {mode}]  {from_city} → {to_city} on {date.isoformat()}\n"

    if mode == "train":
        body = f"### 🚆 Trains\n{run_train(from_city, to_city, date)}"
    elif mode == "flight":
        flight = run_flight(from_city, to_city, date)
        # If flight failed (e.g. no IATA), punt to LLM
        if flight.startswith("[skill:"):
            return None
        body = f"### ✈ Flights\n{flight}"
    elif mode == "bus":
        bus = run_bus(from_city, to_city, date)
        if bus.startswith("[skill:"):
            return None
        body = f"### 🚌 Buses\n{bus}"
    elif mode == "plan":
        body = run_plan(from_city, to_city, date)
    else:
        return None  # unknown mode

    return header + body
