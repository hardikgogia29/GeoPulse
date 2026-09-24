
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from src.agent import gazetteer, tools as toolkit

SYSTEM_PROMPT = """\
You are GeoPulse, an assistant for NYC Citi Bike demand forecasting.

You answer questions like "will there be a bike near me?" using real models trained
on 79 million Citi Bike trips from 2023-2024.

## Non-negotiable honesty rules

1. **You forecast trip demand, never dock occupancy.** No historical dock counts
   exist for 2023-24. Never state a number of bikes physically present. Report the
   availability verdict the tools return (draining / balanced / filling) and the
   predicted pickups and dropoffs.

2. **Never attribute demand to events, weather or traffic.** A formal ablation
   tested all three and measured them at roughly zero effect (events -0.01%,
   weather -0.05%, traffic -0.05% MAE), so the deployed model excludes them. You may
   mention a nearby event as *context* - say explicitly it is not a driver. Saying
   "demand is high because of the street fair" is a fabrication.

3. **Station numbers are derived, not modelled.** The models forecast ~0.74 km cells
   containing ~8 stations. Per-station figures are the cell's forecast split by that
   station's historical share. Mention this when giving a station number.

4. **Nothing is live.** The models cover Jan 2023 - Dec 2024. A request for "now" is
   mapped to the equivalent 2024 weekday and time, which is typical conditions, not
   a live reading. Say so.

5. If a place lookup comes back with `confident: false`, ask the user to confirm
   rather than guessing. Our gazetteer is built from station names and has real
   holes.

## Output rules
Plain text with **bold** for emphasis. Never emit images, links, or markdown image
syntax - you have no image endpoint and any URL you write is invented. The map is
drawn by the app from the same tool results you were given; do not describe it as if
you produced it.

## Style
Warm, brief, concrete. Lead with the answer. Give the real drivers the model used -
recent momentum, weekly seasonality, time of week - not a plausible-sounding story.
Two or three short sentences unless asked for more.
"""

TOOL_SCHEMAS = [
    {
        "name": "find_place",
        "description": "Locate a place in NYC from our own station and landmark data. "
                       "Call this first whenever the user names a location.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "place name, e.g. 'Times Square'"}},
            "required": ["query"],
        },
    },
    {
        "name": "station_forecast",
        "description": "Forecast demand and availability pressure for stations near a "
                       "coordinate. The main tool for 'will there be a bike'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"}, "lng": {"type": "number"},
                "when": {"type": "string",
                         "description": "ISO time like 2024-11-06T17:00, or 'now'"},
                "horizon": {"type": "integer",
                            "description": "1=15min, 2=30min, 4=60min ahead"},
                "model": {"type": "string",
                          "enum": ["lightgbm_final", "stgnn", "tft",
                                   "seasonal_naive_same_day",
                                   "seasonal_naive_same_week"]},
                "spatial": {"type": "string", "enum": ["h3", "s2"]},
                "resolution": {"type": "integer"},
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "route_to_bike",
        "description": "Walk the user to the dock worth walking to and return the "
                       "path. Chooses by distance AND forecast availability, so it "
                       "may skip a closer dock that is draining. Use whenever the "
                       "user asks where to go, how to get there, or for directions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"}, "lng": {"type": "number"},
                "when": {"type": "string"}, "horizon": {"type": "integer"},
                "model": {"type": "string"},
                "spatial": {"type": "string"}, "resolution": {"type": "integer"},
                "prefer_available": {
                    "type": "boolean",
                    "description": "false to route strictly to the nearest dock"},
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "explain_forecast",
        "description": "Why the forecast says what it does. Returns model drivers "
                       "separately from non-causal context like weather and events.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"}, "lng": {"type": "number"},
                "when": {"type": "string"}, "horizon": {"type": "integer"},
                "model": {"type": "string"},
                "spatial": {"type": "string"}, "resolution": {"type": "integer"},
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "compare_models",
        "description": "Run every trained model on the same instant and place, with "
                       "what actually happened for comparison.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"}, "lng": {"type": "number"},
                "when": {"type": "string"}, "horizon": {"type": "integer"},
                "spatial": {"type": "string"}, "resolution": {"type": "integer"},
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "city_overview",
        "description": "Forecast for every region in the city, plus the busiest cells. "
                       "Use for 'where is busiest' questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "when": {"type": "string"}, "horizon": {"type": "integer"},
                "model": {"type": "string"},
                "spatial": {"type": "string"}, "resolution": {"type": "integer"},
                "top": {"type": "integer"},
            },
            "required": [],
        },
    },
]

def _distance_phrase(km: float) -> str:
    """Metres below a kilometre; nobody says '0 km away'."""
    if km < 0.02:
        return "right here"
    if km < 1:
        return f"{km * 1000:.0f} m away"
    return f"{km:.1f} km away"


# Intent words are stripped alongside filler. "compare models at Union Square"
# otherwise scores the whole phrase against the gazetteer, diluting the real place
# below the confidence floor and silently falling back to the previous location.
_FILLER = re.compile(
    r"\b(will|there|be|any|bike|bikes|bicycle|bicycles|available|at|near|nearby|"
    r"me|my|i|am|in|the|a|an|is|are|to|from|can|you|tell|find|what|about|around|"
    r"nearest|station|stations|right|now|please|hey|hi|"
    r"compare|comparison|model|models|versus|vs|why|reason|reasons|because|explain|"
    r"how|come|makes|busiest|overview|forecast|predict|prediction|show|many|much|"
    r"get|going|need|want|it|that|this|does|do|did|would|should|could)\b", re.I)
#: openers meaning "keep talking about the place we already established"
_FOLLOWUP = re.compile(
    r"^\s*(why|how come|and|what about|how about|explain|tell me more|more detail|"
    r"because|so|then|ok|okay|thanks|what else|anything else|elaborate)\b", re.I)
_TIME = re.compile(r"\b(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)\b", re.I)
_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2})?)\b")


def _extract_place(message: str) -> str:
    cleaned = _TIME.sub(" ", _ISO.sub(" ", message))
    cleaned = re.sub(r"[?.,!]", " ", cleaned)
    stripped = _FILLER.sub(" ", cleaned)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or cleaned.strip()


def _extract_time(message: str, fallback: str | None) -> str | None:
    iso = _ISO.search(message)
    if iso:
        return iso.group(1).replace(" ", "T")
    clock = _TIME.search(message)
    if clock and fallback:
        hour = int(clock.group(1)) % 12
        if clock.group(3).lower() == "pm":
            hour += 12
        minute = int(clock.group(2) or 0)
        base = datetime.fromisoformat(fallback)
        return base.replace(hour=hour, minute=minute).isoformat()
    return fallback


_MEDIA = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:https?|data):[^)]*\)")


def _sanitise(text: str) -> str:
    """Strip invented media and links from model output.

    An LLM with no image tool will still sometimes emit `![map](https://...)`. The
    frontend escapes it, so it is not an injection risk - it just renders as noise
    and implies a capability that does not exist. Links keep their label and lose
    the URL.
    """
    text = _MEDIA.sub("", text)
    text = _LINK.sub(r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _place_from_calls(calls: list[dict]) -> dict | None:
    """The last place an LLM actually resolved, so the map can follow it."""
    for call in reversed(calls):
        result = call.get("result") or {}
        if call.get("tool") == "find_place" and result.get("ok"):
            return result["best"]
        # station_forecast carries coordinates too, if find_place was skipped
        if call.get("tool") in ("station_forecast", "explain_forecast",
                                "compare_models") and result.get("ok"):
            station = (result.get("station")
                       or (result.get("stations") or [{}])[0].get("station"))
            if station and station.get("lat") is not None:
                return {"name": station["station_name"], "lat": station["lat"],
                        "lng": station["lng"], "source": "station"}
    return None


def _openai_style_tools() -> list[dict]:
    """The same tools in the function-calling shape Mistral expects."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in TOOL_SCHEMAS]


class Assistant:
    """Answers questions by calling the tools.

    Picks a brain from the environment, preferring an explicit `GEOPULSE_LLM`
    ("mistral", "claude", "router"). Keys are read from the environment and never
    stored in the repo.
    """

    def __init__(self, model: str | None = None) -> None:
        preference = (os.environ.get("GEOPULSE_LLM") or "").strip().lower()
        self._client = None
        self.mode = "router"
        self.model = model or ""

        want_mistral = preference in ("", "mistral") and os.environ.get("MISTRAL_API_KEY")
        want_claude = preference in ("", "claude") and os.environ.get("ANTHROPIC_API_KEY")
        if preference == "router":
            return

        if want_mistral and preference != "claude":
            try:
                from mistralai import Mistral

                self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
                self.model = model or os.environ.get("MISTRAL_MODEL",
                                                     "mistral-large-2512")
                self.mode = "mistral"
                return
            except ImportError:
                pass
        if want_claude:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
                self.model = model or os.environ.get("ANTHROPIC_MODEL",
                                                     "claude-sonnet-5")
                self.mode = "claude"
            except ImportError:
                self.mode = "router"

    # ------------------------------------------------------------------ dispatch
    def reply(self, message: str, *, session: dict | None = None) -> dict:
        session = session or {}
        brains = {"claude": self._claude_reply, "mistral": self._mistral_reply}
        brain = brains.get(self.mode)
        if brain is not None:
            try:
                out = brain(message, session)
                out["text"] = _sanitise(out.get("text", ""))
                # the router sets `place` inline; an LLM decides for itself which
                # tools to call, so recover the location it settled on from the
                # find_place result - otherwise the map never follows the answer
                if not out.get("place"):
                    out["place"] = _place_from_calls(out.get("tool_calls", [])) \
                        or session.get("last_place")
                return out
            except Exception as exc:                      # noqa: BLE001
                # a bad key, a rate limit or a network blip should degrade to the
                # deterministic path rather than 500 the whole app
                out = self._router_reply(message, session)
                out["degraded"] = (f"{self.mode} unavailable ({type(exc).__name__}: "
                                   f"{exc}); answered with the built-in router.")
                return out
        return self._router_reply(message, session)

    # ------------------------------------------------------------- mistral brain
    def _mistral_reply(self, message: str, session: dict) -> dict:
        defaults = {k: session.get(k) for k in
                    ("model", "spatial", "resolution", "when", "horizon")
                    if session.get(k) is not None}
        preamble = (f"Current UI selections (use unless the user overrides): "
                    f"{json.dumps(defaults)}") if defaults else ""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += session.get("history") or []
        messages.append({"role": "user",
                         "content": f"{preamble}\n\n{message}".strip()})

        used: list[dict] = []
        for _ in range(6):                                 # bounded tool loop
            response = self._client.chat.complete(
                model=self.model, messages=messages,
                tools=_openai_style_tools(), tool_choice="auto", max_tokens=1200)
            choice = response.choices[0].message
            calls = getattr(choice, "tool_calls", None)
            if not calls:
                text = (choice.content or "").strip()
                return {"text": text, "tool_calls": used, "mode": "mistral",
                        "history": (session.get("history") or [])
                        + [{"role": "user", "content": message},
                           {"role": "assistant", "content": text}]}
            messages.append({"role": "assistant", "content": choice.content or "",
                             "tool_calls": calls})
            for call in calls:
                name = call.function.name
                try:
                    arguments = (json.loads(call.function.arguments)
                                 if isinstance(call.function.arguments, str)
                                 else call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                fn = toolkit.TOOLS.get(name)
                payload = ({"error": f"unknown tool {name}"} if fn is None
                           else fn(**arguments))
                used.append({"tool": name, "input": arguments, "result": payload})
                messages.append({"role": "tool", "name": name,
                                 "tool_call_id": call.id,
                                 "content": json.dumps(payload, default=str)[:20000]})

        return {"text": "I got stuck working that out - try rephrasing?",
                "tool_calls": used, "mode": "mistral"}

    # -------------------------------------------------------------- claude brain
    def _claude_reply(self, message: str, session: dict) -> dict:
        defaults = {k: session.get(k) for k in
                    ("model", "spatial", "resolution", "when", "horizon")
                    if session.get(k) is not None}
        preamble = (f"Current UI selections (use unless the user overrides): "
                    f"{json.dumps(defaults)}") if defaults else ""
        history = session.get("history") or []
        messages = history + [{"role": "user",
                               "content": f"{preamble}\n\n{message}".strip()}]

        used: list[dict] = []
        for _ in range(6):                                 # bounded tool loop
            response = self._client.messages.create(
                model=self.model, max_tokens=1200,
                system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, messages=messages)
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                return {"text": text.strip(), "tool_calls": used,
                        "mode": "claude",
                        "history": messages + [{"role": "assistant",
                                                "content": text}]}
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = toolkit.TOOLS.get(block.name)
                payload = ({"error": f"unknown tool {block.name}"} if fn is None
                           else fn(**block.input))
                used.append({"tool": block.name, "input": block.input,
                             "result": payload})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(payload, default=str)[:20000]})
            messages.append({"role": "user", "content": results})

        return {"text": "I got stuck working that out - try rephrasing?",
                "tool_calls": used, "mode": "claude"}

    # -------------------------------------------------------------- router brain
    def _router_reply(self, message: str, session: dict) -> dict:
        lowered = message.lower()
        spatial = session.get("spatial", "h3")
        resolution = int(session.get("resolution", 8))
        model = session.get("model", "lightgbm_final")
        horizon = int(session.get("horizon", 4))
        when = _extract_time(message, session.get("when"))
        used: list[dict] = []

        def run(name, **kwargs):
            payload = toolkit.TOOLS[name](**kwargs)
            used.append({"tool": name, "input": kwargs, "result": payload})
            return payload

        if any(w in lowered for w in ("busiest", "overview", "hotspot", "citywide",
                                      "whole city", "across nyc")):
            data = run("city_overview", when=when, horizon=horizon, model=model,
                       spatial=spatial, resolution=resolution)
            if not data.get("ok"):
                return {"text": data.get("error", "Could not build the overview."),
                        "tool_calls": used, "mode": "router"}
            top = data["busiest"][:3]
            names = ", ".join(f"{r['region_id'][:8]}… ({r['cumulative_pickups']:.0f})"
                              for r in top)
            return {
                "place": session.get("last_place"),
                "text": (f"Across NYC at {data['time']['local_pretty']}, "
                         f"{data['model_label']} expects "
                         f"{data['total_predicted_pickups']:.0f} pickups in the next "
                         f"{data['window_minutes']} minutes. Busiest cells: {names}."),
                "tool_calls": used, "mode": "router",
            }

        place_query = _extract_place(message)
        remembered = session.get("last_place")
        found, best, hedge = None, None, ""

        # A follow-up opener ("why...", "and what about...") does not veto a lookup -
        # "and what about Prospect Park?" names a real new place. It just raises the
        # bar: to move off the place we were discussing, the match has to be strong,
        # not merely above the usual floor. That keeps the stray word in "why is that
        # the case?" (which matches "Case St & 94 St" at 0.70) from teleporting us.
        is_followup = bool(_FOLLOWUP.match(message.strip())) and remembered is not None
        threshold = 0.75 if is_followup else gazetteer.CONFIDENCE_FLOOR
        if place_query:
            found = run("find_place", query=place_query)
            if found.get("ok") and found["best"]["score"] < threshold:
                found = {**found, "confident": False}

        # A follow-up like "why is that?" has no location in it. Matching whatever
        # stray word survives the filler strip lands the user in another borough
        # ("the case" -> "Case St & 94 St"), so a weak match defers to the place we
        # were already talking about.
        weak = found is None or not found.get("ok") or not found.get("confident")
        if weak and remembered:
            best = remembered
        elif found and found.get("ok"):
            best = found["best"]
            if not found["confident"]:
                hedge = (f" (I matched that loosely to **{best['name']}** — correct "
                         f"me if that's wrong.)")
        else:
            return {"text": (f"I couldn't find “{place_query}”. Our map is built "
                             f"from Citi Bike station names, so try a nearby "
                             f"intersection or park."),
                    "tool_calls": used, "mode": "router"}

        if best is None:
            return {"text": "Tell me where you are - a corner or a landmark, like "
                            "'Times Square' or 'W 21 St & 6 Ave'.",
                    "tool_calls": used, "mode": "router"}

        wants_why = any(w in lowered for w in ("why", "reason", "because", "explain",
                                               "how come", "what makes"))
        wants_compare = any(w in lowered for w in ("compare", "models", "which model",
                                                   "versus", " vs "))
        wants_route = any(w in lowered for w in ("take me", "route", "directions",
                                                 "walk", "how do i get", "navigate",
                                                 "get there", "go to", "nearest bike"))

        if wants_route:
            data = run("route_to_bike", lat=best["lat"], lng=best["lng"], when=when,
                       horizon=horizon, model=model, spatial=spatial,
                       resolution=resolution)
            if not data.get("ok"):
                return {"text": data.get("error", "Could not plan a route."),
                        "tool_calls": used, "mode": "router", "place": best}
            station = data["target"]["station"]
            if not data.get("walkable", True):
                return {"text": (f"The closest dock to **{best['name']}** is "
                                 f"**{station['station_name']}**, "
                                 f"{data['walk_metres'] / 1000:.1f} km away — too far "
                                 f"to walk. Citi Bike doesn't cover this part of the "
                                 f"city."),
                        "tool_calls": used, "mode": "router", "place": best}
            smarter = ("" if data["chose_nearest"] else
                       " That isn't the closest dock — the nearer one is draining, so "
                       "this is the better bet.")
            hops = (f" The path runs through {data['graph_hops']} intermediate "
                    f"dock{'s' if data['graph_hops'] > 1 else ''}."
                    if data["graph_hops"] else "")
            return {"text": (f"Head to **{station['station_name']}** — "
                             f"**{data['walk_metres']} m**, about "
                             f"**{data['walk_minutes']:.0f} min** on foot.{smarter}"
                             f"{hops} It's currently "
                             f"{data['target']['availability']['label'].lower()}."),
                    "tool_calls": used, "mode": "router", "place": best}

        if wants_compare:
            data = run("compare_models", lat=best["lat"], lng=best["lng"], when=when,
                       horizon=horizon, spatial=spatial, resolution=resolution)
            if not data.get("ok"):
                return {"text": data.get("error", "Comparison failed."),
                        "tool_calls": used, "mode": "router"}
            lines = "; ".join(f"{m['label']} {m['region_pickups']:.0f}"
                              for m in data["models"])
            actual = data.get("actual_region_pickups")
            tail = (f" Actual: **{actual:.0f}**."
                    if actual is not None else "")
            return {"text": (f"Around **{best['name']}**{hedge}, predicted pickups in "
                             f"the cell for the {data['bin_minutes']}-minute bin "
                             f"{horizon * data['bin_minutes']} min ahead — "
                             f"{lines}.{tail}"),
                    "tool_calls": used, "mode": "router",
                    "place": best}

        data = run("station_forecast", lat=best["lat"], lng=best["lng"], when=when,
                   horizon=horizon, model=model, spatial=spatial,
                   resolution=resolution)
        if not data.get("ok"):
            return {"text": data.get("error", "Forecast failed."),
                    "tool_calls": used, "mode": "router"}
        if not data["stations"]:
            return {"text": f"No Citi Bike stations near {best['name']}.",
                    "tool_calls": used, "mode": "router"}

        first = data["stations"][0]
        station = first["station"]
        verdict = first["availability"]
        text = (f"Nearest dock to **{best['name']}**{hedge} is "
                f"**{station['station_name']}**, "
                f"{_distance_phrase(station['distance_km'])}. "
                f"Over the next {data['window_minutes']} minutes "
                f"({data['time']['local_pretty']}) {data['model_label']} expects "
                f"**~{first['predicted_pickups']:.0f} pickups** and "
                f"~{first['predicted_dropoffs']:.0f} returns there — "
                f"**{verdict['label'].lower()}**. "
                f"{verdict['detail'][:1].upper()}{verdict['detail'][1:]}.")
        if data["time"]["mode"] == "typical_conditions":
            text += (" That's typical conditions for this weekday and time — the "
                     "models cover 2023-24, so nothing here is live.")

        if wants_why:
            why = run("explain_forecast", lat=best["lat"], lng=best["lng"], when=when,
                      horizon=horizon, model=model, spatial=spatial,
                      resolution=resolution)
            if why.get("available"):
                # .capitalize() would lowercase the rest, turning "Wednesday" into
                # "wednesday"; only the first character should change
                bullets = " ".join(f"{d['detail'][:1].upper()}{d['detail'][1:]}."
                                   for d in why["drivers"][:3])
                text += f"\n\n**Why:** {bullets}"
                events = why.get("events_nearby") or []
                if events:
                    text += (f"\n\nNearby: {events[0]['name']} at "
                             f"{events[0]['venue']} ({events[0]['distance_km']} km). "
                             f"Context only — our ablation found events don't move "
                             f"demand measurably.")
        return {"text": text, "tool_calls": used, "mode": "router", "place": best}
