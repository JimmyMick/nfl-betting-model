"""An LLM as an independent pick'em "expert".

Generates a winner + confidence (50-100) for each game in a week, reasoning
**blind** — it never sees the model's probability or the Vegas line. It is fed
only factual, current data (records, recent results, key injuries, weather,
home field) plus any free-text notes the user adds per game (rivalry, off-field
distractions, etc.), and is explicitly encouraged to back underdogs when the
facts warrant rather than just picking the better record.

Its picks are written to the same ``predictions/picks/{season}-wk{NN}.csv`` as
the human experts (via ``submit.merge_player_picks``), so grading, calibration,
and the leaderboard treat it like any other player — no new scoring code.

Provider-pluggable (Claude default, OpenAI-switchable) over plain HTTP with
httpx, so no extra SDK dependencies. Keys come from the environment or a repo
``.env`` (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
import pandas as pd

from . import submit as submit_mod

_REPO = Path(__file__).resolve().parent.parent

# Home-stadium city per team, for best-effort weather lookups (outdoor games).
_TEAM_CITY = {
    "ARI": "Glendale AZ", "ATL": "Atlanta", "BAL": "Baltimore",
    "BUF": "Orchard Park NY", "CAR": "Charlotte", "CHI": "Chicago",
    "CIN": "Cincinnati", "CLE": "Cleveland", "DAL": "Arlington TX",
    "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay", "HOU": "Houston",
    "IND": "Indianapolis", "JAX": "Jacksonville", "KC": "Kansas City MO",
    "LV": "Las Vegas", "LAC": "Inglewood CA", "LAR": "Inglewood CA",
    "MIA": "Miami Gardens FL", "MIN": "Minneapolis", "NE": "Foxborough MA",
    "NO": "New Orleans", "NYG": "East Rutherford NJ", "NYJ": "East Rutherford NJ",
    "PHI": "Philadelphia", "PIT": "Pittsburgh", "SF": "Santa Clara CA",
    "SEA": "Seattle", "TB": "Tampa", "TEN": "Nashville", "WAS": "Landover MD",
}


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines) — no python-dotenv dependency."""
    env = _REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── Context assembly (the "raw data" the expert reasons over) ─────────────────

def team_records(games: pd.DataFrame, season: int, upto_week: int) -> dict[str, str]:
    """W-L record per team from completed games this season before ``upto_week``."""
    done = games[(games["season"] == season) & (games["week"] < upto_week)
                 & games["home_win"].notna()]
    rec: dict[str, list[int]] = {}
    for _, g in done.iterrows():
        hw = g["home_win"] == 1
        for team, won in ((g["home_team"], hw), (g["away_team"], not hw)):
            r = rec.setdefault(team, [0, 0])
            r[0 if won else 1] += 1
    return {t: f"{w}-{l}" for t, (w, l) in rec.items()}


def recent_results(games: pd.DataFrame, season: int, team: str, upto_week: int,
                   n: int = 4) -> str:
    """Compact 'W 24-17 vs X' style string of a team's last ``n`` games."""
    done = games[(games["season"] == season) & (games["week"] < upto_week)
                 & games["home_win"].notna()
                 & ((games["home_team"] == team) | (games["away_team"] == team))]
    done = done.sort_values("week").tail(n)
    out = []
    for _, g in done.iterrows():
        home = g["home_team"] == team
        opp = g["away_team"] if home else g["home_team"]
        pf = g["home_score"] if home else g["away_score"]
        pa = g["away_score"] if home else g["home_score"]
        res = "W" if pf > pa else "L"
        loc = "vs" if home else "@"
        out.append(f"{res} {int(pf)}-{int(pa)} {loc} {opp}")
    return "; ".join(out) if out else "no games yet"


def team_injuries(inj: pd.DataFrame, season: int, week: int, team: str,
                  limit: int = 8) -> str:
    """Key injury-report entries for a team-week (Out/Doubtful first)."""
    if inj.empty:
        return "none reported"
    t = inj[(inj["season"] == season) & (inj["week"] == week)
            & (inj["team"] == team) & inj["report_status"].notna()].copy()
    if t.empty:
        return "none reported"
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    t["_o"] = t["report_status"].map(order).fillna(3)
    t = t.sort_values("_o").head(limit)
    parts = []
    for _, r in t.iterrows():
        inj_txt = r.get("report_primary_injury")
        inj_txt = f", {inj_txt}" if isinstance(inj_txt, str) and inj_txt else ""
        parts.append(f"{r['position']} {r['full_name']} ({r['report_status']}{inj_txt})")
    return "; ".join(parts)


def game_weather(home_team: str, roof, gameday) -> str:
    """Best-effort weather string; 'indoor' for domes, else a wttr.in forecast."""
    roof_s = str(roof).lower() if roof is not None else ""
    if roof_s in ("dome", "closed"):
        return "indoor (climate-controlled)"
    city = _TEAM_CITY.get(home_team)
    if not city:
        return "unknown"
    try:
        r = httpx.get(f"https://wttr.in/{city}", params={"format": "j1"}, timeout=8)
        r.raise_for_status()
        data = r.json()
        day = str(pd.to_datetime(gameday).date()) if gameday is not None else None
        for d in data.get("weather", []):
            if d.get("date") == day:
                noon = d["hourly"][len(d["hourly"]) // 2]
                return (f"{noon['tempF']}°F, wind {noon['windspeedMiles']} mph, "
                        f"{noon['weatherDesc'][0]['value']}")
        cur = data["current_condition"][0]
        return (f"{cur['temp_F']}°F, wind {cur['windspeedMiles']} mph, "
                f"{cur['weatherDesc'][0]['value']} (current)")
    except Exception:
        return "unavailable"


def assemble_context(games: pd.DataFrame, inj: pd.DataFrame, season: int,
                     week: int, with_weather: bool = True) -> list[dict]:
    """One context dict per game in (season, week)."""
    slate = games[(games["season"] == season) & (games["week"] == week)]
    records = team_records(games, season, week)
    ctx = []
    for _, g in slate.sort_values(["gameday", "game_id"]).iterrows():
        home, away = g["home_team"], g["away_team"]
        ctx.append({
            "game_id": submit_mod.game_id(season, week, away, home),
            "away": away, "home": home,
            "away_record": records.get(away, "0-0"),
            "home_record": records.get(home, "0-0"),
            "away_form": recent_results(games, season, away, week),
            "home_form": recent_results(games, season, home, week),
            "away_injuries": team_injuries(inj, season, week, away),
            "home_injuries": team_injuries(inj, season, week, home),
            "weather": game_weather(home, g.get("roof"), g.get("gameday"))
            if with_weather else "n/a",
            "div_game": bool(g.get("div_game", 0)),
        })
    return ctx


# ── Prompt + LLM call ─────────────────────────────────────────────────────────

SYSTEM = (
    "You are a sharp, opinionated NFL analyst making straight-up WINNER picks for "
    "an expert pick'em pool. You are scored on accuracy AND calibration (Brier / "
    "log loss), so set confidence honestly: 50 means a true coin flip, 100 means "
    "near-certain. You do NOT see betting lines or any statistical model's "
    "prediction — judge each game yourself from the factual data provided "
    "(records, recent results, injuries, weather, home field) plus any extra "
    "notes. Be willing to back an underdog when injuries, weather, matchups, a "
    "rivalry, or off-field distractions warrant it — do NOT simply pick the team "
    "with the better record every week; a well-reasoned contrarian call is "
    "encouraged. Reply with ONLY a JSON object, no prose."
)


def build_user_prompt(ctx: list[dict], notes: dict[str, str], season: int,
                      week: int) -> str:
    lines = [f"NFL {season}, Week {week}. Pick a winner and confidence (50-100) "
             "for each game.\n"]
    for c in ctx:
        lines.append(f"### {c['away']} ({c['away_record']}) @ {c['home']} "
                     f"({c['home_record']}){'  [division game]' if c['div_game'] else ''}")
        lines.append(f"- {c['away']} recent: {c['away_form']}")
        lines.append(f"- {c['home']} recent: {c['home_form']}")
        lines.append(f"- {c['away']} injuries: {c['away_injuries']}")
        lines.append(f"- {c['home']} injuries: {c['home_injuries']}")
        lines.append(f"- Weather ({c['home']} home): {c['weather']}")
        note = notes.get(c["game_id"], "").strip()
        if note:
            lines.append(f"- EXTRA INTEL: {note}")
        lines.append("")
    lines.append(
        'Return JSON: {"picks": [{"game": "AWAY@HOME", "winner": "TEAM_ABBR", '
        '"confidence": <50-100 int>, "rationale": "one sentence"}]}. '
        "Use the exact team abbreviations shown.")
    return "\n".join(lines)


def call_llm(system: str, user: str, provider: str, model: str, api_key: str,
             temperature: float = 0.5) -> str:
    if provider == "anthropic":
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": 4000, "temperature": temperature,
                  "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=120)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    elif provider == "openai":
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "content-type": "application/json"},
            json={"model": model, "temperature": temperature,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise ValueError(f"Unknown provider: {provider}")


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def parse_picks(text: str, ctx: list[dict]) -> dict[str, dict]:
    """Map the LLM's JSON back to ``game_id -> {pick, confidence, rationale}``."""
    data = _extract_json(text)
    by_pair = {(c["away"], c["home"]): c for c in ctx}
    valid_teams = {c["away"] for c in ctx} | {c["home"] for c in ctx}
    out: dict[str, dict] = {}
    for p in data.get("picks", []):
        game = str(p.get("game", "")).replace(" ", "")
        away, _, home = game.partition("@")
        c = by_pair.get((away, home))
        if c is None:
            continue
        winner = str(p.get("winner", "")).strip()
        if winner not in (c["away"], c["home"]) or winner not in valid_teams:
            continue
        try:
            conf = int(round(float(p.get("confidence", 50))))
        except (TypeError, ValueError):
            conf = 50
        conf = max(50, min(100, conf))
        out[c["game_id"]] = {"pick": winner, "confidence": conf,
                             "rationale": str(p.get("rationale", "")).strip()}
    return out


# ── Provider resolution + orchestration ───────────────────────────────────────

def resolve_provider(provider: str | None = None) -> tuple[str, str, str, str]:
    """Return (provider, model, api_key, player_name) from args + env/.env."""
    _load_dotenv()
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if provider == "anthropic":
        model = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        default_name = "Claude"
    elif provider == "openai":
        model = os.environ.get("LLM_MODEL", "gpt-4o")
        key = os.environ.get("OPENAI_API_KEY", "")
        default_name = "GPT"
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    name = os.environ.get("AI_PLAYER_NAME", default_name)
    return provider, model, key, name


def generate_picks(games: pd.DataFrame, inj: pd.DataFrame, season: int, week: int,
                   notes: dict[str, str] | None = None,
                   provider: str | None = None, temperature: float = 0.5,
                   with_weather: bool = True) -> tuple[dict[str, dict], str]:
    """Assemble context, call the LLM, return ``(picks_by_game_id, player_name)``."""
    prov, model, key, name = resolve_provider(provider)
    if not key:
        raise RuntimeError(
            f"No API key for provider '{prov}'. Set "
            f"{'ANTHROPIC_API_KEY' if prov == 'anthropic' else 'OPENAI_API_KEY'} "
            "in the environment or repo .env.")
    ctx = assemble_context(games, inj, season, week, with_weather=with_weather)
    if not ctx:
        raise RuntimeError(f"No games found for {season} week {week}.")
    user = build_user_prompt(ctx, notes or {}, season, week)
    text = call_llm(SYSTEM, user, prov, model, key, temperature=temperature)
    picks = parse_picks(text, ctx)
    return picks, name
