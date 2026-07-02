"""Ask natural-language questions answered from *our own model's* output.

This is the "brain + data" glue: a real LLM (Claude default, OpenAI-switchable)
handles the conversation, but it is grounded **only** in the artifacts the
weekly pipeline already exports to ``predictions/cloud/`` — the current
model-vs-market preview slate, the season-to-date grade, and the pick'em
leaderboard. The LLM never invents scores or matchups; if the answer isn't in
the supplied data it is told to say so.

Deliberately light: it reads the committed CSVs (pandas) and calls the provider
over httpx. No training, no nflreadpy fetch, no extra SDK — so it runs anywhere
the cloud artifacts are checked out, and can back a chat box in the local app.

The LLM plumbing (``_load_dotenv`` / ``resolve_provider`` / ``call_llm``) is
shared verbatim with ``llm_picker`` so there is one place to change providers.

Examples
--------
    uv run chat.py "who does the model like most this week and why?"
    uv run chat.py            # interactive REPL
    LLM_PROVIDER=openai uv run chat.py "how good has the model been vs Vegas?"
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import cloud
from .llm_picker import call_llm, resolve_provider

# ── Grounding: turn the exported artifacts into a compact text digest ─────────

def _pick_side(row: pd.Series) -> tuple[str, float]:
    """Team the model favors in a preview row + its win prob for that team."""
    p = float(row["model_home_prob"])
    return (row["home_team"], p) if p >= 0.5 else (row["away_team"], 1.0 - p)


def format_preview(preview: pd.DataFrame, meta: dict) -> str:
    """The current week's model-vs-market slate, sorted by disagreement."""
    if preview is None or preview.empty:
        return "No weekly preview is available yet."
    season = meta.get("preview_season", "?")
    week = meta.get("preview_week", "?")
    df = preview.copy()
    df["_absedge"] = df["edge"].abs()
    df = df.sort_values("_absedge", ascending=False)
    lines = [f"CURRENT PREVIEW — {season} Week {week} (win probabilities shown "
             "for the HOME team; edge is model minus market on the home team, so "
             "positive = model higher on home; sorted by disagreement):"]
    for _, r in df.iterrows():
        home, away = r["home_team"], r["away_team"]
        model_home = float(r["model_home_prob"])
        market_home = float(r["market_home_prob"])
        edge = float(r["edge"])
        pick, _ = _pick_side(r)
        mkt_pick = home if market_home >= 0.5 else away
        agree = "agree" if pick == mkt_pick else "DISAGREE"
        lines.append(
            f"- {away} @ {home}: model {home} {model_home:.0%} / market {home} "
            f"{market_home:.0%} (edge {edge:+.0%} on {home}); "
            f"model picks {pick}, market picks {mkt_pick} ({agree}); "
            f"key driver: {r['driver']}")
    return "\n".join(lines)


def format_grade(graded: pd.DataFrame, meta: dict) -> str:
    """Season-to-date scoreboard: how the model has done vs the Vegas market."""
    if graded is None or graded.empty:
        return "No graded results are available yet."
    season = meta.get("grade_season", "?")
    through = meta.get("grade_through_week", "?")
    n = len(graded)
    m_correct = int(graded["model_correct"].sum())
    k_correct = int(graded["market_correct"].sum()) if "market_correct" in graded else None
    lines = [f"SEASON GRADE — {season} through Week {through} ({n} games):",
             f"- Model straight-up: {m_correct}-{n - m_correct} "
             f"({m_correct / n:.1%})"]
    if k_correct is not None:
        lines.append(f"- Market (Vegas favorite) straight-up: {k_correct}-"
                     f"{n - k_correct} ({k_correct / n:.1%})")
        diff = m_correct - k_correct
        verdict = ("model ahead" if diff > 0 else
                   "market ahead" if diff < 0 else "dead even")
        lines.append(f"- Model vs market: {diff:+d} games ({verdict})")
    # Brier score (calibration) on the model's home-win probability.
    if "model_home_prob" in graded and "home_win" in graded:
        g = graded.dropna(subset=["model_home_prob", "home_win"])
        if not g.empty:
            brier = ((g["model_home_prob"] - g["home_win"]) ** 2).mean()
            lines.append(f"- Model Brier score: {brier:.3f} "
                         "(lower is better; 0.25 = coin flip)")
    return "\n".join(lines)


def format_leaderboard(scored: pd.DataFrame, graded: pd.DataFrame) -> str:
    """Pick'em standings (players + AI experts), if any picks exist yet."""
    if scored is None or scored.empty:
        return "No pick'em picks have been submitted yet."
    try:
        from .picks import leaderboard
        lb = leaderboard(scored, graded)
    except Exception:
        return "Pick'em picks exist but the leaderboard could not be computed."
    if lb is None or lb.empty:
        return "No pick'em picks have been submitted yet."
    lines = ["PICK'EM LEADERBOARD (players + AI experts, ranked; 'vs Model' is "
             "player accuracy minus the model's on the same games):"]
    for _, r in lb.iterrows():
        lines.append(
            f"- {r['Player']}: {r['Record']} on {r['Picks']} picks; "
            f"vs Model {r['vs Model']}; Brier {r['Brier']}; Log loss {r['Log loss']}")
    return "\n".join(lines)


def build_context(art_dir=cloud.ARTIFACT_DIR, preview=None,
                  preview_meta=None) -> tuple[str, dict]:
    """Assemble the full grounding digest.

    By default it reads the committed ``predictions/cloud/`` artifacts. Callers
    with a *live* preview in hand (e.g. the dashboard, which just trained the
    model on the current slate) can pass ``preview`` (a DataFrame with the
    standard preview columns) plus ``preview_meta`` (``preview_season`` /
    ``preview_week``) to ground the current-slate block on that instead of the
    last export — so the chat answers about the week the user is actually
    looking at. The season grade / pick'em blocks still come from disk.
    """
    arts = cloud.load_artifacts(art_dir)
    meta = dict(arts.get("meta", {}))
    preview_frame = arts.get("preview")
    if preview is not None:
        preview_frame = preview
        meta.update(preview_meta or {})
    blocks = [
        format_preview(preview_frame, meta),
        format_grade(arts.get("graded"), meta),
        format_leaderboard(arts.get("scored"), arts.get("graded")),
    ]
    return "\n\n".join(blocks), meta


# ── Prompt + answer ───────────────────────────────────────────────────────────

SYSTEM = (
    "You are a sharp, concise NFL analyst embedded in a personal forecasting "
    "app. You answer questions using ONLY the DATA CONTEXT provided below, which "
    "comes from the user's own win-probability model and its season track "
    "record. Ground every claim in that data: cite the model's win probability, "
    "the market-implied probability, the edge, or the key driver when relevant. "
    "The 'edge' is the model's home-team win probability minus the market's, so "
    "a positive edge means the model is higher on the home team than Vegas is "
    "(and a large edge either way marks the model's biggest disagreements). If a "
    "question asks about "
    "something not in the context (a specific player's yards, a game not on the "
    "slate, next week, etc.), say plainly that it's not in the current data "
    "rather than guessing. Be opinionated and readable; no betting advice, just "
    "what the model thinks and why."
)


def answer_question(question: str, provider: str | None = None,
                    temperature: float = 0.3,
                    art_dir=cloud.ARTIFACT_DIR, preview=None,
                    preview_meta=None) -> str:
    """Answer one question grounded in the model artifacts.

    Pass ``preview`` / ``preview_meta`` to ground the current-slate block on a
    live preview frame instead of the last committed export (see
    ``build_context``).
    """
    prov, model, key, _ = resolve_provider(provider)
    if not key:
        raise RuntimeError(
            f"No API key for provider '{prov}'. Set "
            f"{'ANTHROPIC_API_KEY' if prov == 'anthropic' else 'OPENAI_API_KEY'} "
            "in the environment or repo .env.")
    context, meta = build_context(art_dir, preview=preview,
                                  preview_meta=preview_meta)
    if not meta:
        raise RuntimeError(
            f"No model artifacts found in {art_dir}. Run predict.py / grade.py "
            "with --export-dir first (the weekly crons do this automatically).")
    user = f"DATA CONTEXT:\n{context}\n\nQUESTION: {question}"
    return call_llm(SYSTEM, user, prov, model, key, temperature=temperature).strip()


def _repl(provider: str | None, temperature: float, art_dir) -> None:
    context, meta = build_context(art_dir)
    prov, model, _, _ = resolve_provider(provider)
    wk = f"{meta.get('preview_season', '?')} Wk {meta.get('preview_week', '?')}"
    print(f"NFL model chat — grounded in {wk} artifacts, via {prov}/{model}.")
    print("Ask a question (Ctrl-D or 'quit' to exit).\n")
    while True:
        try:
            q = input("you > ").strip()
        except EOFError:
            print()
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        try:
            print(f"\n{answer_question(q, provider, temperature, art_dir)}\n")
        except Exception as e:  # keep the REPL alive on a transient API error
            print(f"[error] {e}\n", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ask natural-language questions about the model's current view")
    ap.add_argument("question", nargs="*",
                    help="one-shot question; omit for an interactive REPL")
    ap.add_argument("--provider", choices=["anthropic", "openai"],
                    help="override LLM_PROVIDER (default: anthropic)")
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    if args.question:
        try:
            print(answer_question(" ".join(args.question), args.provider,
                                  args.temperature))
        except (RuntimeError, ValueError) as e:
            raise SystemExit(str(e))
    else:
        _repl(args.provider, args.temperature, cloud.ARTIFACT_DIR)
