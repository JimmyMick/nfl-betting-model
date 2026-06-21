"""Tuesday grading: score the model's picks and track the season vs the market.

After a week's games are final, this grades the model's straight-up picks (✓/✗)
and maintains a season-to-date tracker comparing the model to Vegas on both
accuracy and calibration (log loss / Brier). Same validated setup as the weekly
preview: the model is trained once on every season before the target, then used
to score every completed game in the target season.

This is a *scorekeeping* companion to predict.py — it answers "how is the model
doing?", not "who should I bet?".

Examples
--------
    uv run grade.py --season 2024 --week 10
    uv run grade.py --season 2024 --week 10 --out predictions/2024-grade-wk10.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from predict import _prepare_frame, _train_for
from nfl_betting_model import picks as picks_mod
from nfl_betting_model.features import market_home_prob


def grade_season(season: int, through_week: int | None = None,
                 train_start: int = 2010, kind: str = "logistic") -> pd.DataFrame:
    """Score every completed game in ``season`` (optionally up to ``through_week``).

    Returns one row per played game with the model's and market's home-win
    probabilities, each side's straight-up pick, the actual winner, and whether
    each pick was correct.
    """
    df, cols = _prepare_frame(season, train_start)
    pipe = _train_for(df, cols, season, kind)

    s = df[(df["season"] == season) & df["home_win"].notna()].copy()
    if through_week is not None:
        s = s[s["week"] <= through_week]
    if s.empty:
        raise SystemExit(f"No completed games found for {season} "
                         f"through week {through_week}.")

    s["model_home_prob"] = pipe.predict_proba(s[cols])[:, 1]
    s["market_home_prob"] = market_home_prob(s).to_numpy()

    s["winner"] = np.where(s["home_win"] == 1, s["home_team"], s["away_team"])
    s["model_pick"] = np.where(s["model_home_prob"] >= 0.5, s["home_team"], s["away_team"])
    s["market_pick"] = np.where(s["market_home_prob"] >= 0.5, s["home_team"], s["away_team"])
    s["model_correct"] = (s["model_pick"] == s["winner"]).astype(int)
    s["market_correct"] = (s["market_pick"] == s["winner"]).astype(int)
    return s.sort_values(["week", "game_id"]).reset_index(drop=True)


def _record(correct: pd.Series) -> str:
    """e.g. '10-6 (63%)' from a 0/1 correctness series."""
    w = int(correct.sum())
    n = len(correct)
    return f"{w}-{n - w} ({w / n:.0%})" if n else "0-0 (—)"


def _calibration(s: pd.DataFrame) -> tuple[float, float, float, float]:
    """(model log loss, model Brier, market log loss, market Brier).

    Market metrics use only games with a usable line; NaN if none.
    """
    y = s["home_win"].to_numpy()
    pm = s["model_home_prob"].to_numpy()
    m_ll = log_loss(y, pm, labels=[0, 1])
    m_br = brier_score_loss(y, pm)
    mkt = s["market_home_prob"].to_numpy()
    mask = ~np.isnan(mkt)
    if mask.sum():
        k_ll = log_loss(y[mask], mkt[mask], labels=[0, 1])
        k_br = brier_score_loss(y[mask], mkt[mask])
    else:
        k_ll = k_br = float("nan")
    return m_ll, m_br, k_ll, k_br


def weekly_summary(s: pd.DataFrame) -> pd.DataFrame:
    """Per-week records plus a cumulative season-to-date row per metric."""
    rows = []
    for wk, g in s.groupby("week"):
        rows.append({
            "Week": str(int(wk)),
            "Games": len(g),
            "Model": _record(g["model_correct"]),
            "Market": _record(g["market_correct"]),
        })
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "Week": "Season", "Games": len(s),
        "Model": _record(s["model_correct"]),
        "Market": _record(s["market_correct"]),
    }
    return out


def _md_table(df: pd.DataFrame, bold_first_col: bool = False) -> list[str]:
    """Render a DataFrame as GitHub-markdown table lines."""
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return lines


def render_picks(scored: pd.DataFrame, s: pd.DataFrame,
                 through_week: int) -> list[str]:
    """Pick'em leaderboard section: season standings + this week's head-to-head."""
    if scored is None or scored.empty:
        return []
    lines = ["", "## Pick'em leaderboard", ""]
    board = picks_mod.leaderboard(scored, s)
    lines.append("_Each player vs the model **on the games they picked**. "
                 "Brier / log loss use picks that carried a confidence._")
    lines.append("")
    lines += _md_table(board)

    this_week = scored[scored["week"] == through_week]
    if not this_week.empty:
        lines += ["", f"### Week {through_week} picks", ""]
        wk = (this_week.assign(ok=this_week["correct"])
              .groupby("player")
              .agg(Picks=("correct", "size"), Hits=("correct", "sum"))
              .reset_index())
        wk["Record"] = wk["Hits"].astype(str) + "-" + (wk["Picks"] - wk["Hits"]).astype(str)
        wk = wk[["player", "Record", "Picks"]].rename(columns={"player": "Player"})
        lines += _md_table(wk)
    return lines


def render(s: pd.DataFrame, season: int, through_week: int,
           scored: pd.DataFrame | None = None) -> str:
    m_acc = s["model_correct"].mean()
    k_acc = s["market_correct"].mean()
    m_ll, m_br, k_ll, k_br = _calibration(s)
    lines = [f"# NFL model — {season} season-to-date (through Week {through_week})", ""]
    lines.append(
        f"**Model {_record(s['model_correct'])} straight-up · "
        f"Market {_record(s['market_correct'])} · "
        f"{m_acc - k_acc:+.0%} vs market**"
    )
    lines.append(
        f"_Calibration — model log loss {m_ll:.3f} / Brier {m_br:.3f}; "
        f"market {k_ll:.3f} / {k_br:.3f}._"
    )
    lines.append("")

    # This week's game-by-game grades.
    last = s[s["week"] == through_week]
    if not last.empty:
        lines.append(f"## Week {through_week} results")
        lines.append("")
        lines.append("| Matchup | Model pick | Result |")
        lines.append("|---|---|---|")
        for _, r in last.iterrows():
            prob = max(r["model_home_prob"], 1 - r["model_home_prob"])
            mark = "✓" if r["model_correct"] else "✗"
            lines.append(
                f"| {r['away_team']} @ {r['home_team']} "
                f"| {r['model_pick']} {prob:.0%} | {r['winner']} {mark} |"
            )
        lines.append("")

    # Season-to-date tracker.
    lines.append("## Week-by-week")
    lines.append("")
    summary = weekly_summary(s)
    lines.append("| " + " | ".join(summary.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(summary.columns)) + "|")
    for _, r in summary.iterrows():
        cells = [f"**{c}**" if r["Week"] == "Season" else str(c) for c in r]
        lines.append("| " + " | ".join(cells) + " |")

    lines += render_picks(scored, s, through_week)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade model picks vs the market")
    ap.add_argument("--season", type=int, help="omit with --auto")
    ap.add_argument("--week", type=int, help="grade through this completed week; "
                    "omit with --auto")
    ap.add_argument("--auto", action="store_true",
                    help="detect the most recent completed week from the live "
                         "schedule (for the scheduled Tuesday grade)")
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--model", choices=["logistic", "gbm"], default="logistic")
    ap.add_argument("--out", default=None, help="also write the markdown to this path")
    ap.add_argument("--export-dir", nargs="?", const="", default=None,
                    help="also export graded games + scored picks as CSV here "
                         "for the cloud dashboard (bare flag uses "
                         "predictions/cloud)")
    args = ap.parse_args()

    if args.auto:
        from nfl_betting_model.weeks import detect_target
        season, week = detect_target("grade", args.season)
    elif args.season is not None and args.week is not None:
        season, week = args.season, args.week
    else:
        ap.error("provide --season and --week, or --auto")

    s = grade_season(season, week, args.train_start, args.model)

    all_picks = picks_mod.load_all_picks(season, week)
    scored = picks_mod.score(all_picks, s) if not all_picks.empty else None

    report = render(s, season, week, scored)
    print("\n" + report)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n")
        print(f"\nWrote {path}")

    if args.export_dir is not None:
        from nfl_betting_model import cloud
        out_dir = Path(args.export_dir) if args.export_dir else cloud.ARTIFACT_DIR
        cloud.write_grade_artifacts(s, scored, season, week, out_dir)
        print(f"Exported cloud artifacts to {out_dir}")


if __name__ == "__main__":
    main()
