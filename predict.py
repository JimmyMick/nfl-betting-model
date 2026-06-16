"""Weekly inference: calibrated win probabilities for a slate of games.

Trains the validated full-feature GBM (isotonic-calibrated) on every season
before the target, then predicts the requested season+week using strictly
pre-game features (recent form / EPA / Elo / Madden QB+starter OVR computed
from games already played). Emits a model-vs-market table sorted by the size
of the disagreement -- the genuinely interesting part.

This is a *probability/preview* tool, not a betting tip sheet: it reports what
the model thinks and where it differs from Vegas. No picks, no EV claims.

Examples
--------
    uv run predict.py --season 2024 --week 10
    uv run predict.py --season 2026 --week 1 --out predictions/2026-wk01.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_betting_model import (
    data, epa as epa_mod, model, qb as qb_mod, starters as starters_mod,
)
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features, market_home_prob

# Interpretable diffs (home minus away) used to name a "key driver" per game.
# Each is signed so positive favours the home team.
DRIVER_FEATURES = {
    "qb_ovr_diff": "QB rating",
    "net_epa_diff": "net EPA/play",
    "elo_diff": "Elo",
    "starter_ovr_diff": "roster talent",
    "form_margin_diff": "recent margin",
}


def _carry_forward(table: pd.DataFrame, value_cols: list[str],
                   games: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Fill missing (target game, team) rows by carrying each team's most recent
    prior-game values forward.

    For a not-yet-played game there is no play-by-play, so qb/starter OVR has no
    row. The OVR is a season-fixed launch rating, so the best pre-game estimate
    is the team's latest known starter(s). This is a no-op for games already in
    the table, so historical weeks stay exactly as validated.
    """
    meta = games[["game_id", "season", "week"]]
    hist = table.merge(meta, on="game_id", how="left").sort_values(["season", "week"])
    present = set(zip(table["game_id"], table["team"]))
    rows = []
    for _, g in target.iterrows():
        for side in ("home_team", "away_team"):
            team = g[side]
            if (g["game_id"], team) in present:
                continue
            h = hist[(hist["team"] == team) & (
                (hist["season"] < g["season"])
                | ((hist["season"] == g["season"]) & (hist["week"] < g["week"]))
            )].dropna(subset=value_cols, how="all")
            if h.empty:
                continue
            last = h.iloc[-1]
            rows.append({"game_id": g["game_id"], "team": team,
                         **{c: last[c] for c in value_cols}})
    if rows:
        table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    return table


def _prob_str(home_team: str, away_team: str, home_prob: float) -> str:
    """Favoured team + its win probability, e.g. 'BUF 58%'."""
    if home_prob >= 0.5:
        return f"{home_team} {home_prob:.0%}"
    return f"{away_team} {1 - home_prob:.0%}"


def _drivers(target: pd.DataFrame) -> list[str]:
    """For each game, name the strongest factor *supporting the model's pick* —
    the largest z-scored diff whose sign agrees with the model's leaned side.
    Z-scores put the different diffs on a comparable scale across the slate."""
    cols = [c for c in DRIVER_FEATURES if c in target.columns]
    z = pd.DataFrame(index=target.index)
    for c in cols:
        s = pd.to_numeric(target[c], errors="coerce")
        sd = s.std(ddof=0)
        z[c] = (s - s.mean()) / sd if sd and not np.isnan(sd) else 0.0
    out = []
    for _, row in target.iterrows():
        # +1 if the model leans home, -1 if away. A feature supports the pick
        # when its signed diff shares that sign.
        lean = 1.0 if row["model_home_prob"] >= 0.5 else -1.0
        best_feat, best_mag = None, 0.0
        for c in cols:
            val = target.loc[row.name, c]
            if pd.isna(val) or np.sign(val) != lean:
                continue
            mag = abs(z.loc[row.name, c])
            if mag > best_mag:
                best_feat, best_mag = c, mag
        if best_feat is None:
            out.append("model interactions")
        else:
            side = row["home_team"] if lean > 0 else row["away_team"]
            out.append(f"{DRIVER_FEATURES[best_feat]} → {side}")
    return out


def predict_week(season: int, week: int, train_start: int = 2010,
                 kind: str = "logistic") -> pd.DataFrame:
    seasons = list(range(train_start, season + 1))
    print(f"Loading {seasons[0]}-{seasons[-1]} (schedules, Elo, EPA, Madden) ...")
    games = data.load_games(seasons, include_unplayed=True)
    elo_table = compute_elo(games)

    # Play-by-play / snap tables only exist for seasons with games played; a
    # not-yet-started season has no pbp, so build those tables on played seasons
    # and let carry-forward project the latest starters onto the future slate.
    played_seasons = sorted(
        int(s) for s in games.loc[games["home_score"].notna(), "season"].unique()
    )
    pbp_seasons = [s for s in seasons if s in played_seasons]
    epa_table = epa_mod.team_game_epa(pbp_seasons)
    qb_table = qb_mod.starting_qb_ovr(pbp_seasons)
    starter_table = starters_mod.starter_unit_ovr(pbp_seasons)

    # Project starters onto not-yet-played games (carry each team's last known
    # starter OVR forward); no-op when the slate already has play-by-play.
    target_games = games[(games["season"] == season) & (games["week"] == week)]
    qb_table = _carry_forward(qb_table, ["qb_ovr"], games, target_games)
    starter_table = _carry_forward(
        starter_table, ["ol_ovr", "dl_ovr", "db_ovr", "starter_ovr"],
        games, target_games,
    )

    df, cols = build_features(
        games, epa_table=epa_table, elo_table=elo_table,
        qb_table=qb_table, starter_table=starter_table,
    )

    target = df[(df["season"] == season) & (df["week"] == week)].copy()
    if target.empty:
        raise SystemExit(f"No games found for {season} week {week}.")

    # Validated setup: train on every season strictly before the target season.
    train_df = df[df["season"] < season]
    pipe = model.train(train_df, cols, kind=kind, calibrate="isotonic")

    target["model_home_prob"] = pipe.predict_proba(target[cols])[:, 1]
    target["market_home_prob"] = market_home_prob(target).to_numpy()
    target["edge"] = target["model_home_prob"] - target["market_home_prob"]
    target["driver"] = _drivers(target)
    return target.sort_values("edge", key=lambda s: s.abs(), ascending=False)


def render(target: pd.DataFrame, season: int, week: int) -> str:
    lines = [f"# NFL model — {season} Week {week}", ""]
    graded = target["home_win"].notna().any()

    # Top-line: the biggest model-vs-market disagreements.
    big = target.head(3)
    flags = []
    for _, r in big.iterrows():
        side = r["home_team"] if r["edge"] > 0 else r["away_team"]
        flags.append(f"{side} (+{abs(r['edge']):.0%} vs market)")
    lines.append("**Biggest model-vs-market disagreements:** " + ", ".join(flags))
    lines.append("")

    headers = ["Matchup", "Model", "Market", "Edge", "Key driver"]
    if graded:
        headers.append("Result")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for _, r in target.iterrows():
        matchup = f"{r['away_team']} @ {r['home_team']}"
        model_s = _prob_str(r["home_team"], r["away_team"], r["model_home_prob"])
        mkt_s = _prob_str(r["home_team"], r["away_team"], r["market_home_prob"])
        side = r["home_team"] if r["edge"] > 0 else r["away_team"]
        edge_s = f"{side} +{abs(r['edge']):.0%}"
        row = f"| {matchup} | {model_s} | {mkt_s} | {edge_s} | {r['driver']} |"
        if graded:
            if pd.isna(r["home_win"]):
                res = "—"
            else:
                winner = r["home_team"] if r["home_win"] == 1 else r["away_team"]
                model_pick = r["home_team"] if r["model_home_prob"] >= 0.5 else r["away_team"]
                mark = "✓" if winner == model_pick else "✗"
                res = f"{winner} {mark}"
            row += f" {res} |"
        lines.append(row)

    if graded:
        played = target[target["home_win"].notna()]
        if len(played):
            picks = (played["model_home_prob"] >= 0.5).astype(int)
            acc = (picks == played["home_win"]).mean()
            mkt_picks = (played["market_home_prob"] >= 0.5).astype(int)
            mkt_acc = (mkt_picks == played["home_win"]).mean()
            lines += ["", f"_Model went {acc:.0%} straight-up "
                      f"({len(played)} games); market {mkt_acc:.0%}._"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly NFL win-probability preview")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--model", choices=["logistic", "gbm"], default="logistic",
                    help="logistic = saner tail probabilities (preview default); "
                         "gbm = marginally better aggregate calibration, uglier tails")
    ap.add_argument("--out", default=None, help="also write the markdown to this path")
    args = ap.parse_args()

    target = predict_week(args.season, args.week, args.train_start, args.model)
    report = render(target, args.season, args.week)
    print("\n" + report)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
