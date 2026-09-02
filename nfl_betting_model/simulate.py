"""Monte Carlo simulation of the rest of an NFL season.

Uses the trained win-probability model to assign a home-win probability to every
remaining regular-season game, then rolls the season forward thousands of times.
Each simulation completes the standings, seeds the playoffs (4 division winners +
3 wildcards per conference), and plays out the bracket to a Super Bowl champion.
Aggregated across sims this yields, per team: projected final wins and the
probability of making the playoffs / winning the division / earning the #1 seed /
winning the conference / winning it all.

A *fun* projection, not a betting product. Approximations (documented):
  * Team ratings are held fixed across a simulation (no in-sim Elo updating).
  * Playoff-game probabilities come from current Elo + home-field for the higher
    seed (the regular-season market/model line isn't available for games that
    aren't scheduled).
  * Seeding tiebreakers use record, then division wins, then conference wins,
    then a coin flip — a close approximation of the NFL's deeper tiebreakers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import availability as avail_mod, data, epa as epa_mod, model, qb as qb_mod
from .elo import HFA, compute_elo
from .features import FORM_WINDOW, build_features

# Stable 2024+ division alignment: team -> (conference, division).
DIVISIONS: dict[str, tuple[str, str]] = {}
for _conf, _divs in {
    "AFC": {"East": ["BUF", "MIA", "NE", "NYJ"], "North": ["BAL", "CIN", "CLE", "PIT"],
            "South": ["HOU", "IND", "JAX", "TEN"], "West": ["DEN", "KC", "LV", "LAC"]},
    "NFC": {"East": ["DAL", "NYG", "PHI", "WAS"], "North": ["CHI", "DET", "GB", "MIN"],
            "South": ["ATL", "CAR", "NO", "TB"], "West": ["ARI", "LA", "SF", "SEA"]},
}.items():
    for _d, _teams in _divs.items():
        for _t in _teams:
            DIVISIONS[_t] = (_conf, f"{_conf} {_d}")


def _current_state(games: pd.DataFrame, epa_table: pd.DataFrame | None,
                   season: int) -> dict[str, dict]:
    """Each team's *current* rolling form/EPA (last ``FORM_WINDOW`` played games)
    plus its season-to-date win rate. Frozen and carried onto future games — the
    rolling features are otherwise undefined for weeks with no played games in
    their window (which is why build_features drops them)."""
    long = data.to_long(games)
    if epa_table is not None:
        long = long.merge(epa_table, on=["game_id", "team"], how="left")
    played = long[long["points_for"].notna()].sort_values(["team", "gameday"])
    state: dict[str, dict] = {}
    for team, grp in played.groupby("team"):
        last5 = grp.tail(FORM_WINDOW)
        seas = grp[grp["season"] == season]
        s = {
            "form_pf": last5["points_for"].mean(),
            "form_pa": last5["points_against"].mean(),
            "form_winrate": last5["won"].mean(),
            "off_epa": last5["off_epa"].mean() if "off_epa" in last5 else np.nan,
            "def_epa": last5["def_epa"].mean() if "def_epa" in last5 else np.nan,
            "season_winrate": seas["won"].mean() if len(seas) else np.nan,
        }
        s["form_margin"] = s["form_pf"] - s["form_pa"]
        state[team] = s
    return state


def game_probs(season: int, train_start: int = 2010):
    """Return ``(reg_df, elo_latest)``: this season's regular-season games with a
    model ``p_home`` for the unplayed ones, plus each team's current Elo rating.

    Unplayed games get their probability from the model applied to *frozen*
    current team state (form/EPA carried forward, current Elo per matchup, the
    primary starter's QB OVR, and a healthy injury report), so every remaining
    week is covered — not just the near ones build_features can roll form for.
    """
    from predict import _carry_forward  # reuse the primary-starter carry-forward

    seasons = list(range(train_start, season + 1))
    games = data.load_games(seasons, include_unplayed=True)
    elo_table = compute_elo(games)

    played_seasons = sorted(
        int(s) for s in games.loc[games["home_score"].notna(), "season"].unique())
    pbp_seasons = [s for s in seasons if s in played_seasons]
    epa_table = epa_mod.team_game_epa(pbp_seasons)
    qb_table = qb_mod.starting_qb_ovr(pbp_seasons)

    reg = games[(games["season"] == season) & (games["game_type"] == "REG")].copy()
    remaining = reg[reg["home_win"].isna()]
    qb_table = _carry_forward(qb_table, ["qb_ovr"], games, remaining)
    avail_table = avail_mod.team_out_talent(seasons)

    # Train on completed history exactly as the live model does.
    df, cols = build_features(games, epa_table=epa_table, elo_table=elo_table,
                              qb_table=qb_table, avail_table=avail_table)
    pipe = model.train(df[df["season"] < season], cols, kind="logistic",
                       calibrate="sigmoid")

    # Build the remaining-game feature matrix from frozen current team state.
    reg = reg.join(elo_table[["elo_diff", "elo_prob"]])
    rem = reg[reg["home_win"].isna()].copy()
    state = _current_state(games, epa_table, season)
    qb_key = qb_table.set_index(["game_id", "team"])["qb_ovr"]

    def sdiff(key: str) -> np.ndarray:
        h = rem["home_team"].map(lambda t: state.get(t, {}).get(key, np.nan))
        a = rem["away_team"].map(lambda t: state.get(t, {}).get(key, np.nan))
        return (h - a).to_numpy()

    def net(t: str) -> float:
        s = state.get(t, {})
        return s.get("off_epa", np.nan) - s.get("def_epa", np.nan)

    feat = pd.DataFrame(index=rem.index)
    feat["form_pf_diff"] = sdiff("form_pf")
    feat["form_pa_diff"] = sdiff("form_pa")
    feat["form_margin_diff"] = sdiff("form_margin")
    feat["form_winrate_diff"] = sdiff("form_winrate")
    feat["season_winrate_diff"] = sdiff("season_winrate")
    feat["rest_diff"] = rem["home_rest"].fillna(7) - rem["away_rest"].fillna(7)
    feat["div_game"] = rem["div_game"].fillna(0).astype(float)
    feat["off_epa_diff"] = sdiff("off_epa")
    feat["def_epa_diff"] = sdiff("def_epa")
    feat["net_epa_diff"] = (rem["home_team"].map(net) - rem["away_team"].map(net)).to_numpy()
    feat["elo_diff"] = rem["elo_diff"].to_numpy()
    feat["elo_prob"] = rem["elo_prob"].to_numpy()
    feat["qb_ovr_diff"] = [qb_key.get((gid, h), np.nan) - qb_key.get((gid, a), np.nan)
                           for gid, h, a in zip(rem["game_id"], rem["home_team"],
                                                rem["away_team"])]
    feat["out_avail_diff"] = 0.0

    reg["p_home"] = np.nan
    reg.loc[rem.index, "p_home"] = pipe.predict_proba(feat[cols])[:, 1]

    # Current rating = the pre-game Elo carried onto a future game.
    elo_latest: dict[str, float] = {}
    es = games[["season", "home_team", "away_team", "home_score"]].join(
        elo_table[["home_elo_pre", "away_elo_pre"]])
    es = es[es["season"] == season]
    fut = es[es["home_score"].isna()]
    for _, g in (fut if not fut.empty else es).iterrows():
        elo_latest.setdefault(g["home_team"], g["home_elo_pre"])
        elo_latest.setdefault(g["away_team"], g["away_elo_pre"])
    return reg, elo_latest


def _elo_prob(elo_a: np.ndarray, elo_b: np.ndarray, host_a: bool) -> np.ndarray:
    """P(team A beats team B); A hosts unless ``host_a`` is False (neutral)."""
    edge = (elo_a + (HFA if host_a else 0.0)) - elo_b
    return 1.0 / (1.0 + 10.0 ** (-edge / 400.0))


def simulate(season: int, n: int = 10000, train_start: int = 2010, seed: int = 0):
    """Run the season ``n`` times; return a per-team projection DataFrame."""
    reg_df, elo = game_probs(season, train_start)
    teams = sorted(elo)
    T = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    is_afc = np.array([DIVISIONS[t][0] == "AFC" for t in teams])
    div_name = np.array([DIVISIONS[t][1] for t in teams])
    elo_arr = np.array([elo[t] for t in teams])
    rng = np.random.default_rng(seed)

    played = reg_df[reg_df["home_win"].notna()]
    remaining = reg_df[reg_df["home_win"].isna()]

    # Base tallies from games already played.
    base_w = np.zeros(T); base_dw = np.zeros(T); base_cw = np.zeros(T)
    for _, g in played.iterrows():
        h, a = idx[g["home_team"]], idx[g["away_team"]]
        w, l = (h, a) if g["home_win"] == 1 else (a, h)
        base_w[w] += 1
        if div_name[h] == div_name[a]:
            base_dw[w] += 1
        if is_afc[h] == is_afc[a]:
            base_cw[w] += 1

    # Simulate remaining games: outcome matrix (n_sims x n_games).
    rh = remaining["home_team"].map(idx).to_numpy()
    ra = remaining["away_team"].map(idx).to_numpy()
    p = remaining["p_home"].to_numpy()
    same_div = div_name[rh] == div_name[ra]
    same_conf = is_afc[rh] == is_afc[ra]
    G = len(remaining)
    home_wins = rng.random((n, G)) < p

    wins = np.tile(base_w, (n, 1)).astype(float)
    dwins = np.tile(base_dw, (n, 1)).astype(float)
    cwins = np.tile(base_cw, (n, 1)).astype(float)
    for gi in range(G):
        h, a, hw = rh[gi], ra[gi], home_wins[:, gi]
        aw = ~hw
        wins[:, h] += hw; wins[:, a] += aw
        if same_div[gi]:
            dwins[:, h] += hw; dwins[:, a] += aw
        if same_conf[gi]:
            cwins[:, h] += hw; cwins[:, a] += aw

    # Seeding score: wins, then division wins, then conference wins, then a
    # per-sim coin flip so exact ties resolve fairly.
    score = wins + 1e-3 * dwins + 1e-6 * cwins + rng.random((n, T)) * 1e-9

    made = np.zeros(T); won_div = np.zeros(T); top_seed = np.zeros(T)
    conf_champ = np.zeros(T); sb_champ = np.zeros(T)
    champ_by_conf = {}

    for afc in (True, False):
        cols = np.where(is_afc == afc)[0]              # 16 team indices
        sc = score[:, cols]
        local_div = div_name[cols]
        is_dw = np.zeros((n, len(cols)), bool)
        for d in np.unique(local_div):
            sub = np.where(local_div == d)[0]
            best = sub[np.argmax(sc[:, sub], axis=1)]
            is_dw[np.arange(n), best] = True
        # Division winners rank above all wildcards, each block by score.
        order = np.argsort(-(sc + 1e6 * is_dw), axis=1)
        seed_team = cols[order]                        # (n,16) global idx by seed
        seven = seed_team[:, :7]

        for k in range(7):
            np.add.at(made, seven[:, k], 1)
        # Seeds 1-4 are the division winners by construction (the 1e6 boost sorts
        # every division winner above every wildcard).
        np.add.at(won_div, seed_team[:, :4].ravel(), 1)
        np.add.at(top_seed, seed_team[:, 0], 1)

        # Bracket by seed slot (0=seed1 ... 6=seed7); higher seed (lower idx) hosts.
        elo_seed = elo_arr[seven]                      # (n,7)

        def play(a, b, neutral=False):
            ea = elo_seed[np.arange(n), a]; eb = elo_seed[np.arange(n), b]
            a_host = _elo_prob(ea, eb, host_a=not neutral)
            return np.where(rng.random(n) < a_host, a, b)

        w1 = play(np.full(n, 1), np.full(n, 6))        # 2 vs 7
        w2 = play(np.full(n, 2), np.full(n, 5))        # 3 vs 6
        w3 = play(np.full(n, 3), np.full(n, 4))        # 4 vs 5
        div_field = np.sort(np.stack([np.zeros(n, int), w1, w2, w3], axis=1), axis=1)
        d1 = play(div_field[:, 0], div_field[:, 3])    # 1 vs lowest
        d2 = play(div_field[:, 1], div_field[:, 2])    # middle two
        cf = np.stack([d1, d2], axis=1)
        host = cf.min(axis=1); opp = cf.max(axis=1)
        champ_seed = play(host, opp)
        champ_team = seven[np.arange(n), champ_seed]
        np.add.at(conf_champ, champ_team, 1)
        champ_by_conf[afc] = (champ_team, elo_arr[champ_team])

    # Super Bowl (neutral site).
    (a_team, a_elo), (n_team, n_elo) = champ_by_conf[True], champ_by_conf[False]
    a_win = rng.random(n) < _elo_prob(a_elo, n_elo, host_a=False)
    sb_teams = np.where(a_win, a_team, n_team)
    np.add.at(sb_champ, sb_teams, 1)

    out = pd.DataFrame({
        "team": teams,
        "conference": [DIVISIONS[t][0] for t in teams],
        "division": [DIVISIONS[t][1] for t in teams],
        "wins_now": base_w.astype(int),
        "games_played": [int((played["home_team"] == t).sum()
                             + (played["away_team"] == t).sum()) for t in teams],
        "proj_wins": wins.mean(axis=0).round(1),
        "make_playoffs": made / n,
        "win_division": won_div / n,
        "top_seed": top_seed / n,
        "win_conference": conf_champ / n,
        "win_sb": sb_champ / n,
    })
    return out.sort_values(["win_sb", "proj_wins"], ascending=False).reset_index(drop=True)
