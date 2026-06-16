"""Turn model probabilities into bets and measure ROI against the market.

Beating the market on accuracy is not the goal — finding *mispriced* games is.
For each test game we compare the model's win probability to the price actually
offered (implied probability *with* vig) and bet a side only when its expected
value is positive by at least ``ev_threshold``. Stakes are flat 1 unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def american_to_payout(odds: pd.Series | np.ndarray) -> np.ndarray:
    """Profit multiplier per 1 unit staked for American moneyline odds.

    +150 -> 1.5 profit on a win; -200 -> 0.5 profit on a win.
    """
    odds = pd.to_numeric(pd.Series(odds), errors="coerce").to_numpy(dtype=float)
    return np.where(odds < 0, 100.0 / np.abs(odds), odds / 100.0)


def american_to_implied(odds: pd.Series | np.ndarray) -> np.ndarray:
    """Implied win probability *including* vig (the price you must beat)."""
    odds = pd.to_numeric(pd.Series(odds), errors="coerce").to_numpy(dtype=float)
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


@dataclass
class BettingResult:
    ev_threshold: float
    n_games: int          # games with usable odds
    n_bets: int
    wins: int
    staked: float
    profit: float

    @property
    def roi(self) -> float:
        return self.profit / self.staked if self.staked else float("nan")

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_bets if self.n_bets else float("nan")

    @property
    def hit_pct(self) -> float:
        return self.n_bets / self.n_games if self.n_games else float("nan")

    def __str__(self) -> str:
        if not self.n_bets:
            return f"  EV>={self.ev_threshold:.0%}: no qualifying bets"
        return (
            f"  EV>={self.ev_threshold:>4.0%}: bets={self.n_bets:>3d} "
            f"({self.hit_pct:.0%} of slate)  win={self.win_rate:.3f}  "
            f"profit={self.profit:+.2f}u  ROI={self.roi:+.1%}"
        )


def evaluate_betting(
    test_df: pd.DataFrame,
    p_home: np.ndarray,
    ev_threshold: float = 0.0,
) -> BettingResult:
    """Flat-stake +EV betting on a single test slate.

    For each game, the side with the higher expected value is a candidate; we
    place a 1-unit bet only if that EV clears ``ev_threshold``.
    """
    y = test_df["home_win"].to_numpy()
    home_ml = test_df["home_moneyline"]
    away_ml = test_df["away_moneyline"]

    b_home = american_to_payout(home_ml)
    b_away = american_to_payout(away_ml)
    usable = ~(np.isnan(b_home) | np.isnan(b_away))

    p_away = 1.0 - p_home
    ev_home = p_home * b_home - (1.0 - p_home)   # = p*b - (1-p)
    ev_away = p_away * b_away - (1.0 - p_away)

    bet_home = ev_home >= ev_away                  # pick the better side
    side_ev = np.where(bet_home, ev_home, ev_away)
    side_b = np.where(bet_home, b_home, b_away)
    side_won = np.where(bet_home, y == 1, y == 0)

    place = usable & (side_ev >= ev_threshold)
    n_bets = int(place.sum())
    wins = int(side_won[place].sum())
    profit = float(np.where(side_won[place], side_b[place], -1.0).sum())

    return BettingResult(
        ev_threshold=ev_threshold,
        n_games=int(usable.sum()),
        n_bets=n_bets,
        wins=wins,
        staked=float(n_bets),
        profit=profit,
    )


def combine(results: list[BettingResult]) -> BettingResult:
    """Pool per-season results into one (e.g. for a multi-season backtest)."""
    if not results:
        raise ValueError("no results to combine")
    return BettingResult(
        ev_threshold=results[0].ev_threshold,
        n_games=sum(r.n_games for r in results),
        n_bets=sum(r.n_bets for r in results),
        wins=sum(r.wins for r in results),
        staked=sum(r.staked for r in results),
        profit=sum(r.profit for r in results),
    )


def betting_report(
    test_df: pd.DataFrame,
    p_home: np.ndarray,
    thresholds: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10),
) -> str:
    lines = ["  betting vs market price (flat 1u, +EV side):"]
    for t in thresholds:
        lines.append(str(evaluate_betting(test_df, p_home, ev_threshold=t)))
    return "\n".join(lines)
