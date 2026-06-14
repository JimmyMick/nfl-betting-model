"""Train and evaluate the moneyline win-probability model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as feat


def build_pipeline() -> Pipeline:
    """Impute, standardize, then fit L2-regularized logistic regression.

    Median imputation handles early-season rows that lack a season-to-date
    win rate, so we keep every game rather than dropping it.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )


@dataclass
class Evaluation:
    n: int
    accuracy: float
    log_loss: float
    brier: float
    auc: float
    # Market benchmark on the same games (NaN if odds missing).
    market_accuracy: float
    market_log_loss: float
    market_brier: float

    def __str__(self) -> str:
        return (
            f"games={self.n}\n"
            f"  model  : acc={self.accuracy:.3f}  logloss={self.log_loss:.3f}  "
            f"brier={self.brier:.3f}  auc={self.auc:.3f}\n"
            f"  market : acc={self.market_accuracy:.3f}  "
            f"logloss={self.market_log_loss:.3f}  brier={self.market_brier:.3f}"
        )


def time_split(df: pd.DataFrame, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on every season before ``test_season``, test on that season."""
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]
    return train, test


def train(train_df: pd.DataFrame) -> Pipeline:
    pipe = build_pipeline()
    pipe.fit(train_df[feat.FEATURE_COLS], train_df["home_win"])
    return pipe


def evaluate(pipe: Pipeline, test_df: pd.DataFrame) -> Evaluation:
    y = test_df["home_win"].to_numpy()
    p = pipe.predict_proba(test_df[feat.FEATURE_COLS])[:, 1]

    # Market benchmark where moneyline is available.
    mkt = feat.market_home_prob(test_df).to_numpy()
    mask = ~np.isnan(mkt)
    if mask.sum() > 0:
        m_acc = accuracy_score(y[mask], (mkt[mask] >= 0.5).astype(int))
        m_ll = log_loss(y[mask], mkt[mask], labels=[0, 1])
        m_br = brier_score_loss(y[mask], mkt[mask])
    else:
        m_acc = m_ll = m_br = float("nan")

    return Evaluation(
        n=len(test_df),
        accuracy=accuracy_score(y, (p >= 0.5).astype(int)),
        log_loss=log_loss(y, p, labels=[0, 1]),
        brier=brier_score_loss(y, p),
        auc=roc_auc_score(y, p),
        market_accuracy=m_acc,
        market_log_loss=m_ll,
        market_brier=m_br,
    )
