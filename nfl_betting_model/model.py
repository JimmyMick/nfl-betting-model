"""Train and evaluate the moneyline win-probability model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as feat


def build_pipeline(kind: str = "logistic") -> Pipeline:
    """Return a fitted-ready pipeline.

    ``logistic`` — median impute + standardize + L2 logistic regression.
    ``gbm`` — HistGradientBoosting (handles NaNs and interactions natively).
    """
    if kind == "gbm":
        return Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        early_stopping=True,
                        random_state=0,
                    ),
                ),
            ]
        )
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


def train(train_df: pd.DataFrame, feature_cols: list[str], kind: str = "logistic") -> Pipeline:
    pipe = build_pipeline(kind)
    pipe.fit(train_df[feature_cols], train_df["home_win"])
    return pipe


def evaluate(pipe: Pipeline, test_df: pd.DataFrame, feature_cols: list[str]) -> Evaluation:
    y = test_df["home_win"].to_numpy()
    p = pipe.predict_proba(test_df[feature_cols])[:, 1]

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
