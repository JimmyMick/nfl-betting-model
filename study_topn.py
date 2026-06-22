"""Throwaway study: how do the model's most-confident games do each week,
across multiple seasons? Computes top-1/3/5-by-confidence weekly records per
season, with the market's record on the same games for comparison.

Run:  uv run python study_topn.py
"""

from __future__ import annotations

import pandas as pd

from grade import grade_season

SEASONS = range(2019, 2025)  # 2019–2024
KIND = "logistic"


def topn(g: pd.DataFrame, n: int) -> pd.DataFrame:
    g = g.assign(conf=g["model_home_prob"].apply(lambda p: max(p, 1 - p)))
    return pd.concat(
        grp.sort_values("conf", ascending=False).head(n)
        for _, grp in g.groupby("week")
    )


def rec(c: pd.Series) -> str:
    w, n = int(c.sum()), len(c)
    return f"{w}-{n - w} ({w / n:.1%})" if n else "—"


rows = []
pooled = {1: [], 3: [], 5: []}
for season in SEASONS:
    try:
        g = grade_season(season, None, 2010, KIND)
    except Exception as e:  # noqa: BLE001
        print(f"{season}: skipped ({e})")
        continue
    row = {"Season": season, "Games": len(g),
           "All model": f"{g['model_correct'].mean():.1%}"}
    for n in (1, 3, 5):
        t = topn(g, n)
        pooled[n].append(t)
        row[f"Top{n} model"] = rec(t["model_correct"])
        row[f"Top{n} market"] = rec(t["market_correct"])
    rows.append(row)
    print(f"done {season}: {len(g)} games")

df = pd.DataFrame(rows)
print("\n=== Per season ===")
print(df.to_string(index=False))

print("\n=== Pooled 2019–2024 ===")
for n in (1, 3, 5):
    if pooled[n]:
        alln = pd.concat(pooled[n])
        print(f"Top {n}/wk: model {rec(alln['model_correct'])} | "
              f"market {rec(alln['market_correct'])} | n={len(alln)}")
