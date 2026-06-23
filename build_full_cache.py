"""Resumable per-season cache of the QB table (the pbp-heavy input to the FULL
model) so the availability-upgrade FULL validation runs without re-loading
play-by-play. Starters are snap-based and loaded live (lighter). Mirrors
build_epa_cache.py.

Run (repeatedly if killed):  ./.venv/bin/python build_full_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from nfl_betting_model import qb as qb_mod

CACHE = Path("data/full_cache")
CACHE.mkdir(parents=True, exist_ok=True)
SEASONS = list(range(2010, 2026))

done = 0
for s in SEASONS:
    qbp = CACHE / f"qb_{s}.parquet"
    if qbp.exists():
        continue
    print(f"building qb {s} ...", flush=True)
    qb_mod.starting_qb_ovr([s]).to_parquet(qbp)
    print(f"  cached {s}", flush=True)
    done += 1

missing = [s for s in SEASONS if not (CACHE / f"qb_{s}.parquet").exists()]
print(f"\nbuilt {done} new | missing {missing}", flush=True)
sys.exit(0 if not missing else 1)
