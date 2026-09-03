"""Static NFL team metadata (name, conference, division, ESPN logo URL).

Baked from nflverse (nflreadpy.load_teams) so the key-free cloud app needs no
network fetch; logo URLs are ESPN CDN links the browser loads directly.
Regenerate with scripts embedded in git history if the league realigns.
"""

from __future__ import annotations

TEAMS: dict[str, dict] = {
    'ARI': {"name": 'Arizona Cardinals', "conf": 'NFC', "division": 'NFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png'},
    'ATL': {"name": 'Atlanta Falcons', "conf": 'NFC', "division": 'NFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png'},
    'BAL': {"name": 'Baltimore Ravens', "conf": 'AFC', "division": 'AFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png'},
    'BUF': {"name": 'Buffalo Bills', "conf": 'AFC', "division": 'AFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png'},
    'CAR': {"name": 'Carolina Panthers', "conf": 'NFC', "division": 'NFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500-dark/car.png'},
    'CHI': {"name": 'Chicago Bears', "conf": 'NFC', "division": 'NFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png'},
    'CIN': {"name": 'Cincinnati Bengals', "conf": 'AFC', "division": 'AFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png'},
    'CLE': {"name": 'Cleveland Browns', "conf": 'AFC', "division": 'AFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png'},
    'DAL': {"name": 'Dallas Cowboys', "conf": 'NFC', "division": 'NFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png'},
    'DEN': {"name": 'Denver Broncos', "conf": 'AFC', "division": 'AFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png'},
    'DET': {"name": 'Detroit Lions', "conf": 'NFC', "division": 'NFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png'},
    'GB': {"name": 'Green Bay Packers', "conf": 'NFC', "division": 'NFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png'},
    'HOU': {"name": 'Houston Texans', "conf": 'AFC', "division": 'AFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png'},
    'IND': {"name": 'Indianapolis Colts', "conf": 'AFC', "division": 'AFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png'},
    'JAX': {"name": 'Jacksonville Jaguars', "conf": 'AFC', "division": 'AFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png'},
    'KC': {"name": 'Kansas City Chiefs', "conf": 'AFC', "division": 'AFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png'},
    'LA': {"name": 'Los Angeles Rams', "conf": 'NFC', "division": 'NFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png'},
    'LAC': {"name": 'Los Angeles Chargers', "conf": 'AFC', "division": 'AFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png'},
    'LV': {"name": 'Las Vegas Raiders', "conf": 'AFC', "division": 'AFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png'},
    'MIA': {"name": 'Miami Dolphins', "conf": 'AFC', "division": 'AFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png'},
    'MIN': {"name": 'Minnesota Vikings', "conf": 'NFC', "division": 'NFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png'},
    'NE': {"name": 'New England Patriots', "conf": 'AFC', "division": 'AFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png'},
    'NO': {"name": 'New Orleans Saints', "conf": 'NFC', "division": 'NFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png'},
    'NYG': {"name": 'New York Giants', "conf": 'NFC', "division": 'NFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png'},
    'NYJ': {"name": 'New York Jets', "conf": 'AFC', "division": 'AFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png'},
    'PHI': {"name": 'Philadelphia Eagles', "conf": 'NFC', "division": 'NFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png'},
    'PIT': {"name": 'Pittsburgh Steelers', "conf": 'AFC', "division": 'AFC North', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png'},
    'SEA': {"name": 'Seattle Seahawks', "conf": 'NFC', "division": 'NFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png'},
    'SF': {"name": 'San Francisco 49ers', "conf": 'NFC', "division": 'NFC West', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png'},
    'TB': {"name": 'Tampa Bay Buccaneers', "conf": 'NFC', "division": 'NFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png'},
    'TEN': {"name": 'Tennessee Titans', "conf": 'AFC', "division": 'AFC South', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png'},
    'WAS': {"name": 'Washington Commanders', "conf": 'NFC', "division": 'NFC East', "logo": 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png'},
}

def logo(abbr: str) -> str | None:
    """ESPN logo URL for a team abbreviation (None if unknown)."""
    return TEAMS.get(abbr, {}).get("logo")

def name(abbr: str) -> str:
    """Full team name for an abbreviation (falls back to the abbr)."""
    return TEAMS.get(abbr, {}).get("name", abbr)
