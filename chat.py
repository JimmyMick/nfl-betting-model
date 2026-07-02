"""Ask natural-language questions about the model's current view — CLI entry.

Thin wrapper over ``nfl_betting_model.chat``. Answers are grounded in the
artifacts the weekly pipeline exports to ``predictions/cloud/`` (the current
model-vs-market preview, the season grade, the pick'em leaderboard), so this
runs with no training and no live data fetch.

Examples
--------
    uv run chat.py "who does the model like most this week and why?"
    uv run chat.py                       # interactive REPL
    LLM_PROVIDER=openai uv run chat.py "how has the model done vs Vegas?"
"""

from nfl_betting_model.chat import main

if __name__ == "__main__":
    main()
