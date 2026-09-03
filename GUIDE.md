# 🏈 Urban Platform Zone Pick'em — Quick Guide

Every week you pick NFL winners against three opponents: your buddies, a **stats
model**, and an **AI**. Beat all three for bragging rights till February. Lose to
the computer and… we'll be gentle.

## 🔑 Getting in
Hit **"Log in with Descope"** with the email you gave Jim. Locked out? That's not
a bug, that's Jim — go bug him back.

## 🗂️ The tabs (in order)
| Tab | What it's for |
|---|---|
| **📝 Blog** | Jim's notes — what's new in the app, the season, fun stats. Start here. |
| **Weekly preview** | The model's read on this week's games (+ the AI's picks). |
| **Make picks** | Lock in your winners and confidence each week. |
| **Season tracker** | How the model's doing vs Vegas, week by week. |
| **🗓️ Schedule** | The full-season slate, with results as games finish. |
| **🏆 Playoff odds** | A Monte Carlo sim of the rest of the season. |
| **Pick'em leaderboard** | Standings. Who's hot, who's not. |
| **📈 Paper play** | A fake-money experiment betting one game a week. |
| **📖 Guide** | You're here. |

---

## 📝 Blog
Jim's spot to post notes to the group — app updates, season takes, historical
stats worth chewing on. Newest post sits at the top. Read-only for the crew;
Jim publishes from his machine.

## 📊 Weekly preview (read it, don't worship it)
The model's game-by-game read for the upcoming week.
- **Model** — the model's win % for its pick, e.g. `BUF 74%`.
- **Market** — what Vegas implies.
- **Edge** — the side the model values **more than Vegas prices it**, and by how
  much. A *mispricing*, not a winner pick — and **not a tip**. Heads up: it can
  name the **underdog even when the model still thinks the favorite wins** — that
  just means the model has the favorite lower than Vegas does (e.g. model has the
  Chargers 75% to win but Vegas says 82%, so the leftover value shows as
  "ARI +7%"). It's about *price*, not *who wins*.
- **Key driver** — the main reason (QB, talent, form, injuries…).
- **🤖 AI expert** — the robot's picks and reasoning show here **before kickoff**.
  (Your picks stay secret — only the bot shows its cards early.)

> ⚠️ **A big edge is NOT a betting tip.** It usually means the *model* is wrong,
> not Vegas — we tested it, betting these "edges" loses money. It's a sharp second
> opinion, not a crystal ball, and it aims to be *as good as* Vegas, not better.
> (That's already hard.)

## ✅ Make picks (do this every week)
1. Open **Make picks**.
2. Click the team you think **wins** each game.
3. Set **confidence, 50–100**: **50** = pure coin flip, **100** = bet-the-house lock.
4. Hit **Submit my picks.**

Change your mind? Re-submit anytime **before kickoff** — your last answer counts.
(Yes, you can stall for the injury report. We see you.)

> **Confidence isn't decoration.** You're graded on *wins* **and** *honesty*, so a
> confident-but-wrong 100 stings more than a careful 60. Slapping 100 on
> everything is a great way to finish last.

## 📈 Season tracker
The model's own report card, week by week: its record vs the Vegas favorite, plus
a **calibration** grade (does an 80% call actually win ~80% of the time?). This is
about the *model* — your standings live on the leaderboard.

## 🗓️ Schedule
The full-season matchup grid. Pick a week (defaults to the next unplayed one) to
see who's playing, when, and — once games finish — the final scores and winners.
Team helmets included, because we're fancy now.

## 🏆 Playoff odds
A **Monte Carlo** simulation: the model plays out the entire rest of the season
thousands of times, then tallies how often each team makes the playoffs, wins its
division, grabs the #1 seed, wins its conference, and wins the Super Bowl. Refreshed
every Tuesday, so you can watch a team's odds swing week to week. **Just for fun —
not a betting product.**

## 🏆 Pick'em leaderboard
Where the bragging rights live.
- **Record** — your W-L and win %.
- **vs Model** — *the money stat.* Your accuracy minus the model's on the same
  games. **Green = you're beating the computer.** Screenshot it. Frame it.
- **Brier / Log loss** — your **calibration** grade (lower = better). Translation:
  when you say 80%, are you right about 80% of the time? Honest forecasters win
  here; chronic over- or under-confidence gets punished.
- After a week's graded, open the **"🤖 why it picked"** drawer to see the AI's
  logic next to the ✓/✗.

## 📈 Paper play (the science experiment)
Betting **all** of the model's edges loses money — we tested it. But digging in, we
found one weird exception: if each week you bet **only the single biggest
disagreement** on the board (and take the *model's* side, usually an underdog), the
last 10 years of history would've actually **made money** (+23%). Could be a real
edge. Could be dumb luck the past happened to reward — no way to know from old data,
since the model was built looking at those same years.

So this tab is the honest tiebreaker: **track it live, going forward, on games
nobody's seen yet.**

- Every week the model's **one biggest disagreement** gets a pretend **10-unit**
  bet on the model's side, at that week's actual odds.
- **It's Monopoly money.** Nobody's wagering a dime. This is a scoreboard, not a
  tip sheet.
- What you'll see: the running **record**, **profit** (in units), **ROI**, a
  chart of the bankroll over the season, and every week's play with the odds and
  a ✓/✗.

If it keeps printing, maybe we found something. If it face-plants, we proved it
was luck — and that's a *useful* answer too. Either way: **do not bet your rent
on this.** 🧪

## 🤖 The AI expert
There's a robot in the pool. It picks **blind** — never sees the model or the
line — reasoning from raw facts (records, form, injuries, weather) plus whatever
intel Jim slips it. It's coached to fade shaky favorites, not rubber-stamp them.
- **Before kickoff:** its picks **and reasoning** show in the **Weekly preview**.
- **After grading:** open the **"🤖 why it picked"** drawer on the leaderboard.

Smart bot, but it's reading the same tea leaves as everyone else. Beating it is
very doable. Don't lose to the robot.

## 💡 Tips
- **Submit early, tweak late.**
- **Be honest with confidence** — calibration wins seasons.
- **The model's a tool, not your boss.** The leaderboard rewards *your* read.

Good luck, and may your upset specials hit. 🍀
